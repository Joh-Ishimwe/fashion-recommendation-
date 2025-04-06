# src/api.py
import csv
from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
import os
import logging

import pandas as pd
from data import load_from_mongo, upload_to_mongo
from preprocess import preprocess_data, preprocess_new_data
from data import get_mongo_collection
from train import train_and_evaluate
from predict import make_predictions
from utils import increment_model_version, get_model_version, load_metrics, save_metrics
from sklearn.model_selection import train_test_split
from pydantic import BaseModel

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

class FashionItem(BaseModel):
    gender: str
    masterCategory: str
    subCategory: str
    articleType: str
    baseColour: str
    season: str
    year: int

def initialize_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Model artifacts not found. Training the model...")
        try:
            df = load_from_mongo()
            if df.empty:
                logger.warning("No data in MongoDB. Upload data first.")
                return False
            
            sample_size = min(10000, len(df))
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            X, y, feature_columns, scaler, le, encoders = preprocess_data(df, encoder_dir="models/", apply_smote=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            best_model = train_and_evaluate(
                X_train, y_train, feature_columns, le,
                data_dir="data/", model_dir="models/",
                simplify_training=True,
                le_classes=le.classes_
            )
            
            os.makedirs("models", exist_ok=True)
            os.makedirs("data", exist_ok=True)  # Ensure data directory exists
            joblib.dump(best_model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
            joblib.dump(le, LABEL_ENCODER_PATH)
            for col, encoder in encoders.items():
                joblib.dump(encoder, f"models/{col}_encoder.pkl")
            
            app.state.model = best_model
            app.state.scaler = scaler
            app.state.feature_columns = feature_columns
            app.state.le = le
            app.state.encoders = encoders
            logger.info("Model trained and loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}", exc_info=True)
            return False

    else:
        # Load existing artifacts
        app.state.model = joblib.load(MODEL_PATH)
        app.state.scaler = joblib.load(SCALER_PATH)
        app.state.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        app.state.le = joblib.load(LABEL_ENCODER_PATH)
        app.state.encoders = {
            col: joblib.load(f"models/{col}_encoder.pkl")
            for col in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
        }
        logger.info("Model artifacts loaded successfully")
        return True

# Initialize model at startup
try:
    if not initialize_model():
        logger.error("Model initialization failed. Ensure MongoDB has data.")
except Exception as e:
    logger.error(f"Model artifacts not found: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail="Model artifacts not found. Please train the model first.")


@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion Recommendation API"}

@app.get("/fashion_items/")
async def get_fashion_items(gender: str = None, limit: int = 10):
    try:
        logger.info(f"Fetching fashion items with gender={gender}, limit={limit}")
        collection = get_mongo_collection()
        query = {"gender": {"$regex": f"^{gender}$", "$options": "i"}} if gender else {}
        items = list(collection.find(query).limit(limit))
        for item in items:
            item["_id"] = str(item["_id"])
        logger.info(f"Retrieved {len(items)} items")
        return {"items": items}
    except Exception as e:
        logger.error(f"Error in get_fashion_items: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/predict/")
async def predict_usage(item: FashionItem):
    """Predict the usage category for a fashion item."""
    try:
        logger.info(f"Received prediction request: {item.model_dump()}")
        raw_columns = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year"]
        input_data = pd.DataFrame([item.model_dump()], columns=raw_columns)
        X_new_scaled = preprocess_new_data(input_data, app.state.feature_columns, app.state.scaler, encoder_dir=os.path.dirname(MODEL_PATH))
        y_pred = app.state.model.predict(X_new_scaled)
        predicted_usage = app.state.le.inverse_transform(y_pred)[0]
        return {"predicted_usage": predicted_usage}
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/upload_data/")
async def upload_data(file: UploadFile = File(...)):
    """Upload a CSV file to MongoDB for retraining."""
    try:
        # Log the filename for debugging
        logger.info(f"Received file: {file.filename}")

        # Validate file type
        if not isinstance(file.filename, str) or not file.filename.lower().endswith('.csv'):
            logger.error("Invalid file type: File must be a CSV")
            raise HTTPException(status_code=400, detail="File must be a CSV")

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        logger.info(f"Saving temporary file to {temp_file_path}")
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()  
            temp_file.write(content)

        # Read the CSV file with error handling for malformed data
        logger.info("Reading CSV file...")
        try:
            df = pd.read_csv(temp_file_path, quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding='utf-8')
            logger.info(f"CSV read successfully. Shape: {df.shape}")
            logger.info(f"Columns in CSV: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

        # Validate required columns
        required_columns = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year", "usage"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in CSV: {missing_columns}")
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")

        # Upload to MongoDB
        logger.info("Uploading to MongoDB...")
        upload_to_mongo(df)
        logger.info("Upload to MongoDB completed.")
        
        # Remove the temporary file
        os.remove(temp_file_path)
        logger.info("Temporary file removed.")
        
        return {"message": f"Successfully uploaded {len(df)} records to MongoDB"}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")
    finally:
        await file.close()  


#retrain
@app.post("/retrain/")
async def retrain_model():
    """Retrain the model using data from MongoDB."""
    try:
        logger.info("Starting retrain process")
        df = load_from_mongo()

        # Load data from MongoDB
        logger.info("Loading data from MongoDB")
        df = load_from_mongo()
        if df.empty:
            logger.error("No data found in MongoDB for retraining")
            raise HTTPException(status_code=400, detail="No data found in MongoDB for retraining")
        logger.info(f"Loaded {len(df)} records from MongoDB")

        # Validate schema
        required_columns = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year", "usage"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns in MongoDB data: {missing_columns}")
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")

        # Sample data
        sample_size = min(10000, len(df))
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} records from {len(df)} for training")
            df = df.sample(n=sample_size, random_state=42)

        # Preprocess data with improved function
        logger.info("Preprocessing data")
        X, y, feature_columns, scaler, le, encoders = preprocess_data(
            df, encoder_dir=os.path.dirname(MODEL_PATH), apply_smote=False
        )
        logger.info(f"Label encoder classes: {le.classes_.tolist()}")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(f"Split data into train ({len(X_train)}) and test ({len(X_test)}) sets")

        # Save test set
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
        pd.DataFrame({"usage": y_test}).to_csv("data/y_test.csv", index=False)
        logger.info("Saved test data")

        # Train model (simplified for Render)
        logger.info("Starting model training")
        best_model = train_and_evaluate(
            X_train, y_train, feature_columns, le,
            data_dir="data/", model_dir="models/",
            simplify_training=True,
            le_classes=le.classes_
        )
        logger.info("Model training completed")

        # Save artifacts
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
        joblib.dump(le, LABEL_ENCODER_PATH)
        for col, encoder in encoders.items():
            joblib.dump(encoder, os.path.join(os.path.dirname(MODEL_PATH), f"{col}_encoder.pkl"))

        # Increment version
        new_version = increment_model_version()
        logger.info(f"Incremented model version to {new_version}")

        # Update in-memory artifacts
        app.state.model = best_model
        app.state.scaler = scaler
        app.state.feature_columns = feature_columns
        app.state.le = le
        app.state.encoders = encoders

        return {"message": f"Model retrained successfully. New version: {new_version}"}
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")
@app.get("/metrics/")
async def get_metrics():
    """Return the current model's performance metrics and version."""
    try:
        metrics = load_metrics()
        logger.info(f"Loaded metrics: {metrics}")  
        
        # Extract weighted avg metrics from classification_report
        weighted_avg = metrics.get("classification_report", {}).get("weighted avg", {})
        
        return {
            "current_version": get_model_version(),  
            "classification_report": metrics.get("classification_report", {}),
            "accuracy": metrics.get("classification_report", {}).get("accuracy", 0.0),
            "f1_score": weighted_avg.get("f1-score", 0.0),
            "precision": weighted_avg.get("precision", 0.0),
            "recall": weighted_avg.get("recall", 0.0),
            "confusion_matrix": metrics.get("confusion_matrix", []),
            "training_samples": metrics.get("training_samples", 0),
            "test_samples": metrics.get("test_samples", 0),
            "timestamp": metrics.get("timestamp", "")
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)