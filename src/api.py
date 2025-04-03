# src/api.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import csv
from preprocess import preprocess_data, preprocess_new_data
from train import train_and_evaluate
from data import get_mongo_collection, upload_to_mongo, load_from_mongo
from utils import get_model_version, increment_model_version, save_metrics, load_metrics
from datetime import datetime
from models import FashionItem
from config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, LABEL_ENCODER_PATH
import shutil
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Recommendation API")

# Load the trained model and artifacts (make them global so we can update them)
try:
    app.state.model = joblib.load(MODEL_PATH)
    app.state.scaler = joblib.load(SCALER_PATH)
    app.state.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    app.state.le = joblib.load(LABEL_ENCODER_PATH)
    # Load the encoders
    app.state.encoders = {}
    categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
    for col in categorical_columns:
        encoder_path = os.path.join(os.path.dirname(MODEL_PATH), f"{col}_encoder.pkl")
        if os.path.exists(encoder_path):
            app.state.encoders[col] = joblib.load(encoder_path)
        else:
            logger.warning(f"Encoder for {col} not found at {encoder_path}")
except FileNotFoundError as e:
    logger.error(f"Model artifacts not found: {str(e)}. Please ensure the model is trained.")
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
            content = await file.read()  # Read the file content
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
        await file.close()  # Ensure the file is closed

@app.post("/retrain/")
async def retrain_model():
    """Retrain the model using data from MongoDB."""
    try:
        logger.info("Starting retrain process")

        # Load data from MongoDB
        logger.info("Loading data from MongoDB")
        df = load_from_mongo()
        if df.empty:
            logger.error("No data found in MongoDB for retraining")
            raise HTTPException(status_code=400, detail="No data found in MongoDB for retraining")
        logger.info(f"Loaded {len(df)} records from MongoDB")

        # Sample a subset of the data to reduce memory usage
        sample_size = min(10000, len(df))  # Use at most 10,000 records
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} records from {len(df)} for training")
            df = df.sample(n=sample_size, random_state=42)

        # Preprocess the data
        logger.info("Preprocessing data")
        X, y, feature_columns, scaler, le, encoders = preprocess_data(df, encoder_dir=os.path.dirname(MODEL_PATH))
        logger.info("Data preprocessing completed")

        # Split the data into train and test sets
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info(f"Split data into train ({len(X_train)}) and test ({len(X_test)}) sets")

        # Save the test set for evaluation
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
        pd.DataFrame({"usage": y_test}).to_csv("data/y_test.csv", index=False)
        logger.info("Saved test data to data/X_test.csv and data/y_test.csv")

        # Train the model with simplified parameters
        logger.info("Starting model training")
        best_model = train_and_evaluate(
            X_train, y_train, feature_columns, le,
            data_dir="data/", model_dir="models/",
            simplify_training=True,
            le_classes=le.classes_
        )
        logger.info("Model training completed")

        # Evaluate on test set to get metrics
        logger.info("Evaluating model on test set")
        y_pred = best_model.predict(X_test)
        y_test_encoded = le.transform(y_test) if isinstance(y_test, (pd.Series, list)) else y_test
        report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
        logger.info("Model evaluation completed")

        # Save metrics
        logger.info("Saving metrics")
        metrics = {
            "version": get_model_version(),
            "classification_report": report,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "timestamp": datetime.now().isoformat()
        }
        save_metrics(metrics)
        logger.info("Metrics saved")

        # Save new artifacts
        logger.info("Saving model artifacts")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
        joblib.dump(le, LABEL_ENCODER_PATH)
        logger.info("Model artifacts saved")

        # Update the in-memory model and artifacts
        app.state.model = best_model
        app.state.scaler = scaler
        app.state.feature_columns = feature_columns
        app.state.le = le
        app.state.encoders = encoders
        logger.info("Updated in-memory model and artifacts")

        # Increment model version
        new_version = increment_model_version()
        logger.info(f"Incremented model version to {new_version}")

        return {"message": f"Model retrained successfully. New version: {new_version}"}
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/metrics/")
async def get_metrics():
    """Return the current model's performance metrics and version."""
    try:
        metrics = load_metrics()
        metrics["current_version"] = get_model_version()
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)