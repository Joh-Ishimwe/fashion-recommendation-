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
except FileNotFoundError as e:
    logger.error(f"Model artifacts not found: {str(e)}. Please ensure the model is trained.")
    raise HTTPException(status_code=500, detail="Model artifacts not found. Please train the model first.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion Recommendation API"}

@app.get("/fashion_items/")
async def get_fashion_items(gender: str = None, limit: int = 10):
    try:
        collection = get_mongo_collection()
        query = {"gender": gender} if gender else {}
        items = list(collection.find(query).limit(limit))
        for item in items:
            item["_id"] = str(item["_id"])
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/predict/")
async def predict_usage(item: FashionItem):
    """Predict the usage category for a fashion item."""
    try:
        logger.info(f"Received prediction request: {item.model_dump()}")
        raw_columns = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year"]
        input_data = pd.DataFrame([item.model_dump()], columns=raw_columns)
        X_new_scaled = preprocess_new_data(input_data, app.state.feature_columns, app.state.scaler)
        y_pred = app.state.model.predict(X_new_scaled)
        predicted_usage = app.state.le.inverse_transform(y_pred)[0]
        return {"predicted_usage": predicted_usage}
    except Exception as e:
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
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload error: {str(e)}")
    finally:
        await file.close()  # Ensure the file is closed

@app.post("/retrain/")
async def retrain_model():
    """Retrain the model using data from MongoDB."""
    try:
        # Load data from MongoDB
        df = load_from_mongo()
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found in MongoDB for retraining")

        # Preprocess the data
        X, y, feature_columns, scaler, le = preprocess_data(df)

        # Train the model
        best_model = train_and_evaluate(X, y, feature_columns, le, data_dir="data/", model_dir="models/")

        # Evaluate on test set to get metrics
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv")["usage"]
        y_test_encoded = le.transform(y_test)
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)

        # Save metrics
        metrics = {
            "version": get_model_version(),
            "classification_report": report
        }
        save_metrics(metrics)

        # Save new artifacts
        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
        joblib.dump(le, LABEL_ENCODER_PATH)

        # Update the in-memory model and artifacts
        app.state.model = best_model
        app.state.scaler = scaler
        app.state.feature_columns = feature_columns
        app.state.le = le

        # Increment model version
        new_version = increment_model_version()

        return {"message": f"Model retrained successfully. New version: {new_version}"}
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/metrics/")
async def get_metrics():
    """Return the current model's performance metrics and version."""
    try:
        metrics = load_metrics()
        metrics["current_version"] = get_model_version()
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # For production, bind to 0.0.0.0 and use the PORT environment variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, port=port)