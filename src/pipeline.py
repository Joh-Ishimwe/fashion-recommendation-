# src/pipeline.py
import os
import pandas as pd
from data import load_data_from_mongo, upload_csv_to_mongo
from preprocess import preprocess_data, preprocess_new_data
from train import train_and_evaluate
from predict import make_predictions
from utils import get_model_version, increment_model_version, save_metrics
import joblib
from sklearn.metrics import classification_report

def run_pipeline(data_dir='data/', model_dir='models/', upload=False, sample_size=None):
    """Run the full ML pipeline with MongoDB data."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if upload:
        print("Uploading data to MongoDB...")
        upload_csv_to_mongo(file_path=f'{data_dir}styles_cleaned.csv')
    
    print("Loading data from MongoDB...")
    df = load_data_from_mongo()
    
    if df.empty:
        raise ValueError("No data found in MongoDB. Please upload data using the /upload_data/ endpoint or set upload=True.")
    
    if sample_size is not None:
        print(f"Sampling {sample_size} records for testing...")
        df = df.sample(n=sample_size, random_state=42)
    
    print("Preprocessing data...")
    X_scaled, y_encoded, feature_columns, scaler, le, encoders = preprocess_data(df)
    
    print("Training model...")
    best_model = train_and_evaluate(
        X_scaled, y_encoded, feature_columns, le, data_dir, model_dir, le_classes=le.classes_  # Pass le.classes_
    )
    
    # Evaluate on test set to get metrics
    X_test = pd.read_csv(f"{data_dir}X_test.csv")
    y_test = pd.read_csv(f"{data_dir}y_test.csv")["usage"]
    y_test_encoded = le.transform(y_test)
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
    
    metrics = {
        "version": get_model_version(),
        "classification_report": report
    }
    save_metrics(metrics)
    
    joblib.dump(best_model, f'{model_dir}best_model.pkl')
    joblib.dump(scaler, f'{model_dir}scaler.pkl')
    joblib.dump(feature_columns, f'{model_dir}feature_columns.pkl')
    joblib.dump(le, f'{model_dir}label_encoder.pkl')
    for col, encoder in encoders.items():
        joblib.dump(encoder, f'{model_dir}{col}_encoder.pkl')
    print("Artifacts saved to models/ directory")
    
    new_version = increment_model_version()
    print(f"Model version updated to {new_version}")
    
    print("Testing prediction...")
    X_new = pd.DataFrame({
        'gender': ['Women'],
        'masterCategory': ['Apparel'],
        'subCategory': ['Topwear'],
        'articleType': ['Tshirts'],
        'baseColour': ['Blue'],
        'season': ['Summer'],
        'year': [2023]
    })
    X_new_scaled = preprocess_new_data(X_new, feature_columns, scaler)
    predicted_categories = make_predictions(f'{model_dir}best_model.pkl', X_new_scaled, le)
    if predicted_categories is not None:
        print("Predicted Usage:", predicted_categories[0])
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    run_pipeline(upload=True, sample_size=10000)