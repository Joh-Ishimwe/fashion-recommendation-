# src/predict.py
import joblib

def make_predictions(model_path, X_new_scaled, le):
    """Make predictions on new data."""
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_new_scaled)
        predicted_categories = le.inverse_transform(y_pred)
        return predicted_categories
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

if __name__ == "__main__":
    pass