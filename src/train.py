# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def train_and_evaluate(X, y, feature_columns, le, data_dir="data/", model_dir="models/", simplify_training=False, le_classes=None):
    try:
        logger.info("Starting train_and_evaluate")

        # Convert y to a pandas Series if it's a NumPy array
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y, name='usage')
        else:
            y_series = y

        # Log class distribution (decode if le_classes is provided)
        logger.info("Class distribution in 'usage' before filtering:")
        if le_classes is not None:
            # Decode the encoded labels using le_classes
            decoded_series = y_series.map(lambda x: le_classes[int(x)])
            logger.info(decoded_series.value_counts().to_string())
        else:
            logger.info(y_series.value_counts().to_string())

        # Filter data if needed (add your filtering logic here)
        # For now, assuming no filtering
        logger.info("Class distribution in 'usage' after filtering:")
        if le_classes is not None:
            logger.info(decoded_series.value_counts().to_string())
        else:
            logger.info(y_series.value_counts().to_string())

        # Define the model
        model = RandomForestClassifier(random_state=42)

        # Simplify training for Render's free tier
        if simplify_training:
            logger.info("Using simplified training parameters")
            model = RandomForestClassifier(
                n_estimators=50,  # Reduced number of trees
                max_depth=5,      # Reduced depth
                n_jobs=1,         # Single thread to reduce memory usage
                random_state=42
            )
            model.fit(X, y)
        else:
            logger.info("Using GridSearchCV for hyperparameter tuning")
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5]
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")

        # Save the model
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        return model
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}", exc_info=True)
        raise