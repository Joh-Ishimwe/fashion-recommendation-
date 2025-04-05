# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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

        # Filter data (if any filtering logic exists; otherwise, remove this block)
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
                n_estimators=50,  
                max_depth=5,      
                n_jobs=1,         
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

        # Load test data for evaluation (assumes X_test and y_test were saved during retrain)
        X_test_path = os.path.join(data_dir, "X_test.csv")
        y_test_path = os.path.join(data_dir, "y_test.csv")
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            logger.error("Test data not found. Please ensure retrain saves X_test and y_test.")
            raise FileNotFoundError("Test data not found")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)["usage"]
        logger.info(f"Loaded X_test with shape: {X_test.shape}")
        logger.info(f"Loaded y_test with shape: {y_test.shape}")
        logger.info(f"y_test sample: {y_test.head().to_list()}")

        # Ensure y_test is in the correct format (encoded labels)
        y_test_encoded = y_test.astype(int)  # Ensure integer type
        logger.info(f"y_test_encoded sample: {y_test_encoded.head().to_list()}")
        # Verify that all values in y_test_encoded are within the range of le_classes
        if le_classes is not None:
            if not all(0 <= val < len(le_classes) for val in y_test_encoded):
                logger.error(f"y_test_encoded contains values outside the range of le_classes: {set(y_test_encoded)}")
                logger.error(f"le_classes: {le_classes}")
                raise ValueError("y_test_encoded contains invalid label indices")

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        logger.info(f"y_pred sample: {y_pred[:5]}")
        # Verify that y_pred values are within the range of le_classes
        if le_classes is not None:
            if not all(0 <= val < len(le_classes) for val in y_pred):
                logger.error(f"y_pred contains values outside the range of le_classes: {set(y_pred)}")
                logger.error(f"le_classes: {le_classes}")
                raise ValueError("y_pred contains invalid label indices")

        # Compute metrics
        logger.info("Computing evaluation metrics")
        # Define the labels to match le_classes
        labels = list(range(len(le_classes)))  # [0, 1, 2, ..., len(le_classes)-1]
        logger.info(f"Labels for metrics: {labels}")

        # Classification Report
        class_report = classification_report(
            y_test_encoded,
            y_pred,
            labels=labels,  # Specify labels to include all classes
            target_names=le_classes,
            output_dict=True,
            zero_division=0  # Handle cases where a class has no predictions
        )

        # Accuracy
        accuracy = accuracy_score(y_test_encoded, y_pred)

        # F1 Score (weighted average)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', labels=labels, zero_division=0)

        # Precision (weighted average)
        precision = precision_score(y_test_encoded, y_pred, average='weighted', labels=labels, zero_division=0)

        # Recall (weighted average)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', labels=labels, zero_division=0)

        # Confusion Matrix
        logger.info(f"le_classes: {le_classes}")
        logger.info(f"Number of classes: {len(le_classes)}")
        logger.info(f"Labels for confusion matrix: {labels}")
        conf_matrix = confusion_matrix(y_test_encoded, y_pred, labels=labels).tolist()  # Ensure all classes are included
        logger.info(f"Computed confusion matrix: {conf_matrix}")
        # Validate that conf_matrix matches the expected shape
        if len(conf_matrix) != len(le_classes) or any(len(row) != len(le_classes) for row in conf_matrix):
            logger.error(f"Confusion matrix shape {len(conf_matrix)}x{len(conf_matrix[0])} does not match expected {len(le_classes)}x{len(le_classes)}")
            raise ValueError("Confusion matrix shape mismatch")

        # Log the metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"Precision (weighted): {precision:.4f}")
        logger.info(f"Recall (weighted): {recall:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Save the metrics (assumes save_metrics is defined in utils.py)
        from utils import save_metrics, get_model_version  # Import here to avoid circular imports
        metrics = {
            "version": get_model_version(),  # Include the version
            "classification_report": class_report,
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix,
            "training_samples": len(X),
            "test_samples": len(X_test),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        logger.info(f"Saving metrics: {metrics}")
        save_metrics(metrics)
        logger.info("Metrics saved")

        # Save the model
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        return model
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}", exc_info=True)
        raise