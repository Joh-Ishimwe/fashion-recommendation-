# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split  # Added import
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

        if le_classes is None:
            if hasattr(le, 'classes_'):
                le_classes = le.classes_
                logger.info("Using le.classes_ as le_classes was not provided")
            else:
                raise ValueError("le_classes is None and le has no classes_ attribute")

        logger.info(f"le_classes: {le_classes}")

        if isinstance(y, np.ndarray):
            y_series = pd.Series(y, name='usage')
        else:
            y_series = y

        logger.info("Class distribution in 'usage' before filtering:")
        decoded_series = y_series.map(lambda x: le_classes[int(x)])
        logger.info(decoded_series.value_counts().to_string())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        logger.info("Class distribution in 'usage' after filtering:")
        decoded_train_series = pd.Series(y_train).map(lambda x: le_classes[int(x)])
        logger.info(decoded_train_series.value_counts().to_string())

        model = RandomForestClassifier(random_state=42)

        if simplify_training:
            logger.info("Using simplified training parameters")
            model = RandomForestClassifier(
                n_estimators=50,  
                max_depth=5,      
                n_jobs=1,         
                random_state=42
            )
            model.fit(X_train, y_train)
        else:
            logger.info("Using GridSearchCV for hyperparameter tuning")
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5]
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")

        pd.DataFrame(X_test, columns=feature_columns).to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
        pd.DataFrame({'usage': y_test}).to_csv(os.path.join(data_dir, "y_test.csv"), index=False)

        X_test_path = os.path.join(data_dir, "X_test.csv")
        y_test_path = os.path.join(data_dir, "y_test.csv")
        if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
            logger.error("Test data not found")
            raise FileNotFoundError("Test data not found")

        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)["usage"]
        logger.info(f"Loaded X_test with shape: {X_test.shape}")
        logger.info(f"Loaded y_test with shape: {y_test.shape}")
        logger.info(f"y_test sample: {y_test.head().to_list()}")

        y_test_encoded = y_test.astype(int)
        logger.info(f"y_test_encoded sample: {y_test_encoded.head().to_list()}")

        if not all(0 <= val < len(le_classes) for val in y_test_encoded):
            logger.error(f"y_test_encoded contains values outside the range of le_classes: {set(y_test_encoded)}")
            raise ValueError("y_test_encoded contains invalid label indices")

        y_pred = model.predict(X_test)
        logger.info(f"y_pred sample: {y_pred[:5]}")

        if not all(0 <= val < len(le_classes) for val in y_pred):
            logger.error(f"y_pred contains values outside the range of le_classes: {set(y_pred)}")
            raise ValueError("y_pred contains invalid label indices")

        logger.info("Computing evaluation metrics")
        labels = list(range(len(le_classes)))
        logger.info(f"Labels for metrics: {labels}")

        class_report = classification_report(
            y_test_encoded, y_pred, labels=labels, target_names=le_classes, output_dict=True, zero_division=0
        )
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', labels=labels, zero_division=0)
        precision = precision_score(y_test_encoded, y_pred, average='weighted', labels=labels, zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='weighted', labels=labels, zero_division=0)
        conf_matrix = confusion_matrix(y_test_encoded, y_pred, labels=labels).tolist()

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"Precision (weighted): {precision:.4f}")
        logger.info(f"Recall (weighted): {recall:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        from utils import save_metrics, get_model_version
        metrics = {
            "version": get_model_version(),
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

        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        return model
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    pass