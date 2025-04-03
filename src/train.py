# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

def train_and_evaluate(X, y, feature_columns, le, data_dir='data/', model_dir='models/'):
    """Train and evaluate the model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test data for later evaluation in API
    X_test_df = pd.DataFrame(X_test, columns=feature_columns)
    y_test_df = pd.DataFrame(le.inverse_transform(y_test), columns=["usage"])
    X_test_df.to_csv(f"{data_dir}X_test.csv", index=False)
    y_test_df.to_csv(f"{data_dir}y_test.csv", index=False)

    # Define model
    model = xgb.XGBClassifier(eval_metric='mlogloss')  # Removed use_label_encoder

    # Define hyperparameter grid (simplified for testing)
    param_grid = {
        'n_estimators': [100],  # Reduced to one value
        'max_depth': [3, 5],   # Reduced options
        'learning_rate': [0.1] # Reduced to one value
    }

    # Perform GridSearchCV with limited parallelization
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='accuracy', n_jobs=2, verbose=1  # Limit to 2 jobs
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    print("Classification Report:\n", classification_report(
        y_test, y_pred, target_names=le.classes_
    ))

    return best_model

if __name__ == "__main__":
    pass