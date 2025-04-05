# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df, encoder_dir="models/"):
    try:
        logger.info("Starting preprocess_data")
        logger.info(f"Initial DataFrame shape: {df.shape}")

        # Drop rows with missing values
        logger.info("Dropping rows with missing values")
        df = df.dropna()
        logger.info(f"Shape after dropping NA: {df.shape}")

        # Reset the DataFrame index to ensure consistent indexing
        logger.info("Resetting DataFrame index")
        df = df.reset_index(drop=True)

        # Log class distribution before filtering
        logger.info("Class distribution in 'usage' before filtering:")
        logger.info(df['usage'].value_counts().to_string())

        # Filter out rows with invalid values
        logger.info("Filtering rows with invalid values")
        mask = (df['year'] >= 2000) & (df['year'] <= 2025)  
        logger.info(f"Number of rows passing filter: {mask.sum()}")
        df = df[mask] 
        logger.info(f"Shape after filtering: {df.shape}")

        # Log class distribution after filtering
        logger.info("Class distribution in 'usage' after filtering:")
        logger.info(df['usage'].value_counts().to_string())

        # Define categorical and numerical columns
        categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
        numerical_columns = ['year']

        # Encode categorical variables
        logger.info("Encoding categorical variables")
        encoders = {}
        for col in categorical_columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            # Save the encoder
            os.makedirs(encoder_dir, exist_ok=True)
            joblib.dump(encoder, os.path.join(encoder_dir, f"{col}_encoder.pkl"))
            logger.info(f"Saved encoder for {col}")

        # Feature columns
        feature_columns = categorical_columns + numerical_columns
        X = df[feature_columns]

        # Encode the target variable
        logger.info("Encoding target variable 'usage'")
        le = LabelEncoder()
        y = le.fit_transform(df['usage'])
        logger.info(f"Encoded classes: {list(le.classes_)}")

        # Scale numerical features
        logger.info("Scaling numerical features")
        scaler = StandardScaler()
        X = X.copy()  # Avoid SettingWithCopyWarning
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

        logger.info("Preprocessing completed successfully")
        return X, y, feature_columns, scaler, le, encoders
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}", exc_info=True)
        raise

def preprocess_new_data(input_data, feature_columns, scaler, encoder_dir="models/"):
    try:
        logger.info("Starting preprocess_new_data")
        logger.info(f"Input data shape: {input_data.shape}")

        # Define categorical and numerical columns
        categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
        numerical_columns = ['year']

        # Load and apply the encoders for categorical variables
        logger.info("Encoding categorical variables")
        encoders = {}
        for col in categorical_columns:
            encoder_path = os.path.join(encoder_dir, f"{col}_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder for {col} not found at {encoder_path}")
            encoder = joblib.load(encoder_path)
            # Handle unseen labels by mapping to a default value (e.g., the most frequent category)
            input_data[col] = input_data[col].astype(str).map(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            input_data[col] = encoder.transform(input_data[col])
            encoders[col] = encoder

        # Select feature columns
        X = input_data[feature_columns]

        # Scale numerical features
        logger.info("Scaling numerical features")
        X = X.copy()  # Avoid SettingWithCopyWarning
        X[numerical_columns] = scaler.transform(X[numerical_columns])

        logger.info("Preprocessing new data completed successfully")
        return X
    except Exception as e:
        logger.error(f"Error in preprocess_new_data: {str(e)}", exc_info=True)
        raise