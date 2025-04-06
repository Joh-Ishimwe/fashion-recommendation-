# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
import os

try:
    from imblearn.over_sampling import SMOTE # type: ignore
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logging.warning("SMOTE not available; install 'imbalanced-learn' to enable class balancing.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df, encoder_dir="models/", apply_smote=False, min_samples_per_class=10):
    try:
        logger.info("Starting preprocess_data")
        logger.info(f"Initial DataFrame shape: {df.shape}")

        df = df.copy()  # Avoid view/copy issues

        categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
        numerical_columns = ['year']
        df.loc[:, categorical_columns] = df[categorical_columns].fillna('Unknown')
        df.loc[:, numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
        df.loc[:, 'usage'] = df['usage'].fillna('Unknown')
        
        logger.info("Filtering invalid data")
        mask = (df['year'] >= 2000) & (df['year'] <= 2025) & \
               df.apply(lambda row: row['masterCategory'] != 'Footwear' or 
                        row['articleType'] in ['Casual Shoes', 'Sports Shoes'], axis=1)
        df = df[mask]
        logger.info(f"Shape after filtering: {df.shape}")

        logger.info("Class distribution in 'usage':")
        logger.info(df['usage'].value_counts().to_string())

        logger.info("Encoding categorical variables")
        encoders = {}
        for col in categorical_columns:
            encoder = LabelEncoder()
            df.loc[:, col] = encoder.fit_transform(df[col].astype(str))  
            encoders[col] = encoder
            os.makedirs(encoder_dir, exist_ok=True)
            joblib.dump(encoder, os.path.join(encoder_dir, f"{col}_encoder.pkl"))
            logger.info(f"Saved encoder for {col}")

        feature_columns = categorical_columns + numerical_columns
        X = df[feature_columns]

        logger.info("Encoding target variable 'usage'")
        le = LabelEncoder()
        y = le.fit_transform(df['usage'])
        logger.info(f"Encoded classes: {list(le.classes_)}")

        logger.info("Scaling numerical features")
        scaler = StandardScaler()
        X = X.copy()
        X.loc[:, numerical_columns] = scaler.fit_transform(X[numerical_columns])

        if apply_smote and SMOTE_AVAILABLE:
            class_counts = pd.Series(y).value_counts()
            if class_counts.min() >= min_samples_per_class:
                smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min() - 1))
                X, y = smote.fit_resample(X, y)
                logger.info("Applied SMOTE to balance classes")
                logger.info(f"New shape after SMOTE: {X.shape}")
            else:
                logger.warning("Not enough samples per class for SMOTE")
        elif apply_smote and not SMOTE_AVAILABLE:
            logger.warning("SMOTE requested but not available; skipping class balancing.")

        logger.info("Preprocessing completed successfully")
        return X, y, feature_columns, scaler, le, encoders
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}", exc_info=True)
        raise
def preprocess_new_data(input_data, feature_columns, scaler, encoder_dir="models/"):
    """
    Preprocess new data for prediction, maintaining API compatibility.
    """
    try:
        logger.info("Starting preprocess_new_data")
        logger.info(f"Input data shape: {input_data.shape}")

        categorical_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
        numerical_columns = ['year']

        # Handle missing values
        input_data[categorical_columns] = input_data[categorical_columns].fillna('Unknown')
        input_data[numerical_columns] = input_data[numerical_columns].fillna(input_data[numerical_columns].median())

        # Load and apply encoders
        logger.info("Encoding categorical variables")
        for col in categorical_columns:
            encoder_path = os.path.join(encoder_dir, f"{col}_encoder.pkl")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder for {col} not found at {encoder_path}")
            
            encoder = joblib.load(encoder_path)
            
            # Handle unseen labels by mapping them to a known value (e.g., most frequent class)
            unknown_handler = lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            input_data[col] = input_data[col].astype(str).apply(unknown_handler)
            
            input_data[col] = encoder.transform(input_data[col])

        # Select feature columns
        X = input_data[feature_columns]

        # Scale numerical features
        logger.info("Scaling numerical features")
        X = X.copy()
        X[numerical_columns] = scaler.transform(X[numerical_columns])

        logger.info("Preprocessing new data completed successfully")
        return X
    except Exception as e:
        logger.error(f"Error in preprocess_new_data: {str(e)}", exc_info=True)
        raise