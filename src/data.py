# src/data.py
import os
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "Fashion-Styles_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Fashion_data")

def get_mongo_collection():
    """Connect to MongoDB and return the collection."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logger.info("Connected to MongoDB successfully.")
        return collection
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def upload_csv_to_mongo(file_path: str):
    """Upload a CSV file to MongoDB."""
    try:
        logger.info(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"CSV read successfully. Shape: {df.shape}")
        return upload_df_to_mongo(df)
    except Exception as e:
        logger.error(f"Error uploading CSV to MongoDB: {str(e)}")
        raise

def upload_df_to_mongo(df: pd.DataFrame):
    """Upload a DataFrame to MongoDB."""
    try:
        logger.info(f"Uploading DataFrame with shape: {df.shape}")
        collection = get_mongo_collection()
        logger.info("Clearing existing data in MongoDB collection...")
        collection.delete_many({})  # Clear existing data
        logger.info("Converting DataFrame to dict for MongoDB insertion...")
        records = df.to_dict('records')
        logger.info(f"Inserting {len(records)} records into MongoDB...")
        collection.insert_many(records)
        logger.info(f"Uploaded {len(df)} records to MongoDB")
    except Exception as e:
        logger.error(f"Error uploading DataFrame to MongoDB: {str(e)}")
        raise

def load_data_from_mongo():
    """Load data from MongoDB into a DataFrame."""
    try:
        collection = get_mongo_collection()
        data = list(collection.find())
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        logger.info(f"Loaded {len(df)} records from MongoDB.")
        return df
    except Exception as e:
        logger.error(f"Error loading from MongoDB: {str(e)}")
        raise

# Aliases to match pipeline.py imports
upload_to_mongo = upload_df_to_mongo  
load_from_mongo = load_data_from_mongo