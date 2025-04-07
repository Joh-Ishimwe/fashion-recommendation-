# src/utils.py
import json
import os
import logging

logger = logging.getLogger(__name__)

# File to store metrics
METRICS_PATH = "models/metrics.json"
VERSION_PATH = "models/version.txt"

def get_model_version():
    """Get the current model version."""
    try:
        if os.path.exists(VERSION_PATH):
            with open(VERSION_PATH, "r") as f:
                return int(f.read().strip())
        return 0
    except Exception as e:
        logger.error(f"Error reading model version: {str(e)}")
        return 0

def increment_model_version():
    """Increment the model version and return the new version."""
    try:
        current_version = get_model_version()
        new_version = current_version + 1
        os.makedirs(os.path.dirname(VERSION_PATH), exist_ok=True)
        with open(VERSION_PATH, "w") as f:
            f.write(str(new_version))
        return new_version
    except Exception as e:
        logger.error(f"Error incrementing model version: {str(e)}")
        raise

def save_metrics(metrics):
    """Save metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {METRICS_PATH}")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
        raise

def load_metrics():
    """Load metrics from a JSON file."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        raise