# src/utils.py
import json
import os

def get_model_version():
    """Get the current model version."""
    version_file = "models/model_version.txt"
    if not os.path.exists(version_file):
        with open(version_file, "w") as f:
            f.write("1")
        return 1
    with open(version_file, "r") as f:
        return int(f.read().strip())

def increment_model_version():
    """Increment the model version."""
    current_version = get_model_version()
    new_version = current_version + 1
    with open("models/model_version.txt", "w") as f:
        f.write(str(new_version))
    return new_version

def save_metrics(metrics):
    """Save metrics to a file."""
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def load_metrics():
    """Load metrics from a file."""
    metrics_file = "models/metrics.json"
    if not os.path.exists(metrics_file):
        return {"message": "No metrics available"}
    with open(metrics_file, "r") as f:
        return json.load(f)