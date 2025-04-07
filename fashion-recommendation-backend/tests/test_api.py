# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Fashion Recommendation API"}

def test_predict():
    payload = {
        "gender": "women",
        "masterCategory": "apparel",
        "subCategory": "topwear",
        "articleType": "tshirts",
        "baseColour": "red",
        "season": "summer",
        "year": 2023
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert "predicted_usage" in response.json()

def test_upload_data():
    with open("data/styles.csv", "rb") as f:
        response = client.post("/upload_data/", files={"file": ("styles.csv", f, "text/csv")})
    assert response.status_code == 200
    assert "message" in response.json()

def test_retrain():
    response = client.post("/retrain/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_metrics():
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "classification_report" in response.json()