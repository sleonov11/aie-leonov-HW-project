import os
import pytest
from fastapi.testclient import TestClient
from src.service.main import app

MODEL_PATH = os.getenv('MODEL_PATH', 'artifacts/model_pipeline.pkl')

@pytest.fixture(scope="module")
def client():
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True

def test_predict_single(client):
    test_data = {
        "data": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        ]
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert "probabilities" in result
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in [0, 1]

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "avg_response_time_sec" in data