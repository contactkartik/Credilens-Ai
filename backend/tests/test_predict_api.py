"""
API tests for POST /api/predict (Stage 6).

Loads the real trained RoBERTa model - predictions are produced by the
model, never hardcoded.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Test client that triggers the app lifespan (model warm-up)."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_returns_model_output(client):
    """A valid statement returns prediction, category and confidence."""
    statement = (
        "Says the Annies List political group supports "
        "third-trimester abortions on demand."
    )
    response = client.post("/api/predict", json={"text": statement})

    assert response.status_code == 200
    body = response.json()

    # Prediction comes from the trained model's label set
    assert body["prediction"] in ("true_side", "false_side")

    # Category must be consistent with the model prediction
    expected_category = "Credible" if body["prediction"] == "true_side" else "Not Credible"
    assert body["credibility_category"] == expected_category

    # Confidence and probabilities are well-formed
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == {"false_side", "true_side"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert body["confidence"] == pytest.approx(
        max(body["probabilities"].values()), abs=1e-6
    )
    assert body["text"] == statement
    assert "roberta" in body["model"].lower()


def test_predict_is_deterministic(client):
    """The same input yields the same model output across calls."""
    statement = "The economy grew by 4 percent last quarter."
    first = client.post("/api/predict", json={"text": statement})
    second = client.post("/api/predict", json={"text": statement})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["prediction"] == second.json()["prediction"]
    assert first.json()["confidence"] == pytest.approx(
        second.json()["confidence"], abs=1e-6
    )


def test_predict_rejects_empty_text(client):
    response = client.post("/api/predict", json={"text": ""})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]


def test_predict_rejects_whitespace_only_text(client):
    response = client.post("/api/predict", json={"text": "   \n\t  "})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]


def test_predict_rejects_missing_text_field(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_non_string_text(client):
    response = client.post("/api/predict", json={"text": 12345})
    assert response.status_code == 422


def test_predict_rejects_empty_json_body(client):
    response = client.post("/api/predict", json=None)
    assert response.status_code == 422
