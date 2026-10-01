import pytest
from fastapi.testclient import TestClient

from nwpblend.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_forecast_missing_params():
    response = client.get("/forecast")
    # Should fail validation (422 Unprocessable Entity)
    assert response.status_code == 422


def test_drift():
    response = client.get("/drift")
    assert response.status_code == 200
    assert "flags" in response.json()
