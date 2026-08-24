"""
Integration tests for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_kpis_endpoint():
    response = client.get("/api/v1/kpis")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert data["total_events"] > 0


def test_predict_endpoint():
    sample_payload = {
        "id": 114387,
        "rpm": 1000.0,
        "temperature_c": 24.69,
        "z_rms_velocity_mm_s": 1.517,
        "x_rms_velocity_mm_s": 2.0,
        "z_peak_acceleration_g": 0.484,
        "x_peak_acceleration_g": 0.631,
        "z_rms_acceleration_g": 0.09,
        "x_rms_acceleration_g": 0.114,
        "z_kurtosis": 2.392,
        "x_kurtosis": 2.77,
        "z_crest_factor": 3.747,
        "x_crest_factor": 4.269,
        "z_high_freq_rms_accel_g": 0.129,
        "x_high_freq_rms_accel_g": 0.147,
    }
    response = client.post("/api/v1/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert data["predicted_category"] == "cocked_rotor"


def test_full_prescribe_endpoint():
    sample_payload = {
        "id": 114387,
        "rpm": 1000.0,
        "temperature_c": 24.69,
        "z_rms_velocity_mm_s": 1.517,
        "x_rms_velocity_mm_s": 2.0,
        "z_peak_acceleration_g": 0.484,
        "x_peak_acceleration_g": 0.631,
        "z_rms_acceleration_g": 0.09,
        "x_rms_acceleration_g": 0.114,
        "z_kurtosis": 2.392,
        "x_kurtosis": 2.77,
        "z_crest_factor": 3.747,
        "x_crest_factor": 4.269,
        "z_high_freq_rms_accel_g": 0.129,
        "x_high_freq_rms_accel_g": 0.147,
    }
    response = client.post("/api/v1/prescribe", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["has_document"] is True
    assert len(data["prescription"]) > 50
    assert "similarity_summary" in data
