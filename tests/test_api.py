"""
tests/test_api.py
=============================================================================
Integration Testing Suite for the AI Tech Lead API.
Validates the end-to-end inference flow against the Flask test client.
"""

import os
import pytest
import json

def test_health_endpoint(client):
    """Confirm the service uptime and artifact hydration."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["registry_loaded"] is True

def test_models_metadata(client):
    """Confirm the model metadata reporting."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "churn_classifier" in data["models"]
    assert data["feature_count"]["churn"] > 0

def test_churn_prediction_success(client):
    """Test standard churn inference with a valid payload."""
    payload = {
        "Frequency": 45,
        "MonetaryTotal": 12500.5,
        "TotalQuantity": 150,
        "ReturnRatio": 0.05,
        "UniqueInvoices": 15,
        "AvgProductsPerTransaction": 3.5,
        "Age": 34,
        "SupportTicketsCount": 0
    }
    response = client.post(
        "/api/predict/churn",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["success"] is True

def test_churn_prediction_invalid_payload(client):
    """Verify that Pydantic rejects structurally malformed payloads."""
    payload = {
        "Frequency": -1, # Breaks `ge=0` rule
        "MonetaryTotal": "not_a_number"
    }
    response = client.post(
        "/api/predict/churn",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "pydantic_errors" in data
    assert len(data["pydantic_errors"]) > 0

def test_revenue_forecast_success(client):
    """Test revenue regression inference."""
    payload = {
        "Frequency": 10,
        "MonetaryTotal": 2000.0,
        "TotalQuantity": 50,
        "ReturnRatio": 0.0
    }
    response = client.post(
        "/api/predict/revenue",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_revenue" in data
    assert data["success"] is True

def test_segmentation_success(client):
    """Test customer clustering (KMeans) endpoint."""
    payload = {"Frequency": 5, "MonetaryTotal": 500.0, "TotalQuantity": 10}
    response = client.post(
        "/api/segment",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "segment_id" in data
    assert "segment_label" in data
