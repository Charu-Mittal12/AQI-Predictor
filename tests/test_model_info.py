"""
Tests for /model-info endpoint.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient


SUPPORTED = ["Chennai", "Delhi", "Mumbai", "Kolkata", "Ahmedabad"]
FAKE_METADATA = {
    "model_name": "AQIModel",
    "model_family": "LightGBM",
    "model_stage": "Production",
    "git_tag": "v1.0.0",
    "git_commit": "abc123",
    "mlflow_run_id": "run123",
    "primary_metric": "rmse",
    "primary_metric_value": 12.5,
    "forecast_horizon": 24,
}


def _make_app():
    from fastapi import FastAPI
    from app.routers.model_info import router
    app = FastAPI()
    app.include_router(router)
    return app


class TestModelInfoEndpoint:
    """TC-MI-01 to TC-MI-05: /model-info endpoint contract."""

    def test_model_info_returns_200(self):
        """TC-MI-01: /model-info returns HTTP 200."""
        with patch("app.routers.model_info.get_supported_cities", return_value=SUPPORTED),              patch("app.routers.model_info.get_model_metadata", return_value=FAKE_METADATA):
            client = TestClient(_make_app())
            assert client.get("/model-info").status_code == 200

    def test_model_info_contains_model_name(self):
        """TC-MI-02: Response has model_name field."""
        with patch("app.routers.model_info.get_supported_cities", return_value=SUPPORTED),              patch("app.routers.model_info.get_model_metadata", return_value=FAKE_METADATA):
            client = TestClient(_make_app())
            data = client.get("/model-info").json()
            assert "model_name" in data

    def test_model_info_supported_cities_count(self):
        """TC-MI-03: supported_cities_count matches fixture length."""
        with patch("app.routers.model_info.get_supported_cities", return_value=SUPPORTED),              patch("app.routers.model_info.get_model_metadata", return_value=FAKE_METADATA):
            client = TestClient(_make_app())
            data = client.get("/model-info").json()
            assert data["supported_cities_count"] == len(SUPPORTED)

    def test_model_info_has_forecast_horizon(self):
        """TC-MI-04: forecast_horizon_hours is a positive integer."""
        with patch("app.routers.model_info.get_supported_cities", return_value=SUPPORTED),              patch("app.routers.model_info.get_model_metadata", return_value=FAKE_METADATA):
            client = TestClient(_make_app())
            data = client.get("/model-info").json()
            assert isinstance(data.get("forecast_horizon_hours"), int)
            assert data["forecast_horizon_hours"] > 0

    def test_model_info_supported_cities_is_list(self):
        """TC-MI-05: supported_cities is a list."""
        with patch("app.routers.model_info.get_supported_cities", return_value=SUPPORTED),              patch("app.routers.model_info.get_model_metadata", return_value=FAKE_METADATA):
            client = TestClient(_make_app())
            data = client.get("/model-info").json()
            assert isinstance(data["supported_cities"], list)