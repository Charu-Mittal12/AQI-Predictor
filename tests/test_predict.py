"""
Tests for /predict endpoint — all external calls are mocked.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient


SUPPORTED_CITIES = ["Chennai", "Delhi", "Mumbai"]

FAKE_PAYLOAD = {
    "X": np.zeros((1, 10), dtype=np.float32),
    "raw_data": {
        "city": "Chennai",
        "station_id": "openaq_123",
        "station_name": "Velachery",
        "location_id": 123,
        "aqi": 87.0,
        "pm25": 45.0, "pm10": 60.0,
        "no2": 18.0, "so2": 5.0,
        "co": 0.9, "o3": 30.0,
        "timestamp": "2026-04-26T10:00:00+00:00",
        "trend_7d": [90, 85, 88, 92, 87, 83, 87],
        "datasource": "live_ingestion_db",
    },
}

FAKE_RESULT = {
    "city": "Chennai",
    "station_id": "openaq_123",
    "station_name": "Velachery",
    "location_id": 123,
    "current_aqi": 87,
    "primary_pollutant": "PM2.5",
    "last_updated": "2026-04-26T10:00:00+00:00",
    "forecast_24h": [85] * 24,
    "pollutants": {"PM2_5": 45.0, "PM10": 60.0, "NO2": 18.0, "SO2": 5.0, "CO": 0.9, "O3": 30.0},
    "weekly_trend": [90, 85, 88, 92, 87, 83, 87],
    "advisory": "Air quality is satisfactory.",
}


def _make_app():
    from fastapi import FastAPI
    from app.routers.predict import router
    app = FastAPI()
    app.include_router(router)
    return app


COMMON_MOCKS = {
    "app.routers.predict.get_supported_cities": SUPPORTED_CITIES,
    "app.routers.predict.get_location_latest_timestamp": None,
    "app.routers.predict.get_latest_reading": {"timestamp": "2026-04-26T09:00:00+00:00"},
    "app.routers.predict.get_recent_readings": [{}] * 200,
    "app.routers.predict.build_live_payload": FAKE_PAYLOAD,
    "app.routers.predict.predict_next_24h": FAKE_RESULT,
    "app.routers.predict.store_prediction": None,
    "app.routers.predict.make_advisory": "Air quality is satisfactory.",
    "app.routers.predict.PREDICT_REQUESTS": MagicMock(),
    "app.routers.predict.PREDICT_LATENCY": MagicMock(),
    "app.routers.predict.LIVE_FETCH_REQUESTS": MagicMock(),
    "app.routers.predict.LIVE_FETCH_ROWS": MagicMock(),
    "app.routers.predict.LATEST_DB_AGE_HOURS": MagicMock(),
}


class TestPredictEndpointSuccess:
    """TC-P-01 to TC-P-05: Happy path prediction tests."""

    def _patched_client(self):
        patches = {k: patch(k, return_value=v) for k, v in COMMON_MOCKS.items()}
        for p in patches.values():
            p.start()
        client = TestClient(_make_app())
        return client, patches

    def test_predict_returns_200(self):
        """TC-P-01: Valid city and location_id returns HTTP 200."""
        client, patches = self._patched_client()
        try:
            r = client.get("/predict", params={"city": "Chennai", "location_id": 123})
            assert r.status_code == 200
        finally:
            for p in patches.values():
                p.stop()

    def test_predict_response_has_forecast(self):
        """TC-P-02: Response contains forecast_24h with 24 values."""
        client, patches = self._patched_client()
        try:
            r = client.get("/predict", params={"city": "Chennai", "location_id": 123})
            data = r.json()
            assert "forecast_24h" in data
            assert len(data["forecast_24h"]) == 24
        finally:
            for p in patches.values():
                p.stop()

    def test_predict_response_has_advisory(self):
        """TC-P-03: Response contains a non-empty advisory string."""
        client, patches = self._patched_client()
        try:
            r = client.get("/predict", params={"city": "Chennai", "location_id": 123})
            data = r.json()
            assert "advisory" in data
            assert isinstance(data["advisory"], str)
            assert len(data["advisory"]) > 0
        finally:
            for p in patches.values():
                p.stop()

    def test_predict_forecast_values_non_negative(self):
        """TC-P-04: All forecast AQI values must be >= 0."""
        client, patches = self._patched_client()
        try:
            r = client.get("/predict", params={"city": "Chennai", "location_id": 123})
            for val in r.json()["forecast_24h"]:
                assert val >= 0
        finally:
            for p in patches.values():
                p.stop()

    def test_predict_current_aqi_is_integer(self):
        """TC-P-05: current_aqi is a non-negative integer."""
        client, patches = self._patched_client()
        try:
            r = client.get("/predict", params={"city": "Chennai", "location_id": 123})
            aqi = r.json()["current_aqi"]
            assert isinstance(aqi, int)
            assert aqi >= 0
        finally:
            for p in patches.values():
                p.stop()


class TestPredictEndpointErrors:
    """TC-P-06 to TC-P-09: Error handling tests."""

    def test_unsupported_city_returns_400(self):
        """TC-P-06: Unsupported city name returns HTTP 400."""
        with patch("app.routers.predict.get_supported_cities", return_value=SUPPORTED_CITIES),              patch("app.routers.predict.PREDICT_REQUESTS", MagicMock()),              patch("app.routers.predict.PREDICT_LATENCY", MagicMock()),              patch("app.routers.predict.LATEST_DB_AGE_HOURS", MagicMock()),              patch("app.routers.predict.get_latest_reading", return_value=None),              patch("app.routers.predict.get_recent_readings", return_value=[]):
            client = TestClient(_make_app())
            r = client.get("/predict", params={"city": "Atlantis", "location_id": 999})
            assert r.status_code == 400

    def test_missing_location_id_returns_422(self):
        """TC-P-07: Missing required location_id query param returns HTTP 422."""
        client = TestClient(_make_app())
        r = client.get("/predict", params={"city": "Chennai"})
        assert r.status_code == 422

    def test_missing_city_returns_422(self):
        """TC-P-08: Missing city param returns HTTP 422."""
        client = TestClient(_make_app())
        r = client.get("/predict", params={"location_id": 123})
        assert r.status_code == 422

    def test_error_detail_is_string(self):
        """TC-P-09: Error response detail field is a human-readable string."""
        with patch("app.routers.predict.get_supported_cities", return_value=SUPPORTED_CITIES),              patch("app.routers.predict.PREDICT_REQUESTS", MagicMock()),              patch("app.routers.predict.PREDICT_LATENCY", MagicMock()),              patch("app.routers.predict.LATEST_DB_AGE_HOURS", MagicMock()),              patch("app.routers.predict.get_latest_reading", return_value=None),              patch("app.routers.predict.get_recent_readings", return_value=[]):
            client = TestClient(_make_app())
            r = client.get("/predict", params={"city": "Atlantis", "location_id": 999})
            assert isinstance(r.json()["detail"], str)