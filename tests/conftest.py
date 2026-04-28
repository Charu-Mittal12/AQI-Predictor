"""
Shared pytest fixtures for AQI Predictor test suite.
All external services (MLflow, OpenAQ, SQLite) are mocked so tests run offline.
"""
import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))



# ── Minimal env so config.py doesn't raise on import ─────────────────────────
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/mlruns_test")
os.environ.setdefault("MODEL_NAME", "AQIModel")
os.environ.setdefault("MODEL_STAGE", "Production")


@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    """In-memory SQLite DB path shared across a test session."""
    db_path = tmp_path_factory.mktemp("db") / "test_aqi.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cityreadings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            stationid TEXT,
            state TEXT,
            stationname TEXT,
            latitude REAL,
            longitude REAL,
            pm25 REAL, pm10 REAL, no REAL, no2 REAL, nox REAL,
            nh3 REAL, co REAL, so2 REAL, o3 REAL,
            benzene REAL, toluene REAL, xylene REAL,
            aqi REAL,
            temperature2m REAL, relativehumidity2m REAL,
            windspeed10m REAL, winddirection10m REAL,
            precipitation REAL, pressuremsl REAL, cloudcover REAL,
            UNIQUE(city, timestamp)
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            computedat TEXT NOT NULL,
            currentaqi INTEGER,
            forecastjson TEXT,
            pollutantsjson TEXT,
            weeklytrendjson TEXT,
            primarypollutant TEXT,
            UNIQUE(city, computedat)
        );
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_predict_response():
    return {
        "city": "Chennai",
        "station_id": "openaq_123",
        "station_name": "Velachery",
        "location_id": 123,
        "current_aqi": 87,
        "primary_pollutant": "PM2.5",
        "last_updated": "2026-04-26T10:00:00+00:00",
        "forecast_24h": list(range(85, 109)),
        "pollutants": {
            "PM25": 45.2, "PM10": 62.1,
            "NO2": 18.3, "SO2": 5.1,
            "CO": 0.9,  "O3": 31.0,
        },
        "weekly_trend": [90, 85, 88, 92, 87, 83, 87],
        "advisory": "Air quality is satisfactory. Sensitive people should stay alert.",
    }


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([[85.0] * 24])
    model.feature_names_in_ = None
    return model