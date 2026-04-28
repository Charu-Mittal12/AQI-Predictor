"""
model_service.py
-----------------
Uses MLflow model serving endpoint (port 5001) for inference.
MLflow tracking server (port 5000) is used only for metadata/registry UI.

Architecture:
  MLflow tracking server  :5000  → registry, experiment tracking, UI
  MLflow model server     :5001  → inference (/invocations, /ping)
  FastAPI backend         :8000  → calls :5001 for predictions

Start model server before running backend:
  mlflow models serve -m "models:/AQI_Model/Production" -p 5001 --no-conda
"""

import json
import os
from math import isnan
from numbers import Real

import numpy as np
import pandas as pd
import requests

from app.utils.config import MODEL_METADATA_PATH
from app.utils.logger import get_logger

log = get_logger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME       = os.getenv("MODEL_NAME", "AQI_Model")
MODEL_STAGE      = os.getenv("MODEL_STAGE", "Production")
MLFLOW_SERVE_URI = os.getenv("MLFLOW_SERVE_URI", "http://localhost:5001")

INVOCATIONS_URL  = f"{MLFLOW_SERVE_URI}/invocations"
PING_URL         = f"{MLFLOW_SERVE_URI}/ping"

_metadata = None


# ── Model readiness ────────────────────────────────────────────────────────────
def get_model():
    """
    No longer loads a model object into memory.
    Checks that MLflow model server is reachable and returns invocations URL.
    Kept for backward compatibility — main.py calls get_model() on startup.
    """
    if not is_model_ready():
        raise RuntimeError(
            f"MLflow model server not reachable at {MLFLOW_SERVE_URI}\n"
            f"Start it with:\n"
            f"  mlflow models serve -m models:/{MODEL_NAME}/{MODEL_STAGE} -p 5001 --no-conda"
        )
    log.info("MLflow model server confirmed healthy at %s", MLFLOW_SERVE_URI)
    return INVOCATIONS_URL


def is_model_ready() -> bool:
    """
    Pings MLflow model server /ping endpoint.
    Returns True if server is up and responds 200.
    """
    try:
        resp = requests.get(PING_URL, timeout=5)
        if resp.status_code == 200:
            log.info("MLflow model server healthy at %s", MLFLOW_SERVE_URI)
            return True
        log.warning("MLflow model server ping returned %d", resp.status_code)
        return False
    except requests.exceptions.ConnectionError:
        log.error("MLflow model server not reachable at %s", MLFLOW_SERVE_URI)
        return False
    except Exception as exc:
        log.error("Model readiness check failed: %s", exc)
        return False


# ── Metadata ───────────────────────────────────────────────────────────────────
def get_model_metadata() -> dict:
    global _metadata
    if _metadata is None:
        try:
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        except Exception:
            _metadata = {}
    registry_meta = {
        "model_name":  MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "serve_uri":   MLFLOW_SERVE_URI,
    }
    registry_meta.update(_metadata)
    return registry_meta


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_number(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Real):
        if isnan(float(value)):
            return default
        return float(value)
    return default


def _safe_timestamp(raw_data: dict) -> str:
    value = raw_data.get("timestamp")
    return "" if value is None else str(value)


# ── Core inference ─────────────────────────────────────────────────────────────
def _call_mlflow_serve(X: np.ndarray) -> list:
    """
    Sends feature array to MLflow model server /invocations.
    Returns flat list of predictions (24 AQI values).

    MLflow scoring server payload format:
      {"inputs": [[feature1, feature2, ...]]}

    Response format for multi-output model:
      {"predictions": [[v1, v2, ..., v24]]}
    """
    payload = {"inputs": X.tolist()}

    try:
        resp = requests.post(
            INVOCATIONS_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"MLflow model server not reachable at {INVOCATIONS_URL}. "
            "Make sure 'mlflow models serve' is running on port 5001."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"MLflow model server error {resp.status_code}: {resp.text}"
        ) from exc

    result      = resp.json()
    predictions = result.get("predictions", result)

    # Multi-output model: [[v1, v2, ..., v24]]
    if isinstance(predictions, list) and len(predictions) > 0:
        first = predictions[0]
        if isinstance(first, list):
            return first       # [[24 values]] → [24 values]
        return predictions     # [v1, v2, ...] single output

    raise ValueError(f"Unexpected prediction format from MLflow server: {result}")


def predict_next_24h(city: str, X: np.ndarray, raw_data: dict) -> dict:
    """
    Calls MLflow model server for 24h AQI forecast.
    Same signature as before — predict router and all callers unchanged.
    """
    raw_preds = _call_mlflow_serve(X)
    forecast  = [max(0, int(round(float(v)))) for v in raw_preds]

    # Ensure exactly 24 hours
    if len(forecast) < 24:
        forecast = forecast + [forecast[-1]] * (24 - len(forecast))
    forecast = forecast[:24]

    pollutants = {
        "PM2_5": _safe_number(raw_data.get("pm25")),
        "PM10":  _safe_number(raw_data.get("pm10")),
        "NO2":   _safe_number(raw_data.get("no2")),
        "SO2":   _safe_number(raw_data.get("so2")),
        "CO":    _safe_number(raw_data.get("co")),
        "O3":    _safe_number(raw_data.get("o3")),
    }

    primary_lookup = {
        "PM2_5": "PM2.5",
        "PM10":  "PM10",
        "NO2":   "NO2",
        "SO2":   "SO2",
        "CO":    "CO",
        "O3":    "O3",
    }
    primary = primary_lookup[max(pollutants, key=pollutants.get)] if pollutants else "PM2.5"

    current_aqi  = _safe_number(raw_data.get("aqi"), default=float(forecast[0]))
    weekly_trend = raw_data.get("trend_7d") or forecast[:7]

    return {
        "city":              city,
        "station_id":        raw_data.get("station_id"),
        "station_name":      raw_data.get("station_name"),
        "location_id":       raw_data.get("location_id"),
        "current_aqi":       int(round(current_aqi)),
        "primary_pollutant": primary,
        "last_updated":      _safe_timestamp(raw_data),
        "forecast_24h":      forecast,
        "pollutants":        pollutants,
        "weekly_trend":      [int(round(float(v))) for v in weekly_trend],
    }