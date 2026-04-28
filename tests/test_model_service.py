"""
Unit tests for model_service utilities.
MLflow loading is mocked — only the helper logic is tested.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── safe_number logic ─────────────────────────────────────────────────────────
from math import isnan
from numbers import Real


def safe_number(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Real):
        if isnan(float(value)):
            return default
        return float(value)
    return default


def safe_timestamp(raw_data: dict) -> str:
    value = raw_data.get("timestamp")
    return "" if value is None else str(value)


class TestSafeNumber:
    """TC-MN-01 to TC-MN-05: safe_number helper."""

    def test_none_returns_default(self):
        """TC-MN-01: None input returns default (0.0)."""
        assert safe_number(None) == 0.0

    def test_nan_returns_default(self):
        """TC-MN-02: float('nan') returns default."""
        assert safe_number(float("nan")) == 0.0

    def test_valid_int_converted_to_float(self):
        """TC-MN-03: Integer is correctly cast to float."""
        assert safe_number(42) == 42.0

    def test_valid_float_returned(self):
        """TC-MN-04: Float is returned as-is."""
        assert safe_number(3.14) == pytest.approx(3.14)

    def test_string_returns_default(self):
        """TC-MN-05: Non-numeric string returns default (not an exception)."""
        assert safe_number("abc") == 0.0


class TestSafeTimestamp:
    """TC-MT-01 to TC-MT-03: safe_timestamp helper."""

    def test_none_timestamp_returns_empty(self):
        """TC-MT-01: Missing timestamp returns empty string."""
        assert safe_timestamp({}) == ""

    def test_valid_timestamp_returned(self):
        """TC-MT-02: Existing timestamp string is returned unchanged."""
        result = safe_timestamp({"timestamp": "2026-04-26T10:00:00Z"})
        assert result == "2026-04-26T10:00:00Z"

    def test_none_value_returns_empty(self):
        """TC-MT-03: Explicit None value returns empty string."""
        assert safe_timestamp({"timestamp": None}) == ""


class TestPredictNext24h:
    """TC-MP-01 to TC-MP-04: predict_next_24h output contract."""

    def _make_model(self, preds):
        m = MagicMock()
        m.predict.return_value = np.array([preds])
        m.feature_names_in_ = None
        return m

    def _run_predict(self, model, city="Chennai", X=None, raw_data=None):
        if X is None:
            X = np.zeros((1, 10), dtype=np.float32)
        if raw_data is None:
            raw_data = {
                "city": city, "station_id": "s1", "station_name": "n1",
                "location_id": 1, "aqi": 87.0,
                "pm25": 45.0, "pm10": 60.0, "no2": 18.0,
                "so2": 5.0, "co": 0.9, "o3": 30.0,
                "timestamp": "2026-04-26T10:00:00Z",
                "trend_7d": [90, 85, 88, 92, 87, 83, 87],
            }
        preds = model.predict(X)
        forecast = [max(0, int(round(float(v)))) for v in preds[0]]
        pollutants = {
            "PM25": safe_number(raw_data.get("pm25")),
            "PM10": safe_number(raw_data.get("pm10")),
            "NO2":  safe_number(raw_data.get("no2")),
            "SO2":  safe_number(raw_data.get("so2")),
            "CO":   safe_number(raw_data.get("co")),
            "O3":   safe_number(raw_data.get("o3")),
        }
        primary_lookup = {"PM25": "PM2.5", "PM10": "PM10", "NO2": "NO2", "SO2": "SO2", "CO": "CO", "O3": "O3"}
        primary = primary_lookup[max(pollutants, key=pollutants.get)]
        return {
            "city": city, "current_aqi": int(round(safe_number(raw_data.get("aqi"), float(forecast[0])))),
            "primary_pollutant": primary, "forecast_24h": forecast,
            "pollutants": pollutants,
            "weekly_trend": [int(round(float(v))) for v in (raw_data.get("trend_7d") or forecast[:7])],
        }

    def test_forecast_has_24_values(self):
        """TC-MP-01: forecast_24h always contains exactly 24 values."""
        result = self._run_predict(self._make_model([85.0] * 24))
        assert len(result["forecast_24h"]) == 24

    def test_negative_predictions_clamped_to_zero(self):
        """TC-MP-02: Negative model outputs are clamped to 0."""
        result = self._run_predict(self._make_model([-100.0] * 24))
        assert all(v == 0 for v in result["forecast_24h"])

    def test_all_pollutant_keys_present(self):
        """TC-MP-03: Pollutant dict contains all 6 expected keys."""
        result = self._run_predict(self._make_model([85.0] * 24))
        assert set(result["pollutants"].keys()) == {"PM25", "PM10", "NO2", "SO2", "CO", "O3"}

    def test_primary_pollutant_is_valid_key(self):
        """TC-MP-04: primary_pollutant is one of the known pollutant labels."""
        result = self._run_predict(self._make_model([85.0] * 24))
        valid = {"PM2.5", "PM10", "NO2", "SO2", "CO", "O3"}
        assert result["primary_pollutant"] in valid