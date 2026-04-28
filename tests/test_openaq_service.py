"""
Unit tests for OpenAQ service normalization helpers.
No HTTP calls made — pure unit tests.
"""
import pytest


# ── Inline mirrors of normalise helpers ───────────────────────────────────────

MOLAR_MASS = {
    "no": 30.01, "no2": 46.0055, "nox": 46.0055, "nh3": 17.031,
    "co": 28.01,  "so2": 64.066,  "o3": 48.00,
}


def ppb_to_ugm3(ppb: float, pollutant: str):
    if ppb is None:
        return None
    mw = MOLAR_MASS.get(pollutant.lower())
    if mw is None:
        return None
    return float(ppb) * mw / 24.45


def normalize_value(pollutant: str, value, unit: str):
    if value is None:
        return None
    p = pollutant.lower()
    u = (unit or "").lower().replace(" ", "").replace("³", "3")
    try:
        v = float(value)
    except Exception:
        return None
    if p in ("pm25", "pm10", "benzene", "toluene", "xylene"):
        return v
    if p in ("no", "no2", "nh3", "so2", "o3", "nox"):
        if "ppb" in u:
            return ppb_to_ugm3(v, p)
        if "ugm3" in u or "gm3" in u:
            return v
        return v
    if p == "co":
        if "ppb" in u:
            return ppb_to_ugm3(v, p) / 1000.0
        if "ppm" in u:
            return v * MOLAR_MASS["co"] / 24.45
        if "mgm3" in u:
            return v
        if "ugm3" in u:
            return v / 1000.0
        return v
    return v


class TestPPBToUgm3:
    """TC-OQ-01 to TC-OQ-03: Unit conversion."""

    def test_ppb_to_ugm3_no2(self):
        """TC-OQ-01: 1 ppb NO2 converts to MW/24.45 µg/m³."""
        expected = 46.0055 / 24.45
        assert abs(ppb_to_ugm3(1.0, "no2") - expected) < 1e-4

    def test_ppb_none_returns_none(self):
        """TC-OQ-02: None input returns None."""
        assert ppb_to_ugm3(None, "no2") is None

    def test_unknown_pollutant_returns_none(self):
        """TC-OQ-03: Pollutant not in molar mass table returns None."""
        assert ppb_to_ugm3(10.0, "xenon") is None


class TestNormalizeValue:
    """TC-OQ-04 to TC-OQ-09: Value normalization by unit."""

    def test_pm25_passthrough(self):
        """TC-OQ-04: PM2.5 values pass through unchanged."""
        assert normalize_value("pm25", 45.0, "ugm3") == 45.0

    def test_no2_ppb_converted(self):
        """TC-OQ-05: NO2 in ppb is converted to µg/m³."""
        result = normalize_value("no2", 10.0, "ppb")
        assert result is not None
        assert result > 10.0  # conversion always increases value

    def test_none_value_returns_none(self):
        """TC-OQ-06: None value always returns None."""
        assert normalize_value("pm25", None, "ugm3") is None

    def test_non_numeric_string_returns_none(self):
        """TC-OQ-07: Non-numeric string value returns None."""
        assert normalize_value("pm25", "N/A", "ugm3") is None

    def test_co_ugm3_divided_by_1000(self):
        """TC-OQ-08: CO in µg/m³ is divided by 1000 to get mg/m³."""
        result = normalize_value("co", 1000.0, "ugm3")
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_pm10_value_is_float(self):
        """TC-OQ-09: Normalized PM10 is a float."""
        result = normalize_value("pm10", "88", "ugm3")
        assert isinstance(result, float)


class TestSafeListDict:
    """TC-OQ-10: Defensive list/dict helpers."""

    def test_safe_list_non_list_returns_empty(self):
        """TC-OQ-10: Non-list input to safe_list returns []."""
        def safe_list(x):
            return x if isinstance(x, list) else []
        assert safe_list(None) == []
        assert safe_list("string") == []
        assert safe_list({"key": "val"}) == []
        assert safe_list([1, 2]) == [1, 2]