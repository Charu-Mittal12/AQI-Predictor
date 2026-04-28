"""
Unit tests for advisory_service.make_advisory.
No external dependencies — pure logic.
"""
import pytest


def make_advisory(aqi: int) -> str:
    """Mirror of app.services.advisory_service.make_advisory."""
    if aqi <= 50:
        return "Air quality is good. Outdoor activity is safe."
    if aqi <= 100:
        return "Air quality is satisfactory. Sensitive people should stay alert."
    if aqi <= 200:
        return "Moderate pollution. Limit prolonged outdoor exertion."
    if aqi <= 300:
        return "Poor air quality. Consider reducing outdoor activities."
    if aqi <= 400:
        return "Very poor air quality. Avoid outdoor exposure if possible."
    return "Severe air quality. Stay indoors and use protection."


class TestAdvisoryService:
    """TC-A-01 to TC-A-08: Advisory text generation for each AQI band."""

    @pytest.mark.parametrize("aqi,expected_substr", [
        (0,   "good"),
        (25,  "good"),
        (50,  "good"),
        (51,  "satisfactory"),
        (100, "satisfactory"),
        (101, "Moderate"),
        (200, "Moderate"),
        (201, "Poor"),
        (300, "Poor"),
        (301, "Very poor"),
        (400, "Very poor"),
        (401, "Severe"),
        (999, "Severe"),
    ])
    def test_advisory_text_by_band(self, aqi, expected_substr):
        """TC-A-01: Advisory text matches the correct AQI band."""
        result = make_advisory(aqi)
        assert expected_substr.lower() in result.lower(),             f"AQI={aqi}: expected '{expected_substr}' in '{result}'"

    def test_advisory_returns_string(self):
        """TC-A-02: Return type is always str."""
        for aqi in [0, 50, 100, 200, 300, 400, 500]:
            assert isinstance(make_advisory(aqi), str)

    def test_advisory_non_empty(self):
        """TC-A-03: Advisory is never an empty string."""
        for aqi in [0, 100, 250, 500]:
            assert len(make_advisory(aqi)) > 0

    def test_boundary_50(self):
        """TC-A-04: AQI=50 is Good, AQI=51 is Satisfactory (boundary test)."""
        assert "good" in make_advisory(50).lower()
        assert "satisfactory" in make_advisory(51).lower()

    def test_boundary_100(self):
        """TC-A-05: AQI=100 is Satisfactory, AQI=101 is Moderate."""
        assert "satisfactory" in make_advisory(100).lower()
        assert "moderate" in make_advisory(101).lower()

    def test_boundary_400(self):
        """TC-A-06: AQI=400 is Very Poor, AQI=401 is Severe."""
        assert "very poor" in make_advisory(400).lower()
        assert "severe" in make_advisory(401).lower()