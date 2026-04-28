"""
Tests for /cities endpoint.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient


def _make_app():
    from fastapi import FastAPI
    from app.routers.cities import router
    app = FastAPI()
    app.include_router(router)
    return app


SUPPORTED = ["Chennai", "Delhi", "Mumbai", "Kolkata", "Ahmedabad"]


class TestCitiesEndpoint:
    """TC-C-01 to TC-C-04: Cities listing endpoint."""

    def test_cities_returns_200(self):
        """TC-C-01: /cities returns HTTP 200."""
        with patch("app.services.city_service.get_supported_cities", return_value=SUPPORTED):
            client = TestClient(_make_app())
            assert client.get("/cities").status_code == 200

    def test_cities_returns_list(self):
        """TC-C-02: Response body has a 'cities' key containing a list."""
        with patch("app.services.city_service.get_supported_cities", return_value=SUPPORTED):
            client = TestClient(_make_app())
            data = client.get("/cities").json()
            assert "cities" in data
            assert isinstance(data["cities"], list)

    def test_cities_contains_expected_values(self):
        """TC-C-03: Known cities appear in the response."""
        with patch("app.services.city_service.get_supported_cities", return_value=SUPPORTED):
            client = TestClient(_make_app())
            cities = client.get("/cities").json()["cities"]
            assert "Chennai" in cities
            assert "Delhi" in cities

    def test_cities_not_empty(self):
        """TC-C-04: At least one city is always returned."""
        with patch("app.services.city_service.get_supported_cities", return_value=SUPPORTED):
            client = TestClient(_make_app())
            cities = client.get("/cities").json()["cities"]
            assert len(cities) > 0