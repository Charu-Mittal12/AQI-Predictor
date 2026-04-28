"""
Tests for /health and /ready endpoints.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient


def _make_app():
    from fastapi import FastAPI
    from app.routers.health import router
    app = FastAPI()
    app.include_router(router)
    return app


class TestHealthEndpoint:
    """TC-H-01 to TC-H-03: Health endpoint behaviour."""

    def test_health_returns_200(self):
        """TC-H-01: /health always returns HTTP 200 with status ok."""
        client = TestClient(_make_app())
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_response_schema(self):
        """TC-H-02: Response contains exactly the 'status' key."""
        client = TestClient(_make_app())
        data = client.get("/health").json()
        assert "status" in data
        assert isinstance(data["status"], str)

    def test_health_is_fast(self):
        """TC-H-03: /health responds in under 500 ms (no DB or model calls)."""
        import time
        client = TestClient(_make_app())
        start = time.perf_counter()
        client.get("/health")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Health endpoint too slow: {elapsed:.3f}s"


class TestReadyEndpoint:
    """TC-R-01 to TC-R-03: Readiness endpoint with model state."""

    def test_ready_model_loaded(self):
        """TC-R-01: /ready returns status=ready when model is loaded."""
        with patch("app.routers.health.is_model_ready", return_value=True):
            client = TestClient(_make_app())
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is True
            assert data["status"] == "ready"

    def test_ready_model_not_loaded(self):
        """TC-R-02: /ready returns status=degraded when model is not loaded."""
        with patch("app.routers.health.is_model_ready", return_value=False):
            client = TestClient(_make_app())
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is False
            assert data["status"] == "degraded"

    def test_ready_schema(self):
        """TC-R-03: /ready response contains status and model_loaded keys."""
        with patch("app.services.model_service.is_model_ready", return_value=True):
            client = TestClient(_make_app())
            data = client.get("/ready").json()
            assert {"status", "model_loaded"} == set(data.keys())