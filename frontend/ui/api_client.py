import os
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
_TIMEOUT_SHORT   = 10
_TIMEOUT_PREDICT = 90


def _get(path: str, params: dict = None, timeout: int = _TIMEOUT_SHORT) -> dict:
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_health() -> dict:
    return _get("/health")


def get_ready() -> dict:
    return _get("/ready")


def get_cities() -> list[str]:
    return _get("/cities", timeout=20)["cities"]


def get_model_info() -> dict:
    return _get("/model-info")


def get_prediction(city: str, location_id: int) -> dict:
    return _get(
        "/predict",
        params={"city": city, "location_id": location_id},
        timeout=_TIMEOUT_PREDICT,
    )


def is_backend_alive() -> tuple[bool, str]:
    """Returns (alive: bool, message: str)"""
    try:
        data = get_health()
        return True, data.get("status", "ok")
    except ConnectionError:
        return False, "Cannot reach backend — is it running?"
    except Timeout:
        return False, "Backend health check timed out."
    except HTTPError as e:
        return False, f"Backend returned HTTP {e.response.status_code}."
    except Exception as e:
        return False, str(e)