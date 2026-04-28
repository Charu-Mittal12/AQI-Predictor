import json
import requests
from datetime import datetime, timezone
from app.utils.config import OPEN_METEO_BASE_URL, GEOCODE_URL, CITY_COORDS_PATH
from app.utils.logger import get_logger

log = get_logger(__name__)
_CITY_COORDS = None


def _load_coords() -> dict:
    with open(CITY_COORDS_PATH) as f:
        return json.load(f)


def get_coords(city: str) -> tuple:
    global _CITY_COORDS
    if _CITY_COORDS is None:
        _CITY_COORDS = _load_coords()
    if city in _CITY_COORDS:
        entry = _CITY_COORDS[city]
        return (entry["lat"], entry["lon"])
    log.warning(f"{city} not in city_coords.json — geocoding on the fly.")
    r = requests.get(
        GEOCODE_URL,
        params={"name": city, "count": 1, "language": "en"},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        log.error(f"Could not geocode {city}, using Chennai fallback.")
        return (13.0827, 80.2707)
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    _CITY_COORDS[city] = {"lat": lat, "lon": lon}
    return (lat, lon)


def get_weather(city: str) -> dict:
    lat, lon = get_coords(city)
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,pressure_msl,cloud_cover",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    curr = resp.json().get("current", {})
    return {
        "temperature_2m": curr.get("temperature_2m"),
        "relative_humidity_2m": curr.get("relative_humidity_2m"),
        "wind_speed_10m": curr.get("wind_speed_10m"),
        "wind_direction_10m": curr.get("wind_direction_10m"),
        "precipitation": curr.get("precipitation"),
        "pressure_msl": curr.get("pressure_msl"),
        "cloud_cover": curr.get("cloud_cover"),
    }


def get_live_inputs(city: str) -> dict:
    weather_data = get_weather(city)
    return {
        **weather_data,
        "city": city,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00"),
    }
