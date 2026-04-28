import os
import time
import json
import requests
from datetime import datetime, timedelta, timezone
from app.utils.logger import get_logger
from app.services.db_service import get_connection

log = get_logger(__name__)

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
OPENAQ_BASE_URL = os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v3")
HEADERS = {"X-API-Key": OPENAQ_API_KEY, "accept": "application/json"}
RETRY_STATUS_CODES = {429}

TARGET_PARAMS = {
    "pm25", "pm10", "no", "no2", "nox",
    "nh3", "co", "so2", "o3",
    "benzene", "toluene", "xylene"
}

CITY_LOCATION_HINTS = {
    "Delhi": ["R K Puram", "DPCC", "IGI Airport", "Civil Lines"],
    "Mumbai": ["Bandra Kurla Complex", "Mazgaon", "Worli", "Colaba"],
    "Chennai": ["Velachery", "Manali", "Alandur"],
    "Kolkata": ["Rabindra Bharati University", "Bidhannagar", "Victoria"],
    "Ahmedabad": ["SAC ISRO", "Bopal", "Maninagar", "Navrangpura"],
}

MOLAR_MASS = {
    "no": 30.01,
    "no2": 46.0055,
    "nox": 46.0055,
    "nh3": 17.031,
    "co": 28.01,
    "so2": 64.066,
    "o3": 48.00,
}

CANONICAL_UNITS = {
    "pm25": "ug/m3",
    "pm10": "ug/m3",
    "no": "ug/m3",
    "no2": "ug/m3",
    "nox": "ug/m3",
    "nh3": "ug/m3",
    "co": "mg/m3",
    "so2": "ug/m3",
    "o3": "ug/m3",
    "benzene": "ug/m3",
    "toluene": "ug/m3",
    "xylene": "ug/m3",
}


def get_json(url, params=None, retries=2, base_delay=2):
    delay = base_delay
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                sleep_for = int(ra) if ra and ra.isdigit() else delay
                log.warning("OpenAQ rate limited, sleeping %ss (attempt %s/%s)", sleep_for, attempt, retries)
                time.sleep(sleep_for)
                delay = min(delay * 2, 30)
                continue

            if r.status_code in {500, 502, 503, 504}:
                log.warning("OpenAQ server error %s for %s — skipping", r.status_code, url)
                raise RuntimeError(f"OpenAQ server error {r.status_code}")

            r.raise_for_status()
            return r.json()

        except RuntimeError:
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("OpenAQ request failed attempt %s/%s for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)

    raise RuntimeError(f"OpenAQ request failed after {retries} attempts: {last_exc}")


def _safe_dict(x):
    return x if isinstance(x, dict) else {}


def _safe_list(x):
    return x if isinstance(x, list) else []


def ppb_to_ugm3(ppb: float, pollutant: str) -> float | None:
    if ppb is None:
        return None
    try:
        mw = MOLAR_MASS.get(pollutant.lower())
        if mw is None:
            return None
        return float(ppb) * mw / 24.45
    except Exception:
        return None


def ppm_to_mgm3(ppm: float, pollutant: str) -> float | None:
    if ppm is None:
        return None
    try:
        mw = MOLAR_MASS.get(pollutant.lower())
        if mw is None:
            return None
        return float(ppm) * mw / 24.45
    except Exception:
        return None


def normalize_value(pollutant: str, value, unit: str):
    if value is None:
        return None

    p = pollutant.lower()
    u = (unit or "").lower().replace(" ", "").replace("³", "3")

    try:
        v = float(value)
    except Exception:
        return None

    if p in {"pm25", "pm10", "benzene", "toluene", "xylene"}:
        return v

    if p in {"no", "no2", "nh3", "so2", "o3", "nox"}:
        if "ppb" in u:
            return ppb_to_ugm3(v, p)
        if "ug/m3" in u or "µg/m3" in u:
            return v
        return v

    if p == "co":
        if "ppb" in u:
            return ppb_to_ugm3(v, p) / 1000.0
        if "ppm" in u:
            return ppm_to_mgm3(v, p)
        if "mg/m3" in u:
            return v
        if "ug/m3" in u or "µg/m3" in u:
            return v / 1000.0
        return v

    return v


def get_city_location(city: str) -> dict:
    data = get_json(
        f"{OPENAQ_BASE_URL}/locations",
        params={"iso": "IN", "city": city, "limit": 50}
    )
    results = _safe_list(data.get("results"))
    if not results:
        raise ValueError(f"No OpenAQ locations found for {city}")

    hints = CITY_LOCATION_HINTS.get(city, [])
    for hint in hints:
        for loc in results:
            if isinstance(loc, dict) and hint.lower() in loc.get("name", "").lower():
                return loc

    active = [loc for loc in results if isinstance(loc, dict) and _safe_dict(loc.get("datetimeLast")).get("utc")]
    if active:
        active.sort(key=lambda x: _safe_dict(x.get("datetimeLast")).get("utc", ""), reverse=True)
        return active[0]

    first = results[0]
    if not isinstance(first, dict):
        raise ValueError(f"Unexpected location payload for {city}")
    return first


def get_location_latest(location_id: int) -> list:
    data = get_json(f"{OPENAQ_BASE_URL}/locations/{location_id}/latest", params={"limit": 100})
    return _safe_list(data.get("results"))



def get_location_latest_timestamp(location_id: int):
    """
    Returns the most recent measurement UTC timestamp for a given OpenAQ
    location_id as a pandas Timestamp (UTC), or None if unavailable.
    """
    import pandas as pd
    try:
        latest_rows = get_location_latest(location_id)
        timestamps = []
        for row in latest_rows:
            if not isinstance(row, dict):
                continue
            dt = row.get("datetime") or {}
            if isinstance(dt, dict):
                utc = dt.get("utc")
            else:
                utc = None
            if utc:
                ts = pd.to_datetime(utc, utc=True, errors="coerce")
                if pd.notna(ts):
                    timestamps.append(ts)
        if timestamps:
            return max(timestamps).floor("h")
    except Exception as exc:
        log.warning("get_location_latest_timestamp failed for location_id=%s: %s", location_id, exc)
    return None
# ─────────────────────────────────────────────────────────────────────────────


def get_sensor_map(location: dict) -> dict:
    sensor_map = {}
    for s in _safe_list(location.get("sensors")):
        if not isinstance(s, dict):
            continue
        sensor_id = s.get("id")
        parameter = _safe_dict(s.get("parameter"))
        param = parameter.get("name")
        units = parameter.get("units")
        display = parameter.get("displayName")
        if sensor_id and param:
            sensor_map[sensor_id] = {
                "parameter": param,
                "units": units,
                "display_name": display,
            }
    return sensor_map


def normalize_city_hour_row(row, location, sensor_id, meta, city):
    if not isinstance(row, dict):
        return None

    period = row.get("period")
    if isinstance(period, list):
        period = period[0] if period else {}
    period = _safe_dict(period)

    dt_from = period.get("datetimeFrom")
    if isinstance(dt_from, list):
        dt_from = dt_from[0] if dt_from else {}
    dt_from = _safe_dict(dt_from)

    raw_value = row.get("value")
    raw_unit = meta.get("units")
    value = normalize_value(meta.get("parameter"), raw_value, raw_unit)

    loc_coords = _safe_dict(location.get("coordinates"))

    return {
        "city": city,
        "location_id": location.get("id"),
        "location_name": location.get("name"),
        "sensor_id": sensor_id,
        "parameter": meta.get("parameter"),
        "display_name": meta.get("display_name"),
        "units": CANONICAL_UNITS.get(meta.get("parameter"), raw_unit),
        "raw_unit": raw_unit,
        "raw_value": raw_value,
        "value": value,
        "datetime_utc": dt_from.get("utc"),
        "datetime_local": dt_from.get("local"),
        "latitude": loc_coords.get("latitude"),
        "longitude": loc_coords.get("longitude"),
    }


def latest_to_pollutant_map(latest_rows: list, sensor_map: dict) -> dict:
    pollutants = {}
    for row in _safe_list(latest_rows):
        if not isinstance(row, dict):
            continue
        sensor_id = row.get("sensorsId")
        meta = sensor_map.get(sensor_id, {})
        param = meta.get("parameter")
        if not param:
            continue
        pollutants[param] = normalize_value(param, row.get("value"), meta.get("units"))
    return pollutants


def fetch_sensor_hourly_city(sensor_id: int, hours_back: int = 200) -> list:
    dt_from = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    endpoints = [
        f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
        f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements/hourly",
    ]
    last_error = None
    for url in endpoints:
        try:
            data = get_json(url, params={"datetime_from": dt_from, "limit": max(hours_back + 24, 250), "page": 1})
            results = _safe_list(data.get("results"))
            if results:
                return results
        except Exception as e:
            last_error = e
    if last_error:
        raise RuntimeError(f"Hourly fetch failed for sensor {sensor_id}: {last_error}")
    return []


def get_city_history(city: str, hours_back: int = 200, location_id: int = None) -> dict:
    location = get_city_location(city)
    if location_id:
        location["id"] = location_id

    latest = get_location_latest(location.get("id"))
    sensor_map = get_sensor_map(location)

    rows = []
    for sensor_id, meta in sensor_map.items():
        if meta["parameter"] not in TARGET_PARAMS:
            continue
        hourly_rows = fetch_sensor_hourly_city(sensor_id, hours_back=hours_back)
        for row in hourly_rows:
            norm = normalize_city_hour_row(row, location, sensor_id, meta, city)
            if not norm:
                continue
            if norm["parameter"] not in TARGET_PARAMS:
                continue
            if norm["value"] is None:
                continue
            rows.append(norm)

    if not rows:
        raise ValueError(f"No hourly historical rows found for {city} at {location.get('name')}")

    return {
        "city": city,
        "location_id": location.get("id"),
        "location_name": location.get("name"),
        "coordinates": _safe_dict(location.get("coordinates")),
        "latest": latest,
        "history": rows,
    }


def get_live_inputs(city: str, hours_back: int = 200) -> dict:
    location = get_city_location(city)
    latest = get_location_latest(location.get("id"))
    sensor_map = get_sensor_map(location)
    pollutants = latest_to_pollutant_map(latest, sensor_map)
    history_bundle = get_city_history(city, hours_back=hours_back)

    return {
        "source": "OpenAQ",
        "city": city,
        "location_id": location.get("id"),
        "location_name": location.get("name"),
        "coordinates": _safe_dict(location.get("coordinates")),
        "latest": latest,
        "pollutants": pollutants,
        "history": history_bundle["history"],
    }


def fetch_sensor_hours(sensor_id, start_iso, end_iso, limit=1000):
    all_rows = []
    page = 1

    while True:
        data = get_json(
            f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
            params={
                "datetime_from": start_iso,
                "datetime_to": end_iso,
                "limit": limit,
                "page": page,
            },
        )
        results = data.get("results", [])
        if not results:
            break
        all_rows.extend(results)
        if len(results) < limit:
            break
        page += 1

    return all_rows


def normalize_hour_row(row, meta):
    period = row.get("period") or {}
    if isinstance(period, list):
        period = period[0] if period else {}

    dt_from = period.get("datetimeFrom") or {}
    if isinstance(dt_from, list):
        dt_from = dt_from[0] if dt_from else {}

    raw_value = row.get("value")
    raw_unit = meta.get("units")
    parameter = meta["parameter"]

    return {
        "measurement_time": dt_from.get("utc"),
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "value": normalize_value(parameter, raw_value, raw_unit),
        "sensor_id": meta["sensor_id"],
        "parameter": parameter,
        "units": CANONICAL_UNITS.get(parameter, raw_unit),
        "station_id": meta["station_id"],
        "city": meta["city"],
        "openaq_location_id": meta["openaq_location_id"],
    }


def get_station_by_openaq_location_id(openaq_location_id: int) -> dict:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT *
        FROM sensor_registry
        WHERE openaq_location_id = ?
        LIMIT 1
        """,
        (openaq_location_id,),
    ).fetchone()
    conn.close()
    if not row:
        raise ValueError(f"No station registry entry found for openaq_location_id={openaq_location_id}")
    return dict(row)


def fetch_city_history_range(
    city: str,
    start_iso: str,
    end_iso: str,
    location_id: int | None = None,
    hours_back: int = 200,
) -> dict:
    if location_id is None:
        raise ValueError("location_id is required for live fetch")

    station = get_station_by_openaq_location_id(location_id)

    location = {
        "id": station["openaq_location_id"],
        "name": station["openaq_location_name"] or station["station_name"],
        "coordinates": {
            "latitude": station.get("latitude"),
            "longitude": station.get("longitude"),
        },
        "sensors": [],
    }

    latest = get_location_latest(location["id"])

    conn = get_connection()
    sensor_rows = conn.execute(
        """
        SELECT sensor_id, parameter, units, station_id, city, openaq_location_id, openaq_location_name, station_name
        FROM sensor_registry
        WHERE openaq_location_id = ?
        ORDER BY parameter, sensor_id
        """,
        (location_id,),
    ).fetchall()
    conn.close()

    if not sensor_rows:
        available_sensors = station.get("available_sensors")
        if not available_sensors:
            raise ValueError(f"No sensors found in registry for openaq_location_id={location_id}")

        sensor_map = {}
        sensor_list = json.loads(available_sensors) if isinstance(available_sensors, str) else available_sensors
        for s in sensor_list:
            if not isinstance(s, dict):
                continue
            sensor_id = s.get("id")
            parameter = _safe_dict(s.get("parameter")).get("name")
            units = _safe_dict(s.get("parameter")).get("units")
            display_name = _safe_dict(s.get("parameter")).get("displayName")
            if sensor_id and parameter:
                sensor_map[sensor_id] = {
                    "parameter": parameter,
                    "units": units,
                    "display_name": display_name,
                }
    else:
        sensor_map = {
            row["sensor_id"]: {
                "parameter": row["parameter"],
                "units": row["units"],
                "display_name": row["parameter"],
            }
            for row in sensor_rows
        }

    rows = []
    for sensor_id, meta in sensor_map.items():
        if meta["parameter"] not in TARGET_PARAMS:
            continue
        try:
            hourly_rows = fetch_sensor_hours(sensor_id, start_iso, end_iso)
        except Exception as exc:
            log.warning("Hourly range fetch failed for sensor %s: %s", sensor_id, exc)
            continue

        sensor_meta = {
            **meta,
            "sensor_id": sensor_id,
            "station_id": station["station_id"],
            "city": station["city"],
            "openaq_location_id": station["openaq_location_id"],
        }

        for row in hourly_rows:
            norm = normalize_hour_row(row, sensor_meta)
            if not norm or norm["measurement_time"] is None:
                continue
            if norm["parameter"] not in TARGET_PARAMS:
                continue
            if norm["value"] is None:
                continue
            rows.append(norm)

    if not rows:
        raise ValueError(f"No hourly rows found for {city} between {start_iso} and {end_iso}")

    return {
        "city": station["city"],
        "location_id": station["openaq_location_id"],
        "location_name": station["openaq_location_name"] or station["station_name"],
        "coordinates": {
            "latitude": station.get("latitude"),
            "longitude": station.get("longitude"),
        },
        "latest": latest,
        "history": rows,
    }
