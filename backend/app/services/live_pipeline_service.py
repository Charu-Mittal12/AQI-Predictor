from datetime import datetime, timezone
from math import isnan

import pandas as pd

from app.services.db_service import get_latest_reading, store_readings_batch
from app.services.openaq_service import fetch_city_history_range, get_location_latest_timestamp
from app.services.weather_service import fetch_weather_history
from app.utils.config import LIVE_HISTORY_HOURS
from app.utils.logger import get_logger

log = get_logger(__name__)

POLLUTANT_COLUMNS = (
    "pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3",
    "benzene", "toluene", "xylene",
)

DB_COLUMNS = (
    "city", "timestamp", "station_id", "state", "station_name",
    "latitude", "longitude",
    *POLLUTANT_COLUMNS,
    "aqi",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "precipitation", "pressure_msl", "cloud_cover",
)

AQI_BREAKPOINTS = {
    "pm25":  [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400),(251,1000,401,500)],
    "pm10":  [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400),(431,2000,401,500)],
    "no2":   [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400),(401,1000,401,500)],
    "so2":   [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400),(1601,3000,401,500)],
    "co":    [(0,1,0,50),(1.1,2,51,100),(2.1,10,101,200),(10.1,17,201,300),(17.1,34,301,400),(34.1,100,401,500)],
    "o3":    [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400),(749,2000,401,500)],
    "nh3":   [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1201,1800,301,400),(1801,3000,401,500)],
}


def _coerce_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if isnan(value):
            return None
        return value
    except Exception:
        return None


def _estimate_sub_index(pollutant: str, value):
    value = _coerce_float(value)
    if value is None or pollutant not in AQI_BREAKPOINTS:
        return None
    for bp_low, bp_high, aqi_low, aqi_high in AQI_BREAKPOINTS[pollutant]:
        if bp_low <= value <= bp_high:
            if bp_high == bp_low:
                return float(aqi_high)
            return (aqi_high - aqi_low) / (bp_high - bp_low) * (value - bp_low) + aqi_low
    last = AQI_BREAKPOINTS[pollutant][-1]
    bp_low, bp_high, aqi_low, aqi_high = last
    if value > bp_high:
        return float(aqi_high)
    return None


def _estimate_aqi(row: pd.Series) -> float | None:
    candidates = [_estimate_sub_index(p, row.get(p)) for p in AQI_BREAKPOINTS]
    candidates = [c for c in candidates if c is not None]
    return float(max(candidates)) if candidates else None


def collect_live_city_history(
    city: str,
    hours_back: int = LIVE_HISTORY_HOURS,
    force_refresh: bool = False,
    location_id: int | None = None,
) -> dict:
    if location_id is None:
        raise ValueError("location_id is required")

    now = pd.Timestamp.now(tz="UTC").floor("h")

    try:
        openaq_latest_ts = get_location_latest_timestamp(location_id)
        end_ts = openaq_latest_ts if openaq_latest_ts is not None else now
    except Exception:
        end_ts = now

    # ── FIX 1: use location_id when computing start_ts ────────────────────────
    # Previously called get_latest_reading(city) which could return a row from
    # a *different* station in the same city, making the fetch window wrong.
    latest = get_latest_reading(city, location_id=location_id)

    if latest and latest.get("timestamp") and not force_refresh:
        last_ts = pd.to_datetime(latest["timestamp"], utc=True, errors="coerce")
        if pd.notna(last_ts):
            start_ts = (last_ts + pd.Timedelta(hours=1)).floor("h")
        else:
            start_ts = end_ts - pd.Timedelta(hours=hours_back)
    else:
        start_ts = end_ts - pd.Timedelta(hours=hours_back)

    if pd.isna(start_ts) or start_ts >= end_ts:
        return {"city": city, "status": "cached", "rows_written": 0}

    start_iso = start_ts.strftime("%Y-%m-%dT%H:00:00Z")
    end_iso   = end_ts.strftime("%Y-%m-%dT%H:00:00Z")

    history_bundle = fetch_city_history_range(
        city=city,
        start_iso=start_iso,
        end_iso=end_iso,
        location_id=location_id,
        hours_back=hours_back,
    )

    pollution_df = build_pollution_frame(history_bundle)
    live_df      = merge_live_inputs(city, pollution_df)

    if live_df.empty:
        live_df = pollution_df.copy()

    # ── FIX 2: dedup by (city, station_id, timestamp) not (city, timestamp) ────
    # Previously drop_duplicates on (city, timestamp) collapsed multiple stations
    # in the same city into one row per hour, silently dropping data.
    live_df = (
        live_df
        .sort_values("timestamp")
        .drop_duplicates(subset=["city", "station_id", "timestamp"], keep="last")
    )

    live_df = live_df.tail(hours_back + 24).copy()
    live_df["timestamp"] = (
        pd.to_datetime(live_df["timestamp"], utc=True, errors="coerce")
        .dt.strftime("%Y-%m-%dT%H:00:00Z")
    )

    payload_rows = live_df.reindex(columns=DB_COLUMNS).to_dict(orient="records")
    store_readings_batch(payload_rows)

    return {
        "city": city,
        "status": "updated",
        "rows_written": len(payload_rows),
        "station_id":   payload_rows[-1]["station_id"]   if payload_rows else None,
        "station_name": payload_rows[-1]["station_name"] if payload_rows else None,
        "location_id": history_bundle.get("location_id"),
    }


def build_pollution_frame(history_bundle: dict) -> pd.DataFrame:
    history = pd.DataFrame(history_bundle["history"])
    if history.empty:
        raise ValueError(
            f"No OpenAQ history rows available for {history_bundle['city']}"
        )

    history["timestamp"] = (
        pd.to_datetime(history["measurement_time"], utc=True, errors="coerce")
        .dt.floor("h")
    )
    history = history.dropna(subset=["timestamp", "parameter", "value"])

    pivot = (
        history
        .groupby(["timestamp", "parameter"], as_index=False)["value"]
        .mean()
        .pivot(index="timestamp", columns="parameter", values="value")
        .reset_index()
    )
    pivot.columns.name = None

    coords             = history_bundle.get("coordinates") or {}
    pivot["city"]      = history_bundle["city"]
    pivot["station_id"] = f"openaq{history_bundle['location_id']}"
    pivot["state"]     = None
    pivot["station_name"] = history_bundle["location_name"]
    pivot["latitude"]  = coords.get("latitude")
    pivot["longitude"] = coords.get("longitude")

    for col in POLLUTANT_COLUMNS:
        if col not in pivot.columns:
            pivot[col] = None

    pivot["aqi"] = pivot.apply(_estimate_aqi, axis=1)
    return pivot


def merge_live_inputs(city: str, pollution_df: pd.DataFrame) -> pd.DataFrame:
    start = pollution_df["timestamp"].min().to_pydatetime()
    end   = max(pollution_df["timestamp"].max().to_pydatetime(), datetime.now(timezone.utc))

    weather_df = fetch_weather_history(city, start=start, end=end)
    weather_df["timestamp"] = (
        pd.to_datetime(weather_df["timestamp"], utc=True, errors="coerce")
        .dt.floor("h")
    )
    weather_df = weather_df.dropna(subset=["timestamp"])

    merged = pollution_df.merge(weather_df, on=["city", "timestamp"], how="left")
    merged = merged.sort_values("timestamp").drop_duplicates(
        subset=["city", "station_id", "timestamp"], keep="last"
    )
    return merged