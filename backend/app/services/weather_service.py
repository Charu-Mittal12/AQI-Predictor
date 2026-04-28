from datetime import datetime, timezone

import pandas as pd
import requests

from app.services.live_data_service import get_coords
from app.utils.config import OPEN_METEO_ARCHIVE_URL
from app.utils.logger import get_logger

log = get_logger(__name__)

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]


def fetch_weather_history(city: str, start: datetime, end: datetime) -> pd.DataFrame:
    lat, lon = get_coords(city)

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)

    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(WEATHER_COLUMNS),
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()

    hourly = response.json().get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError(f"No weather history returned for {city}")

    weather_df = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], utc=True, errors="coerce")
    weather_df = weather_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    weather_df["city"] = city

    log.info(
        "Fetched weather history for %s: %s rows from %s to %s",
        city,
        len(weather_df),
        weather_df["timestamp"].min(),
        weather_df["timestamp"].max(),
    )
    return weather_df