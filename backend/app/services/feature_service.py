import json
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from app.services.db_service import get_recent_readings
from app.utils.config import CITY_ENCODER_PATH, FEATURE_COLUMNS_PATH
from app.utils.logger import get_logger

log = get_logger(__name__)

POLLUTANT_COLS = (
    "pm25", "no", "no2", "nox", "nh3", "co", "so2", "o3",
    "benzene", "toluene", "xylene", "aqi",
)
DISPLAY_ONLY_COLS = ("pm10",)
LAG_HOURS         = (1, 3, 6, 12, 24, 48, 168)
ROLLING_HOURS     = (3, 6, 24, 48, 168)
LAG_HOURS_WEATHER     = (1, 3, 6, 24, 48)
ROLLING_HOURS_WEATHER = (6, 24, 48)
WEATHER_LAG_COLS  = ("temperature_2m", "wind_speed_10m", "relative_humidity_2m", "precipitation")

MIN_LIVE_HISTORY_HOURS = 168


@lru_cache(maxsize=1)
def get_feature_columns() -> list[str]:
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_city_encoder():
    with open(CITY_ENCODER_PATH, "rb") as f:
        return pickle.load(f)


def weekly_trend(df: pd.DataFrame) -> list[int]:
    if "aqi" not in df.columns or df["aqi"].dropna().empty:
        return [0] * 7
    trend_df = df[["timestamp", "aqi"]].dropna().copy()

    # Convert UTC timestamps to IST before grouping by day
    trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], utc=True)
    trend_df["day"] = trend_df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.floor("D")

    daily = (
        trend_df.groupby("day", as_index=False)["aqi"]
        .mean()
        .sort_values("day")
        .tail(7)["aqi"]
        .round()
        .astype(int)
        .tolist()
    )
    if not daily:
        daily = [0]
    if len(daily) < 7:
        daily = [daily[0]] * (7 - len(daily)) + daily
    return daily

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in POLLUTANT_COLS + DISPLAY_ONLY_COLS + WEATHER_LAG_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in POLLUTANT_COLS + DISPLAY_ONLY_COLS + WEATHER_LAG_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].ffill().bfill().fillna(0.0).astype(np.float32)
    return df


def build_live_payload(
    city: str,
    hours: int = 220,
    location_id: int | None = None,
) -> dict:
    
    rows = get_recent_readings(city=city, hours=hours, location_id=location_id)

    if len(rows) < MIN_LIVE_HISTORY_HOURS:
        raise ValueError(
            f"Only {len(rows)} live rows available for {city} "
            f"(location_id={location_id}); need at least {MIN_LIVE_HISTORY_HOURS}."
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) < MIN_LIVE_HISTORY_HOURS:
        raise ValueError(
            f"Only {len(df)} valid timestamped live rows available for {city} "
            f"(location_id={location_id}); need at least {MIN_LIVE_HISTORY_HOURS}."
        )

    df = fill_missing(df)

    feature_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    for col in feature_cols:
        for lag in LAG_HOURS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        shifted = df[col].shift(1)
        for window in ROLLING_HOURS:
            rolling = shifted.rolling(window, min_periods=1)
            df[f"{col}_roll{window}_mean"] = rolling.mean()
            df[f"{col}_roll{window}_std"]  = rolling.std()
            if window in (24, 168):
                df[f"{col}_roll{window}_max"] = rolling.max()
                df[f"{col}_roll{window}_min"] = rolling.min()

    for col in WEATHER_LAG_COLS:
        if col not in df.columns:
            continue
        for lag in LAG_HOURS_WEATHER:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        shifted = df[col].shift(1)
        for window in ROLLING_HOURS_WEATHER:
            rolling = shifted.rolling(window, min_periods=1)
            df[f"{col}_roll{window}_mean"] = rolling.mean()
            df[f"{col}_roll{window}_std"]  = rolling.std()

    df["aqi_diff1"]  = df["aqi"].diff(1)
    df["aqi_diff3"]  = df["aqi"].diff(3)
    df["aqi_diff24"] = df["aqi"].diff(24)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["year"]        = df["timestamp"].dt.year
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].map(
        lambda m: 0 if m in (12, 1, 2) else 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8, 9) else 3
    )

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_of_year"]     = df["timestamp"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    if "wind_direction_10m" in df.columns:
        df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
        df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

    encoder = get_city_encoder()
    try:
        city_code = int(encoder.transform([city])[0])
    except Exception:
        city_code = 0
    df["city_encoded"] = city_code

    if "pm25" in df.columns and "no2" in df.columns:
        df["pm25_x_no2"]  = df["pm25"] * df["no2"]
    if "pm25" in df.columns and "wind_speed_10m" in df.columns:
        df["pm25_x_wind"] = df["pm25"] * df["wind_speed_10m"]
    if "no2" in df.columns and "temperature_2m" in df.columns:
        df["no2_x_temp"]  = df["no2"] * df["temperature_2m"]

    training_feature_cols = get_feature_columns() 
    latest = df.iloc[-1].copy()

    for col in training_feature_cols:
        if col not in latest.index:
            latest[col] = np.nan

    latest_features = latest[training_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = latest_features.to_numpy(dtype=np.float32).reshape(1, -1)

    raw_data                = latest.iloc[0].where(pd.notna(latest.iloc[0]), None).to_dict() if latest.ndim > 1 else latest.where(pd.notna(latest), None).to_dict()
    raw_data["timestamp"]   = latest["timestamp"].isoformat() if hasattr(latest["timestamp"], "isoformat") else str(latest["timestamp"])
    raw_data["trend_7d"]    = weekly_trend(df)
    raw_data["data_source"] = "live_ingestion_db"
    raw_data["station_id"]   = latest.get("station_id")
    raw_data["station_name"] = latest.get("station_name")
    raw_data["location_id"] = location_id

    log.info(
        "Built live feature payload for %s rows=%d shape=%s",
        city, len(df), X.shape,
    )
    return {"X": X, "raw_data": raw_data}