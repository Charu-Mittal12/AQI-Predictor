import argparse
import gc
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

POLLUTANT_COLS = [
    "pm25", "no", "no2", "nox", "nh3", "co", "so2", "o3",
    "benzene", "toluene", "xylene", "aqi"
]
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m"
]
WEATHER_LAG_COLS = ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"]
META_COLS = ["timestamp", "station_id", "city", "state", "station_name", "latitude", "longitude"]

TARGET_COL = "aqi"
N_AHEAD = 24

# Reduced feature generation to keep memory manageable
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_HOURS = [6, 24, 48]
LAG_HOURS_WEATHER = [1, 6, 24]
ROLLING_HOURS_WEATHER = [24]
TOP_MISSINGNESS_FLAGS = 5

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--null-threshold", type=float, default=0.65)
ap.add_argument("--corr-threshold", type=float, default=0.98)
ap.add_argument("--max-engineered-features", type=int, default=180)
args = ap.parse_args()

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

def safe_fill_series(s: pd.Series) -> pd.Series:
    s = s.ffill()
    if s.isnull().any():
        med = s.median()
        if pd.isna(med):
            med = 0.0
        s = s.fillna(med)
    return s.astype("float32")

def drop_near_constant_columns(df: pd.DataFrame, cols: list, threshold: float = 0.999):
    keep, drop = [], []
    for c in cols:
        if c not in df.columns:
            continue
        vc = df[c].value_counts(dropna=False, normalize=True)
        if len(vc) <= 1 or (len(vc) > 0 and vc.iloc[0] >= threshold):
            drop.append(c)
        else:
            keep.append(c)
    return keep, drop

def drop_duplicate_numeric_columns(df: pd.DataFrame, cols: list):
    seen = {}
    dup = []
    keep = []
    for c in cols:
        if c not in df.columns:
            continue
        arr = pd.util.hash_pandas_object(df[c].fillna(-999999), index=False).sum()
        if arr in seen:
            dup.append(c)
        else:
            seen[arr] = c
            keep.append(c)
    return keep, dup

def correlation_prune(df: pd.DataFrame, cols: list, threshold: float):
    usable = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(usable) <= 1:
        return usable, []
    corr = df[usable].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    keep = [c for c in usable if c not in to_drop]
    return keep, to_drop

pf_schema = pq.read_schema(args.input)
all_cols = pf_schema.names
log.info(f"Schema read. Total columns: {len(all_cols)}")

log.info("Reading sample for pruning decisions ...")
sample_cols = [c for c in ["station_id", "city"] + POLLUTANT_COLS + WEATHER_COLS if c in all_cols]
sample = pd.read_parquet(args.input, columns=sample_cols)

null_rates = sample.isnull().mean()
drop_cols = null_rates[null_rates > args.null_threshold].index.tolist()
drop_cols = [c for c in drop_cols if c != TARGET_COL]

feature_cols = [c for c in POLLUTANT_COLS if c not in drop_cols and c in all_cols]
avail_weather_cols = [c for c in WEATHER_COLS if c not in drop_cols and c in all_cols]
avail_weather_lag_cols = [c for c in WEATHER_LAG_COLS if c in avail_weather_cols]

feature_cols, dropped_const_poll = drop_near_constant_columns(sample, feature_cols)
avail_weather_cols, dropped_const_weather = drop_near_constant_columns(sample, avail_weather_cols)
drop_cols += dropped_const_poll + dropped_const_weather

log.info(f"Dropping high-null cols: {drop_cols}")
log.info(f"Retained pollutant cols: {feature_cols}")
log.info(f"Retained weather cols: {avail_weather_cols}")

le = LabelEncoder()
le.fit(sample["city"].astype(str))
with open(out_path.parent / "city_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

stations = sample["station_id"].astype(str).unique().tolist()
log.info(f"Total stations: {len(stations)}")

station_stats = (
    sample.assign(station_id=sample["station_id"].astype(str))
    .groupby("station_id")[TARGET_COL]
    .agg(["mean", "median"])
    .rename(columns={"mean": "station_aqi_mean", "median": "station_aqi_median"})
    .fillna(0.0)
    .astype("float32")
)

city_stats = (
    sample.assign(city=sample["city"].astype(str))
    .groupby("city")[TARGET_COL]
    .agg(["mean", "median"])
    .rename(columns={"mean": "city_aqi_mean", "median": "city_aqi_median"})
    .fillna(0.0)
    .astype("float32")
)

missing_flag_candidates = [c for c in feature_cols + avail_weather_cols if c != TARGET_COL]
missing_flag_cols = (
    null_rates.loc[[c for c in missing_flag_candidates if c in null_rates.index]]
    .sort_values(ascending=False)
    .head(TOP_MISSINGNESS_FLAGS)
    .index.tolist()
)
log.info(f"Missingness flags for: {missing_flag_cols}")

with open(out_path.parent / "xgb_preprocess_meta.pkl", "wb") as f:
    pickle.dump(
        {
            "drop_cols": drop_cols,
            "feature_cols": feature_cols,
            "weather_cols": avail_weather_cols,
            "weather_lag_cols": avail_weather_lag_cols,
            "lag_hours": LAG_HOURS,
            "rolling_hours": ROLLING_HOURS,
            "rolling_weather_hours": ROLLING_HOURS_WEATHER,
            "missing_flag_cols": missing_flag_cols,
            "n_ahead": N_AHEAD,
        },
        f,
    )

load_cols = [c for c in META_COLS + feature_cols + avail_weather_cols if c in all_cols]
target_cols = [f"aqi_t{h}" for h in range(1, N_AHEAD + 1)]
writer = None
writer_schema = None
selected_engineered_cols = None

for i, sid in enumerate(stations):
    sdf = pd.read_parquet(
        args.input,
        filters=[("station_id", "=", sid)],
        columns=load_cols,
    )
    if sdf.empty:
        continue

    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)

    for col in missing_flag_cols:
        if col in sdf.columns:
            sdf[f"{col}_was_missing"] = sdf[col].isnull().astype("int8")

    for col in feature_cols:
        sdf[col] = safe_fill_series(sdf[col])
    for col in avail_weather_cols:
        sdf[col] = safe_fill_series(sdf[col])

    ts = sdf["timestamp"]
    sdf["hour_of_day"] = ts.dt.hour.astype("int8")
    sdf["day_of_week"] = ts.dt.dayofweek.astype("int8")
    sdf["month"] = ts.dt.month.astype("int8")
    sdf["is_weekend"] = (sdf["day_of_week"] >= 5).astype("int8")
    sdf["hour_sin"] = np.sin(2 * np.pi * sdf["hour_of_day"] / 24).astype("float32")
    sdf["hour_cos"] = np.cos(2 * np.pi * sdf["hour_of_day"] / 24).astype("float32")
    sdf["day_of_week_sin"] = np.sin(2 * np.pi * sdf["day_of_week"] / 7).astype("float32")
    sdf["day_of_week_cos"] = np.cos(2 * np.pi * sdf["day_of_week"] / 7).astype("float32")
    sdf["month_sin"] = np.sin(2 * np.pi * sdf["month"] / 12).astype("float32")
    sdf["month_cos"] = np.cos(2 * np.pi * sdf["month"] / 12).astype("float32")

    if "wind_direction_10m" in sdf.columns:
        sdf["wind_dir_sin"] = np.sin(2 * np.pi * sdf["wind_direction_10m"] / 360).astype("float32")
        sdf["wind_dir_cos"] = np.cos(2 * np.pi * sdf["wind_direction_10m"] / 360).astype("float32")

    sdf["city_encoded"] = le.transform(sdf["city"].astype(str)).astype("int16")
    sdf["station_id"] = sdf["station_id"].astype(str)
    sdf["station_aqi_mean"] = sdf["station_id"].map(station_stats["station_aqi_mean"]).fillna(0).astype("float32")
    sdf["station_aqi_median"] = sdf["station_id"].map(station_stats["station_aqi_median"]).fillna(0).astype("float32")
    sdf["city_aqi_mean"] = sdf["city"].astype(str).map(city_stats["city_aqi_mean"]).fillna(0).astype("float32")
    sdf["city_aqi_median"] = sdf["city"].astype(str).map(city_stats["city_aqi_median"]).fillna(0).astype("float32")

    for col in feature_cols:
        for lag in LAG_HOURS:
            sdf[f"{col}_lag{lag}"] = sdf[col].shift(lag).astype("float32")
        shifted = sdf[col].shift(1)
        for w in ROLLING_HOURS:
            roll = shifted.rolling(w, min_periods=1)
            sdf[f"{col}_roll{w}_mean"] = roll.mean().astype("float32")
            sdf[f"{col}_roll{w}_std"] = roll.std().astype("float32")

    for col in avail_weather_lag_cols:
        for lag in LAG_HOURS_WEATHER:
            sdf[f"{col}_lag{lag}"] = sdf[col].shift(lag).astype("float32")
        shifted = sdf[col].shift(1)
        for w in ROLLING_HOURS_WEATHER:
            roll = shifted.rolling(w, min_periods=1)
            sdf[f"{col}_roll{w}_mean"] = roll.mean().astype("float32")

    sdf["aqi_diff1"] = sdf["aqi"].diff(1).astype("float32")
    sdf["aqi_diff24"] = sdf["aqi"].diff(24).astype("float32")

    if "pm25" in sdf.columns and "no2" in sdf.columns:
        sdf["pm25_x_no2"] = (sdf["pm25"] * sdf["no2"]).astype("float32")

    for h in range(1, N_AHEAD + 1):
        sdf[f"aqi_t{h}"] = sdf[TARGET_COL].shift(-h).astype("float32")

    numeric_cols = sdf.select_dtypes(include=[np.number]).columns.tolist()
    feature_like_numeric = [c for c in numeric_cols if c not in target_cols]
    sdf[feature_like_numeric] = sdf[feature_like_numeric].ffill()
    sdf[feature_like_numeric] = sdf[feature_like_numeric].fillna(0)
    sdf = sdf.dropna(subset=target_cols).reset_index(drop=True)

    base_keep = [c for c in META_COLS if c in sdf.columns] + target_cols
    candidate_engineered = [c for c in sdf.columns if c not in base_keep]

    if selected_engineered_cols is None:
        keep1, dup_drop = drop_duplicate_numeric_columns(sdf, candidate_engineered)
        keep2, corr_drop = correlation_prune(sdf, keep1, args.corr_threshold)
        selected_engineered_cols = keep2[:args.max_engineered_features]
        log.info(f"Duplicate-drop: {len(dup_drop)} cols")
        log.info(f"Correlation-drop: {len(corr_drop)} cols")
        log.info(f"Final engineered feature cap: {len(selected_engineered_cols)} cols")

    final_cols = [c for c in META_COLS if c in sdf.columns] + selected_engineered_cols + target_cols
    sdf = sdf[final_cols].copy()

    for col in ["station_id", "city", "state", "station_name"]:
        if col in sdf.columns:
            sdf[col] = sdf[col].astype(str)

    sdf = pd.DataFrame(sdf.to_dict("series"))
    table = pa.Table.from_pandas(sdf, preserve_index=False)

    if writer is None:
        new_fields = []
        for field in table.schema:
            new_fields.append(field.with_type(pa.string()) if pa.types.is_null(field.type) else field)
        writer_schema = pa.schema(new_fields)
        writer = pq.ParquetWriter(str(out_path), writer_schema)

    writer.write_table(table.cast(writer_schema))

    del sdf, table
    gc.collect()

    if (i + 1) % 10 == 0:
        log.info(f"{i + 1}/{len(stations)} stations done ...")

if writer:
    writer.close()

meta = pq.read_metadata(str(out_path))
log.info("=" * 60)
log.info(f"SAVED -> {out_path}")
log.info(f"Total row groups: {meta.num_row_groups}")
log.info(f"Total rows      : {meta.num_rows}")
log.info("=" * 60)
