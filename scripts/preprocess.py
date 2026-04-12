import argparse, gc, logging, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyarrow as pa
import pyarrow.parquet as pq


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


POLLUTANT_COLS        = ["pm25", "no", "no2", "nox", "nh3", "co", "so2", "o3",
                          "benzene", "toluene", "xylene", "aqi"]
LAG_HOURS             = [1, 3, 6, 12, 24, 48, 168]
ROLLING_HOURS         = [3, 6, 24, 48, 168]
LAG_HOURS_WEATHER     = [1, 3, 6, 24, 48]
ROLLING_HOURS_WEATHER = [6, 24, 48]
WEATHER_LAG_COLS      = ["temperature_2m", "wind_speed_10m",
                          "relative_humidity_2m", "precipitation"]
TARGET_COL            = "aqi"
N_AHEAD               = 24


ap = argparse.ArgumentParser()
ap.add_argument("--input",          required=True)
ap.add_argument("--output",         required=True)
ap.add_argument("--null-threshold", type=float, default=0.8)
args = ap.parse_args()

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)


pf_schema = pq.read_schema(args.input)
all_cols  = pf_schema.names
log.info(f"Schema read. Total columns: {len(all_cols)}")

log.info("Reading station list + null rates ...")
sample = pd.read_parquet(args.input, columns=["station_id", "city"] + POLLUTANT_COLS)

null_rates   = sample.isnull().mean()
drop_cols    = null_rates[null_rates > args.null_threshold].index.tolist()
drop_cols    = [c for c in drop_cols if c != "aqi"]
feature_cols = [c for c in POLLUTANT_COLS if c not in drop_cols and c in all_cols]
log.info(f"Dropping high-null cols: {drop_cols}")
log.info(f"Pollutant cols retained: {feature_cols}")

le = LabelEncoder()
le.fit(sample["city"])
enc_path = out_path.parent / "city_label_encoder.pkl"
with open(enc_path, "wb") as f:
    pickle.dump(le, f)
log.info(f"City mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

stations = sample["station_id"].unique().tolist()
log.info(f"Total stations: {len(stations)}")

del sample
gc.collect()
log.info("Sample freed from RAM.")


weather_cols           = ["temperature_2m", "relative_humidity_2m", "precipitation",
                           "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
meta_cols              = ["timestamp", "station_id", "city", "state",
                           "station_name", "latitude", "longitude"]
load_cols              = [c for c in meta_cols + feature_cols + weather_cols if c in all_cols]
target_cols            = [f"aqi_t{h}" for h in range(1, N_AHEAD + 1)]
avail_weather_lag_cols = [c for c in WEATHER_LAG_COLS if c in all_cols]
log.info(f"Weather lag cols: {avail_weather_lag_cols}")


writer        = None
writer_schema = None

for i, sid in enumerate(stations):

    sdf = pd.read_parquet(
        args.input,
        filters=[("station_id", "=", sid)],
        columns=load_cols
    )
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)

    # impute + cast to float32 immediately — halves memory
    sdf[feature_cols] = sdf[feature_cols].ffill().bfill().astype("float32")
    for wc in avail_weather_lag_cols:
        sdf[wc] = sdf[wc].ffill().bfill().astype("float32")

    # pollutant lags — direct assignment, no concat
    for col in feature_cols:
        for lag in LAG_HOURS:
            sdf[f"{col}_lag{lag}"] = sdf[col].shift(lag)

    # pollutant rolling
    for col in feature_cols:
        for w in ROLLING_HOURS:
            s = sdf[col].shift(1)
            sdf[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            sdf[f"{col}_roll{w}_std"]  = s.rolling(w, min_periods=1).std()
            if w in [24, 168]:
                sdf[f"{col}_roll{w}_max"] = s.rolling(w, min_periods=1).max()
                sdf[f"{col}_roll{w}_min"] = s.rolling(w, min_periods=1).min()

    # weather lags — direct assignment
    for col in avail_weather_lag_cols:
        for lag in LAG_HOURS_WEATHER:
            sdf[f"{col}_lag{lag}"] = sdf[col].shift(lag)

    # weather rolling
    for col in avail_weather_lag_cols:
        for w in ROLLING_HOURS_WEATHER:
            s = sdf[col].shift(1)
            sdf[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            sdf[f"{col}_roll{w}_std"]  = s.rolling(w, min_periods=1).std()

    # aqi rate of change
    sdf["aqi_diff1"]  = sdf["aqi"].diff(1)
    sdf["aqi_diff3"]  = sdf["aqi"].diff(3)
    sdf["aqi_diff24"] = sdf["aqi"].diff(24)

    # temporal features
    sdf["hour_of_day"] = sdf["timestamp"].dt.hour
    sdf["day_of_week"] = sdf["timestamp"].dt.dayofweek
    sdf["month"]       = sdf["timestamp"].dt.month
    sdf["year"]        = sdf["timestamp"].dt.year
    sdf["is_weekend"]  = (sdf["day_of_week"] >= 5).astype(int)
    sdf["season"]      = sdf["month"].map(
        lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5]
                  else 2 if m in [6,7,8,9] else 3
    )
    sdf["hour_sin"]        = np.sin(2 * np.pi * sdf["hour_of_day"] / 24)
    sdf["hour_cos"]        = np.cos(2 * np.pi * sdf["hour_of_day"] / 24)
    sdf["month_sin"]       = np.sin(2 * np.pi * sdf["month"] / 12)
    sdf["month_cos"]       = np.cos(2 * np.pi * sdf["month"] / 12)
    sdf["day_of_year"]     = sdf["timestamp"].dt.dayofyear
    sdf["day_of_year_sin"] = np.sin(2 * np.pi * sdf["day_of_year"] / 365)
    sdf["day_of_year_cos"] = np.cos(2 * np.pi * sdf["day_of_year"] / 365)

    if "wind_direction_10m" in sdf.columns:
        sdf["wind_dir_sin"] = np.sin(2 * np.pi * sdf["wind_direction_10m"] / 360)
        sdf["wind_dir_cos"] = np.cos(2 * np.pi * sdf["wind_direction_10m"] / 360)

    sdf["city_encoded"] = le.transform(sdf["city"])

    # interaction features
    if "pm25" in feature_cols and "no2" in feature_cols:
        sdf["pm25_x_no2"]  = sdf["pm25"] * sdf["no2"]
    if "pm25" in feature_cols and "wind_speed_10m" in sdf.columns:
        sdf["pm25_x_wind"] = sdf["pm25"] * sdf["wind_speed_10m"]
    if "no2" in feature_cols and "temperature_2m" in sdf.columns:
        sdf["no2_x_temp"]  = sdf["no2"] * sdf["temperature_2m"]

    # targets — direct assignment
    for h in range(1, N_AHEAD + 1):
        sdf[f"aqi_t{h}"] = sdf[TARGET_COL].shift(-h)

    gc.collect()

    sdf = sdf.dropna(subset=target_cols).reset_index(drop=True)

    # fix null-type string cols
    for col in ["station_id", "city", "state", "station_name"]:
        if col in sdf.columns:
            sdf[col] = sdf[col].astype(str)

    table = pa.Table.from_pandas(sdf, preserve_index=False)

    if writer is None:
        new_fields = []
        for field in table.schema:
            if pa.types.is_null(field.type):
                new_fields.append(field.with_type(pa.string()))
            else:
                new_fields.append(field)
        writer_schema = pa.schema(new_fields)
        writer = pq.ParquetWriter(str(out_path), writer_schema)

    table = table.cast(writer_schema)
    writer.write_table(table)

    del sdf, table
    gc.collect()

    if (i + 1) % 10 == 0:
        log.info(f"  {i+1}/{len(stations)} stations done ...")


if writer:
    writer.close()

log.info("=" * 55)
log.info(f"SAVED -> {out_path}")
log.info(f"Total row groups: {pq.read_metadata(str(out_path)).num_row_groups}")
log.info(f"Total rows      : {pq.read_metadata(str(out_path)).num_rows}")
log.info("=" * 55)
