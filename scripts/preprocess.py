import argparse, gc, logging, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

POLLUTANT_COLS = ["pm25", "no", "no2", "nox", "nh3", "co", "so2", "o3",
                  "benzene", "toluene", "xylene", "aqi"]
LAG_HOURS     = [1, 3, 6, 12, 24, 48, 168]
ROLLING_HOURS = [3, 6, 24, 48, 168]
TARGET_COL    = "aqi"
N_AHEAD       = 24

ap = argparse.ArgumentParser()
ap.add_argument("--input",          required=True)
ap.add_argument("--output",         required=True)
ap.add_argument("--null-threshold", type=float, default=0.8)
args = ap.parse_args()

out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Read schema only — zero data loaded into RAM ───────────────────────────
pf_meta   = pq.read_metadata(args.input)
pf_schema = pq.read_schema(args.input)
all_cols  = pf_schema.names
log.info(f"Schema read. Total columns: {len(all_cols)}")

# ── 2. Read ONLY 3 cols to get null rates + station list ──────────────────────
log.info("Reading station list + computing null rates (minimal load) ...")
sample = pd.read_parquet(args.input, columns=["station_id", "city"] + POLLUTANT_COLS)

null_rates = sample.isnull().mean()
drop_cols  = null_rates[null_rates > args.null_threshold].index.tolist()
drop_cols  = [c for c in drop_cols if c != "aqi"]
log.info(f"Dropping high-null columns: {drop_cols}")

feature_cols = [c for c in POLLUTANT_COLS if c not in drop_cols and c in all_cols]
log.info(f"Pollutant columns retained: {feature_cols}")

# Fit city encoder
le = LabelEncoder()
le.fit(sample["city"])
enc_path = out_path.parent / "city_label_encoder.pkl"
with open(enc_path, "wb") as f:
    pickle.dump(le, f)
log.info(f"City mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

stations = sample["station_id"].unique().tolist()
log.info(f"Total stations: {len(stations)}")

# Free sample — done with it
del sample
gc.collect()
log.info("Sample freed from RAM.")

# ── 3. Define columns to load per station ─────────────────────────────────────
weather_cols = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "pressure_msl", "cloud_cover", "wind_speed_10m", "wind_direction_10m"]
meta_cols    = ["timestamp", "station_id", "city", "state",
                "station_name", "latitude", "longitude"]
load_cols    = [c for c in meta_cols + feature_cols + weather_cols if c in all_cols]
target_cols  = [f"aqi_t{h}" for h in range(1, N_AHEAD + 1)]

# ── 4. Stream: one station at a time ──────────────────────────────────────────
writer = None

for i, sid in enumerate(stations):

    # Load only this station — pyarrow row filter, no full scan
    sdf = pd.read_parquet(
        args.input,
        filters=[("station_id", "=", sid)],
        columns=load_cols
    )
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)

    # Impute per column
    sdf[feature_cols] = sdf[feature_cols].ffill().bfill()

    # Lag features
    lag_data = {
        f"{col}_lag{lag}": sdf[col].shift(lag)
        for col in feature_cols for lag in LAG_HOURS
    }

    # Rolling features
    roll_data = {}
    for col in feature_cols:
        for w in ROLLING_HOURS:
            s = sdf[col].shift(1)
            roll_data[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            roll_data[f"{col}_roll{w}_std"]  = s.rolling(w, min_periods=1).std()

    # Temporal features
    sdf["hour_of_day"] = sdf["timestamp"].dt.hour
    sdf["day_of_week"] = sdf["timestamp"].dt.dayofweek
    sdf["month"]       = sdf["timestamp"].dt.month
    sdf["year"]        = sdf["timestamp"].dt.year
    sdf["is_weekend"]  = (sdf["day_of_week"] >= 5).astype(int)
    sdf["season"]      = sdf["month"].map(
        lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5]
                  else 2 if m in [6,7,8,9] else 3
    )
    sdf["hour_sin"]  = np.sin(2 * np.pi * sdf["hour_of_day"] / 24)
    sdf["hour_cos"]  = np.cos(2 * np.pi * sdf["hour_of_day"] / 24)
    sdf["month_sin"] = np.sin(2 * np.pi * sdf["month"] / 12)
    sdf["month_cos"] = np.cos(2 * np.pi * sdf["month"] / 12)
    sdf["city_encoded"] = le.transform(sdf["city"])

    # Target columns
    target_data = {
        f"aqi_t{h}": sdf[TARGET_COL].shift(-h)
        for h in range(1, N_AHEAD + 1)
    }

    # Single concat — no fragmentation
    sdf = pd.concat(
        [sdf,
         pd.DataFrame(lag_data,    index=sdf.index),
         pd.DataFrame(roll_data,   index=sdf.index),
         pd.DataFrame(target_data, index=sdf.index)],
        axis=1
    )
    del lag_data, roll_data, target_data
    gc.collect()

    # Drop null targets
    sdf = sdf.dropna(subset=target_cols).reset_index(drop=True)

    # Write to parquet — never accumulate in RAM
    table = pa.Table.from_pandas(sdf, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(out_path), table.schema)
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
log.info(f"Total rows: {pq.read_metadata(str(out_path)).num_rows}")
log.info("=" * 55)
