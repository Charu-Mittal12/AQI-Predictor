import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input",     required=True)
ap.add_argument("--kaggle-st", required=True)
ap.add_argument("--config-st", required=True)
ap.add_argument("--mode",      required=True, choices=["all", "single"])
ap.add_argument("--city",      default=None)
ap.add_argument("--start",     required=True)
ap.add_argument("--end",       required=True)
ap.add_argument("--output",    required=True)
args = ap.parse_args()

df = pd.read_csv(args.input, low_memory=False)
df = df.rename(columns={
    "StationId":  "station_id",  "Datetime":   "timestamp",
    "PM2.5":      "pm25",        "PM10":       "pm10",
    "NO":         "no",          "NO2":        "no2",
    "NOx":        "nox",         "NH3":        "nh3",
    "CO":         "co",          "SO2":        "so2",
    "O3":         "o3",          "Benzene":    "benzene",
    "Toluene":    "toluene",     "Xylene":     "xylene",
    "AQI":        "aqi",         "AQI_Bucket": "aqi_bucket",
})

kaggle_st = pd.read_csv(args.kaggle_st).rename(columns={
    "StationId": "station_id", "StationName": "station_name",
    "City": "city", "State": "state",
})[["station_id", "station_name", "city", "state"]]

df = df.merge(kaggle_st, on="station_id", how="left")

# Filter by city only in single mode
if args.mode == "single":
    if not args.city:
        raise ValueError("--city required when mode=single")
    df = df[df["city"].str.lower() == args.city.lower()]
    print(f"mode=single: filtered to city={args.city}")
else:
    # Keep only cities that exist in config/stations.csv
    config_st = pd.read_csv(args.config_st)
    valid_cities = config_st["city"].str.lower().unique()
    df = df[df["city"].str.lower().isin(valid_cities)]
    print(f"mode=all: kept {df['city'].nunique()} cities")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df["timestamp"] = df["timestamp"].dt.floor("h")
df = df[(df["timestamp"] >= args.start) & (df["timestamp"] <= args.end)]

if df.empty:
    raise ValueError(f"No data found after filtering.")

config_st = pd.read_csv(args.config_st)[["station_id", "latitude", "longitude"]]
df = df.merge(config_st, on="station_id", how="left")

num_cols = [c for c in ["pm25","pm10","no","no2","nox","nh3","co",
                         "so2","o3","benzene","toluene","xylene","aqi"]
            if c in df.columns]
grp_cols = [c for c in ["timestamp","station_id","city","state",
                         "station_name","latitude","longitude"]
            if c in df.columns]
df = (df[grp_cols + num_cols]
      .groupby(grp_cols, dropna=False, as_index=False)
      .mean(numeric_only=True))

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(args.output, index=False)
print(f"Saved {len(df)} rows -> {args.output}")
print("Cities   : " + str(df["city"].nunique()))
print("Stations : " + str(df["station_id"].nunique()))
print("Range    : " + str(df["timestamp"].min()) + " to " + str(df["timestamp"].max()))
