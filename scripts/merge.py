import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pollution", required=True)
ap.add_argument("--weather",   required=True)
ap.add_argument("--output",    required=True)
args = ap.parse_args()

pollution = pd.read_parquet(args.pollution)
weather   = pd.read_parquet(args.weather)

pollution["timestamp"] = pd.to_datetime(pollution["timestamp"])
weather["timestamp"]   = pd.to_datetime(weather["timestamp"])

merged = pollution.merge(weather, on=["timestamp", "station_id", "city"], how="left")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(args.output, index=False)
print(f"Merged: {len(merged)} rows x {len(merged.columns)} cols -> {args.output}")
print(f"Columns: {merged.columns.tolist()}")
