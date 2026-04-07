import argparse
import sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
args = ap.parse_args()

df  = pd.read_parquet(args.input)
ok  = True

def check(condition, msg):
    global ok
    if not condition:
        print(f"FAIL: {msg}")
        ok = False
    else:
        print(f"PASS: {msg}")

check(not df[["timestamp","station_id","city"]].isnull().any().any(),
      "No nulls in key columns")

check(df.duplicated(subset=["timestamp","station_id"]).sum() == 0,
      "No duplicate station-hours")

for col in ["pm25","no2","co","so2","o3","aqi"]:
    if col in df.columns:
        check((df[col].dropna() >= 0).all(), f"No negative values in {col}")

if not ok:
    sys.exit(1)

print("\nVALIDATION PASSED")
print(f"Rows     : {len(df)}")
print(f"Stations : {df['station_id'].nunique()}")
print(f"Range    : {df['timestamp'].min()} to {df['timestamp'].max()}")
print("\nNull % per column:")
null_pct = (df.isnull().mean() * 100).round(1)
print(null_pct[null_pct > 0].to_string())
