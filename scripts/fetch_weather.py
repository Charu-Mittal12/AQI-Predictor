import argparse, json, time
from pathlib import Path
import pandas as pd
import requests
import yaml


URL = "https://archive-api.open-meteo.com/v1/archive"


ap = argparse.ArgumentParser()
ap.add_argument("--stations", required=True)
ap.add_argument("--config",   required=True)
ap.add_argument("--start",    required=True)
ap.add_argument("--end",      required=True)
ap.add_argument("--raw-dir",  required=True)
ap.add_argument("--output",   required=True)
args = ap.parse_args()


stations = pd.read_csv(args.stations)
cfg      = yaml.safe_load(open(args.config))
raw_dir  = Path(args.raw_dir)
raw_dir.mkdir(parents=True, exist_ok=True)


#Filter: only cities that have actual pollution data 
poll_path = Path(args.output).parent.parent / "interim" / "pollution.parquet"
if poll_path.exists():
    valid_cities = set(pd.read_parquet(poll_path)["city"].unique())
    before = len(stations)
    stations = stations[stations["city"].isin(valid_cities)].reset_index(drop=True)
    print(f"Filtered stations: {before} → {len(stations)} (kept cities with pollution data)")
else:
    print("Warning: pollution.parquet not found — fetching all cities")


# One API call per unique city 
cities = stations.drop_duplicates(subset=["city"])[["city", "latitude", "longitude"]]
print(f"Fetching weather for {len(cities)} cities ...")


def fetch_with_retry(params, max_retries=5):
    """Fetch Open-Meteo with exponential backoff on 429."""
    for attempt in range(max_retries):
        r = requests.get(URL, params=params, timeout=300)
        if r.status_code == 429:
            wait = 30 * (attempt + 1)   # 30s, 60s, 90s, 120s, 150s
            print(f"    429 rate limit — waiting {wait}s (retry {attempt+1}/{max_retries}) ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {max_retries} retries: {params}")


#  Load already-cached cities to allow resume after crash
city_weather = {}
for _, row in cities.iterrows():
    fname = f"{row['city'].lower().replace(' ', '_')}_{args.start}_{args.end}.json"
    cache = raw_dir / fname
    if cache.exists():
        print(f"  {row['city']} — loaded from cache ✓")
        payload = json.loads(cache.read_text())
        df = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["city"]      = row["city"]
        city_weather[row["city"]] = df
        continue

    params = {
        "latitude":   row["latitude"],
        "longitude":  row["longitude"],
        "start_date": args.start,
        "end_date":   args.end,
        "hourly":     ",".join(cfg["hourly"]),
        "timezone":   cfg.get("timezone", "Asia/Kolkata"),
    }
    print(f"  {row['city']} (lat={row['latitude']}, lon={row['longitude']}) ...")
    payload = fetch_with_retry(params)

    cache.write_text(json.dumps(payload))

    df = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["city"]      = row["city"]
    city_weather[row["city"]] = df
    print(f"    {len(df)} rows fetched")

    time.sleep(8)   # 8s gap between cities → ~7 requests/min (under 10/min limit)


# ── Assign city weather to every station in that city ────────────────────────
frames = []
for _, row in stations.iterrows():
    w = city_weather[row["city"]].copy()
    w["station_id"] = row["station_id"]
    frames.append(w)


weather = pd.concat(frames, ignore_index=True)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
weather.to_parquet(args.output, index=False)


print(f"\nSaved {len(weather)} rows -> {args.output}")
print(f"Cities   : {weather['city'].nunique()}")
print(f"Stations : {weather['station_id'].nunique()}")
