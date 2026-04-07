import argparse
import requests
import time
from pathlib import Path
import pandas as pd

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

def get_coordinates(city: str) -> tuple:
    r = requests.get(GEOCODE_URL,
                     params={"name": city, "count": 1, "language": "en"},
                     timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        print(f"  WARNING: Could not geocode {city}, skipping.")
        return None, None
    return results[0]["latitude"], results[0]["longitude"]

ap = argparse.ArgumentParser()
ap.add_argument("--kaggle-st", required=True)
ap.add_argument("--mode",      required=True, choices=["all", "single"])
ap.add_argument("--city",      default=None)
ap.add_argument("--out",       required=True)
args = ap.parse_args()

st = pd.read_csv(args.kaggle_st).rename(columns={
    "StationId":   "station_id",
    "City":        "city",
    "StationName": "station_name",
    "State":       "state",
})

if args.mode == "single":
    if not args.city:
        raise ValueError("--city is required when mode=single")
    cities = [args.city]
else:
    cities = sorted(st["city"].dropna().unique().tolist())
    print(f"mode=all: found {len(cities)} unique cities")

print(f"Cities to process: {cities}")

rows = []
for city in cities:
    city_stations = st[st["city"].str.lower() == city.lower()]
    if city_stations.empty:
        print(f"  No stations found for {city}, skipping.")
        continue
    print(f"Geocoding {city} ...")
    lat, lon = get_coordinates(city)
    if lat is None:
        continue
    print(f"  {city}: lat={lat}, lon={lon} | stations={len(city_stations)}")
    for _, row in city_stations.iterrows():
        rows.append({
            "station_id": row["station_id"],
            "city":       row["city"],
            "latitude":   lat,
            "longitude":  lon,
        })
    time.sleep(0.5)

out_df = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(args.out, index=False)
print(f"\\nSaved {len(out_df)} stations across {out_df['city'].nunique()} cities -> {args.out}")