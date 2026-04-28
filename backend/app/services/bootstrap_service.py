from pathlib import Path
from app.services.db_service import init_ingestion_tables, upsert_sensor_registry
from app.utils.logger import get_logger
import json

log = get_logger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
STATION_REGISTRY_PATH = BACKEND_DIR / "deployment_bundle" / "preprocessing" / "station_registry.json"


def bootstrap_sensor_registry(station_registry_path: Path = STATION_REGISTRY_PATH):
    if not station_registry_path.exists():
        raise FileNotFoundError(f"station registry not found: {station_registry_path}")

    init_ingestion_tables()
    data = json.loads(station_registry_path.read_text(encoding="utf-8"))

    rows = []
    for station in data:
        sensors = station.get("available_sensors")
        if isinstance(sensors, str):
            try:
                sensors = json.loads(sensors)
            except Exception:
                sensors = []

        if not isinstance(sensors, list):
            continue

        for sensor in sensors:
            sensor_id = sensor.get("id")
            param = sensor.get("parameter") or {}

            if sensor_id is None:
                continue

            if isinstance(param, dict):
                parameter_name = param.get("name")
                units = param.get("units") or sensor.get("units")
            else:
                parameter_name = str(param) if param else None
                units = sensor.get("units")

            if not parameter_name:
                continue

            rows.append({
                "station_id": station.get("station_id"),
                "city": station.get("city"),
                "station_name": station.get("station_name"),
                "openaq_location_id": station.get("openaq_location_id"),
                "openaq_location_name": station.get("openaq_location_name"),
                "sensor_id": sensor_id,
                "parameter": parameter_name,
                "units": units,
                "last_fetched_hour": None,
            })

    upsert_sensor_registry(rows)
    log.info(f"Bootstrapped sensor_registry with {len(rows)} rows.")
    return {"sensor_rows": len(rows)}
