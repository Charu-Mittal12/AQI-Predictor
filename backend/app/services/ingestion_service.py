from datetime import datetime, timedelta, timezone
import pandas as pd
from app.services.db_service import (
    get_sensor_registry,
    get_latest_measurement_time,
    insert_hourly_measurements,
    update_last_fetched,
)
from app.services.openaq_service import fetch_sensor_hours, normalize_hour_row
from app.utils.logger import get_logger

log = get_logger(__name__)

def _to_hour_start(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

def _to_iso(dt: datetime) -> str:
    return _to_hour_start(dt).isoformat().replace("+00:00", "Z")

def sync_one_sensor(sensor_meta: dict, safety_lag_hours: int = 2, backfill_days: int = 30):
    try:
        now = datetime.now(timezone.utc)
        safe_end = _to_hour_start(now - timedelta(hours=safety_lag_hours))

        latest_local = get_latest_measurement_time(sensor_meta["sensor_id"])
        if latest_local:
            start_dt = pd.to_datetime(latest_local, utc=True).to_pydatetime() + timedelta(hours=1)
        else:
            start_dt = safe_end - timedelta(days=backfill_days)

        if start_dt > safe_end:
            return {
                "sensor_id": sensor_meta["sensor_id"],
                "status": "up_to_date",
                "inserted": 0,
                "from": _to_iso(start_dt),
                "to": _to_iso(safe_end),
            }

        raw_rows = fetch_sensor_hours(
            sensor_meta["sensor_id"],
            _to_iso(start_dt),
            _to_iso(safe_end),
        )

        normalized_rows = []
        for row in raw_rows:
            norm = normalize_hour_row(row, sensor_meta)
            if norm["measurement_time"] is None:
                continue
            normalized_rows.append(norm)

        inserted = insert_hourly_measurements(normalized_rows)

        if normalized_rows:
            max_ts = max(r["measurement_time"] for r in normalized_rows if r["measurement_time"] is not None)
            update_last_fetched(sensor_meta["sensor_id"], max_ts)

        return {
            "sensor_id": sensor_meta["sensor_id"],
            "status": "synced",
            "inserted": inserted,
            "from": _to_iso(start_dt),
            "to": _to_iso(safe_end),
        }

    except Exception as exc:
        log.warning("Failed syncing sensor %s: %s", sensor_meta.get("sensor_id"), exc)
        return {
            "sensor_id": sensor_meta.get("sensor_id"),
            "status": "failed",
            "error": str(exc),
        }

def sync_all_sensors():
    sensors = get_sensor_registry()
    results = []

    for sensor_meta in sensors:
        results.append(sync_one_sensor(sensor_meta))

    return results
