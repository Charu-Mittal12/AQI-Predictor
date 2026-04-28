from __future__ import annotations

from datetime import datetime, timedelta
from airflow import DAG
import sys

sys.path.insert(0, "/opt/airflow/backend")

from airflow.operators.python import PythonOperator

from app.services.bootstrap_service import bootstrap_sensor_registry
from app.services.ingestion_service import sync_all_sensors
from app.services.retention_service import prune_old_measurements
from app.services.drift_detection import task_detect_drift
from app.services.db_service import cleanup_old_hourly_measurements, cleanup_old_readings
from app.utils.logger import get_logger

log = get_logger(__name__)

default_args = {
    'owner': 'aqi',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['mittalcharu.59@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}


def bootstrap_task():
    try:
        res = bootstrap_sensor_registry()
        log.info("Bootstrap completed: %s", res)
        return res
    except Exception as exc:
        log.exception("Bootstrap failed: %s", exc)
        raise


def ingest_task():
    try:
        res = sync_all_sensors()
        log.info("Ingestion completed: %s", res)
        return res
    except Exception as exc:
        log.exception("Ingestion failed: %s", exc)
        raise


def cleanup_task():
    """
    Lightweight hourly cleanup.
    - hourly_measurements : keep last 10 days  (raw sensor data)
    - city_readings       : keep last 240h     (10 days — must be > drift baseline window)
    - predictions         : kept by cleanup_dag.py daily, not touched here
    """
    try:
        cleanup_old_hourly_measurements(days=10)
        cleanup_old_readings(hours=240)   # 10 days — covers drift baseline window (6 days)
        deleted = prune_old_measurements(retention_days=90)
        log.info("Cleanup completed, deleted=%s", deleted)
        return {"deleted": deleted}
    except Exception as exc:
        log.exception("Cleanup failed: %s", exc)
        raise


def drift_task():
    """
    Runs rolling window drift detection after every ingestion cycle.

    Strategy : Rolling window (Option B — memory-friendly)
      baseline_window : 4–10 days ago  (older live data = "expected")
      new_window      : last 3 days    (recent live data = "actual")

    Both windows from SQLite city_readings — same source, same era.
    Naturally low PSI with realistic variation on 2-3 features.

    Writes:
      - drift_results table in SQLite  (full history, one row per feature per run)
      - drift_baseline/drift_report.json (latest snapshot)
      - Prometheus Gauges: aqi_drift_psi, aqi_drift_detected, aqi_drift_features_count
    """
    try:
        res = task_detect_drift(window_days=7)
        if res.get("skipped"):
            log.warning("Drift check skipped: %s", res.get("reason"))
        else:
            log.info(
                "Drift check complete — drifted=%d/%d retrain=%s",
                res["summary"]["features_with_drift"],
                res["summary"]["total_features_checked"],
                res["summary"]["retrain_recommended"],
            )
        return res
    except Exception as exc:
        log.exception("Drift check failed: %s", exc)
        raise


with DAG(
    dag_id="aqi_ingestion_pipeline",
    default_args=default_args,
    description="OpenAQ hourly ingestion + drift detection for AQI app",
    schedule="0 * * * *",   # every hour
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["aqi", "ingestion", "openaq", "drift"],
    max_active_runs=1,
) as dag:

    bootstrap = PythonOperator(
        task_id="bootstrap_sensor_registry",
        python_callable=bootstrap_task,
    )

    ingest = PythonOperator(
        task_id="sync_all_sensors",
        python_callable=ingest_task,
    )

    cleanup = PythonOperator(
        task_id="cleanup_old_data",
        python_callable=cleanup_task,
    )

    drift = PythonOperator(
        task_id="check_data_drift",
        python_callable=drift_task,
    )

    # bootstrap → ingest fresh data → cleanup old data → check drift
    bootstrap >> ingest >> cleanup >> drift