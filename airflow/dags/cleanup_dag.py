from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.insert(0, "/opt/airflow/backend")

default_args = {
    'owner': 'aqi',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['mittalcharu.59@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}


def run_cleanup():
    from app.services.db_service import (
        cleanup_old_hourly_measurements,
        cleanup_old_readings,
        cleanup_old_predictions,
    )
    cleanup_old_hourly_measurements(days=10)
    cleanup_old_readings(hours=240)
    cleanup_old_predictions(days=7)
    print("Cleanup complete: hourly=10d, city=240h, predictions=7d")

with DAG(
    dag_id="aqi_daily_cleanup",
    description="Delete stale measurements beyond retention window",
    schedule="0 2 * * *",  # 2AM UTC daily
    start_date=datetime(2026, 4, 1),
    catchup=False,
    default_args=default_args,
    tags=["aqi", "cleanup"],
) as dag:

    cleanup_task = PythonOperator(
        task_id="prune_old_data",
        python_callable=run_cleanup,
    )