"""
Airflow DAG: Full AQI Data Engineering Pipeline

Stages:
  setup_config -> prepare_pollution -> fetch_weather ->
  merge -> validate -> preprocess

Trigger manually from Airflow UI.
Schedule: None (manual trigger or CI trigger)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"
PYTHON  = "/opt/airflow/project/venv/bin/python"

default_args = {
    "owner":            "data-engineer",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="aqi_data_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "data-engineering", "stage-1"],
    doc_md="""
    ## AQI Data Engineering Pipeline

    Runs the full data collection and preprocessing pipeline in order:
    1. setup_config    - Generate config/stations.csv from Kaggle
    2. prepare_pollution - Clean and filter station_hour.csv
    3. fetch_weather   - Fetch historical weather from Open-Meteo
    4. merge           - Join pollution and weather datasets
    5. validate        - Run all 8 data quality checks
    6. preprocess      - Impute nulls, engineer features, create targets

    Trigger manually from the Airflow UI.
    """,
) as dag:

    setup_config = BashOperator(
        task_id="setup_config",
        bash_command=(
            f"{PYTHON} {PROJECT}/scripts/step0_setup_config.py "
            f"--kaggle-st {PROJECT}/data/raw/cpcb/stations.csv "
            f"--city Chennai "
            f"--out {PROJECT}/config/stations.csv"
        ),
    )

    prepare_pollution = BashOperator(
        task_id="prepare_pollution",
        bash_command=(
            f"{PYTHON} {PROJECT}/scripts/step2_prepare_pollution.py "
            f"--input      {PROJECT}/data/raw/cpcb/station_hour.csv "
            f"--kaggle-st  {PROJECT}/data/raw/cpcb/stations.csv "
            f"--config-st  {PROJECT}/config/stations.csv "
            f"--city       Chennai "
            f"--start      2019-01-01 "
            f"--end        2019-03-31 "
            f"--output     {PROJECT}/data/interim/pollution.parquet"
        ),
    )

    fetch_weather = BashOperator(
        task_id="fetch_weather",
        bash_command=(
            f"{PYTHON} {PROJECT}/scripts/step3_fetch_weather.py "
            f"--stations  {PROJECT}/config/stations.csv "
            f"--config    {PROJECT}/config/weather_vars.yaml "
            f"--start     2019-01-01 "
            f"--end       2019-03-31 "
            f"--raw-dir   {PROJECT}/data/raw/openmeteo "
            f"--output    {PROJECT}/data/interim/weather.parquet"
        ),
    )

    merge = BashOperator(
        task_id="merge",
        bash_command=(
            f"{PYTHON} {PROJECT}/scripts/step4_merge.py "
            f"--pollution {PROJECT}/data/interim/pollution.parquet "
            f"--weather   {PROJECT}/data/interim/weather.parquet "
            f"--output    {PROJECT}/data/processed/training_data.parquet"
        ),
    )

    validate = BashOperator(
        task_id="validate",
        bash_command=(
            f"{PYTHON} {PROJECT}/scripts/step5_validate.py "
            f"--input {PROJECT}/data/processed/training_data.parquet"
        ),
    )

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=(
            f"{PYTHON} {PROJECT}/scripts/step6_preprocess.py "
            f"--input          {PROJECT}/data/processed/training_data.parquet "
            f"--output         {PROJECT}/data/processed/features.parquet "
            f"--null-threshold 0.8"
        ),
    )

    # ── DAG order ──────────────────────────────────────────────
    (
        setup_config
        >> prepare_pollution
        >> fetch_weather
        >> merge
        >> validate
        >> preprocess
    )