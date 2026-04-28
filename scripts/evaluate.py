import os
import sys
import json
import logging
import warnings
import yaml
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

FEATURES_PATH   = Path("data/processed/features.parquet")
MODELS_DIR      = Path("models")
METRICS_DIR     = Path("metrics")
PARAMS_PATH     = Path("params.yaml")
MLFLOW_TRACKING = "mlruns"

TRAIN_END = "2018-01-01"
VAL_END   = "2019-06-01"

NON_FEATURE_COLS = [
    "timestamp", "station_id", "city", "state",
    "station_name", "latitude", "longitude", "aqi"
]


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_test_data(horizon: int) -> tuple:
    target_cols  = [f"aqi_t{i}" for i in range(1, horizon + 1)]
    drop_cols    = NON_FEATURE_COLS + target_cols

    pf           = pq.ParquetFile(FEATURES_PATH)
    all_cols     = pf.schema_arrow.names
    feature_cols = [c for c in all_cols if c not in drop_cols]
    keep_cols    = ["timestamp"] + feature_cols + target_cols

    log.info("Loading test split in chunks...")
    chunks = []
    for batch in pf.iter_batches(batch_size=50_000, columns=keep_cols):
        chunk = batch.to_pandas()
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        chunk = chunk[chunk["timestamp"] >= VAL_END]
        if len(chunk) > 0:
            for col in feature_cols + target_cols:
                if col in chunk.columns and chunk[col].dtype == np.float64:
                    chunk[col] = chunk[col].astype(np.float32)
            chunks.append(chunk)
        del batch
        gc.collect()

    if not chunks:
        raise ValueError("Test set is empty — check VAL_END date.")

    df = pd.concat(chunks, ignore_index=True)
    log.info(f"Test rows: {len(df):,}")

    df = df.dropna(subset=target_cols)
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    X_test = df[feature_cols].values.astype(np.float32)
    y_test = df[target_cols].values.astype(np.float32)

    del df
    gc.collect()

    log.info(f"X_test shape: {X_test.shape}")
    return X_test, y_test, feature_cols, target_cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    horizon_metrics = {}
    for h in [1, 6, 12, 24]:
        idx = h - 1
        if idx < y_true.shape[1]:
            horizon_metrics[f"test_mae_t{h}"]  = round(float(
                mean_absolute_error(y_true[:, idx], y_pred[:, idx])), 4)
            horizon_metrics[f"test_rmse_t{h}"] = round(float(
                np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))), 4)
            horizon_metrics[f"test_r2_t{h}"]   = round(float(
                r2_score(y_true[:, idx], y_pred[:, idx])), 4)

    return {
        "test_mae":  round(mae, 4),
        "test_rmse": round(rmse, 4),
        "test_r2":   round(r2, 4),
        **horizon_metrics
    }


def main():
    all_params = load_params()
    training   = all_params.get("training", {})
    model_name = training.get("model", "lgbm").lower().strip()
    horizon    = int(training.get("forecast_horizon", 24))

    model_path = MODELS_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")

    log.info(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)

    X_test, y_test, feature_cols, target_cols = load_test_data(horizon)

    log.info("Running predictions on test set...")
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    log.info(f"\n{'─'*55}")
    log.info(f"  TEST → MAE:{metrics['test_mae']:.4f}  "
             f"RMSE:{metrics['test_rmse']:.4f}  "
             f"R²:{metrics['test_r2']:.4f}")
    log.info(f"{'─'*55}")
    log.info(f"  Per-horizon R²:")
    for h in [1, 6, 12, 24]:
        key = f"test_r2_t{h}"
        if key in metrics:
            log.info(f"    t+{h:>2}h → R²: {metrics[key]:.4f}")
    log.info(f"{'─'*55}\n")

    METRICS_DIR.mkdir(exist_ok=True)
    eval_metrics_path = METRICS_DIR / "eval_scores.json"
    with open(eval_metrics_path, "w") as f:
        json.dump({"model": model_name, **metrics}, f, indent=2)
    log.info(f"Eval metrics saved → {eval_metrics_path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment("AQI_Forecasting")

    # Reopen the same run that train.py created
    run_id_path = Path("metrics/active_run_id.txt")
    if not run_id_path.exists():
        raise FileNotFoundError("No active_run_id.txt found. Run train.py first.")

    with open(run_id_path) as f:
        run_id = f.read().strip()

    log.info(f"Resuming MLflow run: {run_id}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("test_size", len(X_test))
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_artifact(str(eval_metrics_path))

    log.info("Evaluation complete ✓")


if __name__ == "__main__":
    main()
