import gc
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

from pathlib import Path
from datetime import datetime

import optuna
import mlflow
import mlflow.sklearn

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Load Params ───────────────────────────────────────────────────────────────
def load_params() -> dict:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"params.yaml not found at {PARAMS_PATH}")
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)

# ── Load & Validate Data ──────────────────────────────────────────────────────
def load_data(horizon: int, training_cfg: dict) -> tuple:
    target_cols  = [f"aqi_t{i}" for i in range(1, horizon + 1)]
    drop_cols    = NON_FEATURE_COLS + target_cols
    sample_frac  = float(training_cfg.get("sample_frac", 1.0))

    log.info("Reading schema from parquet (zero RAM)...")
    pf           = pq.ParquetFile(FEATURES_PATH)
    all_cols     = pf.schema_arrow.names
    feature_cols = [c for c in all_cols if c not in drop_cols]
    keep_cols    = ["timestamp"] + feature_cols + target_cols
    log.info(f"Features: {len(feature_cols)} | Targets: {len(target_cols)}")

    log.info(f"Reading parquet in chunks (sample_frac={sample_frac})...")
    sampled_chunks = []
    total_rows = 0

    for batch in pf.iter_batches(batch_size=50_000, columns=keep_cols):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        for col in feature_cols + target_cols:
            if col in chunk.columns and chunk[col].dtype == np.float64:
                chunk[col] = chunk[col].astype(np.float32)

        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=42)

        sampled_chunks.append(chunk)
        del batch
        gc.collect()

    df = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()

    log.info(f"Loaded: {total_rows:,} total rows → sampled {len(df):,} rows")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df = df[df["timestamp"] < TRAIN_END].copy()
    val_df   = df[(df["timestamp"] >= TRAIN_END) & (df["timestamp"] < VAL_END)].copy()
    test_df  = df[df["timestamp"] >= VAL_END].copy()

    del df
    gc.collect()

    log.info(f"Train:{len(train_df):,} | Val:{len(val_df):,} | Test:{len(test_df):,}")

    if len(train_df) == 0: raise ValueError("Training set empty — check TRAIN_END date.")
    if len(val_df)   == 0: raise ValueError("Validation set empty — check VAL_END date.")

    train_df = train_df.dropna(subset=target_cols)
    val_df   = val_df.dropna(subset=target_cols)
    test_df  = test_df.dropna(subset=target_cols)

    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    val_df[feature_cols]   = val_df[feature_cols].fillna(medians)
    test_df[feature_cols]  = test_df[feature_cols].fillna(medians)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_cols].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df[target_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df[target_cols].values.astype(np.float32)

    del train_df, val_df, test_df
    gc.collect()

    log.info(f"X_train shape: {X_train.shape} | RAM: {X_train.nbytes/1e6:.1f} MB")

    return (X_train, y_train, X_val, y_val,
            X_test, y_test, feature_cols, target_cols)

# ── Compute Metrics ───────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    split: str, model_name: str) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    horizon_metrics = {}
    for h in [1, 6, 12, 24]:
        idx = h - 1
        if idx < y_true.shape[1]:
            horizon_metrics[f"{split}_mae_t{h}"]  = float(
                mean_absolute_error(y_true[:, idx], y_pred[:, idx]))
            horizon_metrics[f"{split}_rmse_t{h}"] = float(
                np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx])))

    return {
        f"{split}_mae":  round(mae, 4),
        f"{split}_rmse": round(rmse, 4),
        f"{split}_r2":   round(r2, 4),
        **{k: round(v, 4) for k, v in horizon_metrics.items()}
    }

# ── Model Factory ─────────────────────────────────────────────────────────────
def get_model(name: str, params: dict):
    # Strip keys that are not valid model constructor params
    sklearn_skip = {"forecast_horizon", "optuna"}

    if name == "lgbm":
        p = {k: v for k, v in params.items() if k not in sklearn_skip}
        return MultiOutputRegressor(
            lgb.LGBMRegressor(**p, verbose=-1, n_jobs=2), n_jobs=1
        )

    elif name == "xgboost":
        p = {k: v for k, v in params.items() if k not in sklearn_skip}
        return MultiOutputRegressor(
            XGBRegressor(**p, tree_method="hist", verbosity=0, n_jobs=2), n_jobs=1
        )

    elif name == "prophet":
        raise NotImplementedError(
            "Prophet requires single-target training. "
            "Implement ProphetWrapper in scripts/prophet_wrapper.py first."
        )

    elif name == "lstm":
        raise NotImplementedError("LSTM — Phase 2. Complete classical baselines first.")

    elif name == "transformer":
        raise NotImplementedError("Transformer — Phase 2. Complete classical baselines first.")

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Choose from: lgbm, xgboost, prophet, lstm, transformer"
        )

# ── Save Artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model, model_name: str, metrics: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    model_path   = MODELS_DIR / f"{model_name}_model.pkl"
    generic_path = MODELS_DIR / "trained_model.pkl"
    metrics_path = METRICS_DIR / "scores.json"

    joblib.dump(model, model_path)
    joblib.dump(model, generic_path)
    log.info(f"Model saved → {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {metrics_path}")

    return model_path, metrics_path

# Optuna Hyperparameter Search
def run_optuna(X_train, y_train, X_val, y_val, optuna_cfg: dict) -> dict:
    """
    Runs Optuna TPE search over the LGBM search space defined in params.yaml.
    Every trial is logged as a nested MLflow run for full visibility.
    Returns best_params dict (only model constructor keys).
    """
    n_trials = int(optuna_cfg.get("n_trials", 50))
    seed     = int(optuna_cfg.get("random_seed", 42))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                int(optuna_cfg["n_estimators_min"]),
                int(optuna_cfg["n_estimators_max"])
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                float(optuna_cfg["learning_rate_min"]),
                float(optuna_cfg["learning_rate_max"]),
                log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                int(optuna_cfg["num_leaves_min"]),
                int(optuna_cfg["num_leaves_max"])
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                int(optuna_cfg["max_depth_min"]),
                int(optuna_cfg["max_depth_max"])
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                int(optuna_cfg["min_child_samples_min"]),
                int(optuna_cfg["min_child_samples_max"])
            ),
        }

        model = MultiOutputRegressor(
            lgb.LGBMRegressor(**params, verbose=-1, n_jobs=2), n_jobs=1
        )
        model.fit(X_train, y_train)
        r2 = float(r2_score(y_val, model.predict(X_val)))

        # Each trial → nested MLflow run (visible in MLflow UI under parent run)
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_r2", r2)

        return r2

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    log.info(f"Starting Optuna search: {n_trials} trials | seed={seed}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save all trial details as CSV — DVC will track this file
    METRICS_DIR.mkdir(exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv("metrics/optuna_trials.csv", index=False)
    log.info(f"All {n_trials} trials saved → metrics/optuna_trials.csv")

    log.info(f"{'─'*50}")
    log.info(f"  Best val_r2 : {study.best_value:.4f}")
    log.info(f"  Best params : {study.best_params}")
    log.info(f"{'─'*50}")

    return study.best_params

# Main
def main():
    all_params   = load_params()
    training     = all_params.get("training", {})
    model_name   = training.get("model", "lgbm").lower().strip()
    horizon      = int(training.get("forecast_horizon", 24))
    model_params = all_params.get(model_name, {})

    if not model_params:
        raise ValueError(
            f"No params found for '{model_name}' in params.yaml. "
            f"Add a '{model_name}:' section."
        )

    log.info(f"{'='*60}")
    log.info(f"Model     : {model_name.upper()}")
    log.info(f"Horizon   : {horizon} hours")
    log.info(f"Params    : { {k:v for k,v in model_params.items() if k != 'optuna'} }")
    log.info(f"{'='*60}")

    (X_train, y_train, X_val, y_val,
     X_test, y_test, feature_cols, target_cols) = load_data(horizon, training)

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment("AQI_Forecasting")
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:

        #Log base params 
        mlflow.log_param("model",            model_name)
        mlflow.log_param("forecast_horizon", horizon)
        mlflow.log_param("train_size",       len(X_train))
        mlflow.log_param("val_size",         len(X_val))
        mlflow.log_param("test_size",        len(X_test))
        mlflow.log_param("n_features",       len(feature_cols))
        mlflow.log_param("train_end",        TRAIN_END)
        mlflow.log_param("val_end",          VAL_END)

        # Optuna block 
        optuna_cfg = model_params.get("optuna", {})

        if optuna_cfg.get("enabled", False):
            log.info("Optuna tuning ENABLED — running hyperparameter search...")
            mlflow.log_param("optuna_enabled",  True)
            mlflow.log_param("optuna_n_trials", optuna_cfg.get("n_trials", 50))
            mlflow.log_param("optuna_seed",     optuna_cfg.get("random_seed", 42))

            best_params = run_optuna(X_train, y_train, X_val, y_val, optuna_cfg)

            # Merge: best_params overrides the manual values in model_params
            final_params = {**model_params, **best_params}

            # Write best params back to params.yaml so DVC can track them,
            # and disable Optuna so next dvc repro uses the cache instead of re-running
            all_params[model_name].update(best_params)
            all_params[model_name]["optuna"]["enabled"] = False
            with open(PARAMS_PATH, "w") as f:
                yaml.dump(all_params, f, default_flow_style=False, sort_keys=False)
            log.info("Best params written back to params.yaml (optuna.enabled → False)")

            mlflow.log_artifact("metrics/optuna_trials.csv")

        else:
            log.info("Optuna DISABLED — using params.yaml values directly.")
            final_params = model_params

        # Log final model params to MLflow parent run (skip optuna block)
        for k, v in final_params.items():
            if k != "optuna":
                mlflow.log_param(k, v)

        # Train final model 
        log.info("Training final model...")
        model = get_model(model_name, final_params)
        model.fit(X_train, y_train)
        log.info("Training complete.")

        val_metrics = compute_metrics(y_val, model.predict(X_val), "val", model_name)
        all_metrics = {"model": model_name, **val_metrics}

        for k, v in all_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, f"{model_name}_model")
        mlflow.log_artifact(str(PARAMS_PATH))

        log.info(f"\n{'─'*50}")
        log.info(f"  VAL  → MAE:{val_metrics['val_mae']:.4f}  "
                 f"RMSE:{val_metrics['val_rmse']:.4f}  "
                 f"R2:{val_metrics['val_r2']:.4f}")
        log.info(f"{'─'*50}\n")

        # Save for DVC 
        model_path, metrics_path = save_artifacts(model, model_name, all_metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

        # Save run_id for evaluate.py to reuse
        run_id = mlflow.active_run().info.run_id
        Path("metrics").mkdir(exist_ok=True)
        with open("metrics/active_run_id.txt", "w") as f:
            f.write(run_id)

        log.info(f"MLflow Run ID : {run_id}")
        log.info(f"MLflow UI     : mlflow ui → http://localhost:5000")

    log.info("Done ")


if __name__ == "__main__":
    main()