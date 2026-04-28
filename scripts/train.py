import gc
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
import yaml
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
PARAMS_PATH = Path("params.yaml")
MLFLOW_TRACKING = "mlruns"

TRAIN_END = pd.Timestamp("2018-01-01")
VAL_END = pd.Timestamp("2019-06-01")

NON_FEATURE_COLS = [
    "timestamp", "station_id", "city", "state",
    "station_name", "latitude", "longitude", "aqi"
]


def load_params() -> dict:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"params.yaml not found at {PARAMS_PATH}")
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_float_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def get_feature_cols(horizon: int, training_cfg: dict):
    pf = pq.ParquetFile(FEATURES_PATH)
    all_cols = pf.schema_arrow.names
    target_cols = [f"aqi_t{i}" for i in range(1, horizon + 1)]
    feature_cols = [c for c in all_cols if c not in (NON_FEATURE_COLS + target_cols)]

    max_features = training_cfg.get("max_features", None)
    if max_features is not None:
        max_features = int(max_features)
        if 0 < max_features < len(feature_cols):
            log.warning(f"Memory mode: using first {max_features} / {len(feature_cols)} features")
            feature_cols = feature_cols[:max_features]

    return feature_cols, target_cols


def load_single_target_data(feature_cols, target_col, training_cfg):
    sample_frac = float(training_cfg.get("sample_frac", 1.0))
    batch_size = int(training_cfg.get("batch_size_rows", 20000))
    max_train_rows = training_cfg.get("max_train_rows", None)
    max_val_rows = training_cfg.get("max_val_rows", None)
    max_test_rows = training_cfg.get("max_test_rows", None)

    keep_cols = ["timestamp"] + feature_cols + [target_col]
    pf = pq.ParquetFile(FEATURES_PATH)

    train_parts, val_parts, test_parts = [], [], []
    train_rows = val_rows = test_rows = 0
    total_seen = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=keep_cols):
        chunk = batch.to_pandas()
        total_seen += len(chunk)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])

        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=42)

        chunk = chunk.dropna(subset=[target_col])
        if chunk.empty:
            del batch, chunk
            gc.collect()
            continue

        chunk = downcast_df(chunk)

        tchunk = chunk[chunk["timestamp"] < TRAIN_END]
        vchunk = chunk[(chunk["timestamp"] >= TRAIN_END) & (chunk["timestamp"] < VAL_END)]
        schunk = chunk[chunk["timestamp"] >= VAL_END]

        if len(tchunk):
            if max_train_rows is not None:
                remaining = int(max_train_rows) - train_rows
                tchunk = tchunk.iloc[:max(0, remaining)]
            if len(tchunk):
                train_parts.append(tchunk)
                train_rows += len(tchunk)

        if len(vchunk):
            if max_val_rows is not None:
                remaining = int(max_val_rows) - val_rows
                vchunk = vchunk.iloc[:max(0, remaining)]
            if len(vchunk):
                val_parts.append(vchunk)
                val_rows += len(vchunk)

        if len(schunk):
            if max_test_rows is not None:
                remaining = int(max_test_rows) - test_rows
                schunk = schunk.iloc[:max(0, remaining)]
            if len(schunk):
                test_parts.append(schunk)
                test_rows += len(schunk)

        del batch, chunk, tchunk, vchunk, schunk
        gc.collect()

        train_done = (max_train_rows is not None and train_rows >= int(max_train_rows))
        val_done = (max_val_rows is not None and val_rows >= int(max_val_rows))
        test_done = (max_test_rows is None or test_rows >= int(max_test_rows))
        if train_done and val_done and test_done:
            break

    if not train_parts:
        raise ValueError(f"Training set empty for target {target_col}")
    if not val_parts:
        raise ValueError(f"Validation set empty for target {target_col}")

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=keep_cols)

    del train_parts, val_parts, test_parts
    gc.collect()

    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)
    val_df[feature_cols] = val_df[feature_cols].fillna(medians)
    if len(test_df):
        test_df[feature_cols] = test_df[feature_cols].fillna(medians)

    train_df[feature_cols] = downcast_df(train_df[feature_cols])
    val_df[feature_cols] = downcast_df(val_df[feature_cols])
    if len(test_df):
        test_df[feature_cols] = downcast_df(test_df[feature_cols])

    X_train = np.asarray(train_df[feature_cols], dtype=np.float32)
    y_train = np.asarray(train_df[target_col], dtype=np.float32)
    X_val = np.asarray(val_df[feature_cols], dtype=np.float32)
    y_val = np.asarray(val_df[target_col], dtype=np.float32)
    X_test = np.asarray(test_df[feature_cols], dtype=np.float32) if len(test_df) else np.empty((0, len(feature_cols)), dtype=np.float32)
    y_test = np.asarray(test_df[target_col], dtype=np.float32) if len(test_df) else np.empty((0,), dtype=np.float32)

    del train_df, val_df, test_df, medians
    gc.collect()

    log.info(f"{target_col}: seen={total_seen:,} train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} | X_train RAM={X_train.nbytes/1e6:.1f} MB")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_lgb_native(X_train, y_train, X_val, y_val, params: dict):
    p = {k: v for k, v in params.items() if k not in {"optuna", "forecast_horizon"}}
    p.setdefault("objective", "regression")
    p.setdefault("metric", "rmse")
    p.setdefault("verbosity", -1)
    p.setdefault("force_col_wise", True)
    p.setdefault("max_bin", 63)
    p.setdefault("num_leaves", 31)
    p.setdefault("max_depth", 8)
    p.setdefault("min_data_in_leaf", p.pop("min_child_samples", 100) if "min_child_samples" in p else 100)
    p.setdefault("feature_fraction", p.pop("colsample_bytree", 0.8) if "colsample_bytree" in p else 0.8)
    p.setdefault("bagging_fraction", p.pop("subsample", 0.8) if "subsample" in p else 0.8)
    p.setdefault("bagging_freq", p.pop("subsample_freq", 1) if "subsample_freq" in p else 1)
    p.setdefault("learning_rate", 0.05)
    num_boost_round = int(p.pop("n_estimators", 300))
    p.pop("n_jobs", None)
    p.setdefault("num_threads", 1)

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
    booster = lgb.train(
        params=p,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    del dtrain, dval
    gc.collect()
    return booster


def train_xgb(X_train, y_train, params: dict):
    p = {k: v for k, v in params.items() if k not in {"optuna", "forecast_horizon"}}
    p.setdefault("tree_method", "hist")
    p.setdefault("verbosity", 0)
    p.setdefault("n_jobs", 1)
    p.setdefault("learning_rate", 0.05)
    p.setdefault("n_estimators", 300)
    model = XGBRegressor(**p)
    model.fit(X_train, y_train)
    return model


def predict_model(model_name: str, model, X: np.ndarray) -> np.ndarray:
    if model_name == "lgbm":
        return model.predict(X).astype(np.float32)
    return model.predict(X).astype(np.float32)


def save_lgb_model(model, path: Path):
    model.save_model(str(path))


def save_xgb_model(model, path: Path):
    joblib.dump(model, path)


def run_optuna_for_first_target(model_name, base_params, feature_cols, target_cols, optuna_cfg, training_cfg):
    n_trials = int(optuna_cfg.get("n_trials", 10))
    seed = int(optuna_cfg.get("random_seed", 42))
    target_col = target_cols[int(optuna_cfg.get("target_idx", 0))]

    X_train, y_train, X_val, y_val, _, _ = load_single_target_data(feature_cols, target_col, training_cfg)

    def objective(trial):
        if model_name == "lgbm":
            trial_params = {
                "n_estimators": trial.suggest_int("n_estimators", int(optuna_cfg["n_estimators_min"]), int(optuna_cfg["n_estimators_max"])),
                "learning_rate": trial.suggest_float("learning_rate", float(optuna_cfg["learning_rate_min"]), float(optuna_cfg["learning_rate_max"]), log=True),
                "num_leaves": trial.suggest_int("num_leaves", int(optuna_cfg["num_leaves_min"]), int(optuna_cfg["num_leaves_max"])),
                "max_depth": trial.suggest_int("max_depth", int(optuna_cfg["max_depth_min"]), int(optuna_cfg["max_depth_max"])),
                "min_child_samples": trial.suggest_int("min_child_samples", int(optuna_cfg["min_child_samples_min"]), int(optuna_cfg["min_child_samples_max"])),
                "max_bin": int(optuna_cfg.get("max_bin", 63)),
            }
            params = {**base_params, **trial_params}
            model = train_lgb_native(X_train, y_train, X_val, y_val, params)
        else:
            trial_params = {
                "n_estimators": trial.suggest_int("n_estimators", int(optuna_cfg["n_estimators_min"]), int(optuna_cfg["n_estimators_max"])),
                "learning_rate": trial.suggest_float("learning_rate", float(optuna_cfg["learning_rate_min"]), float(optuna_cfg["learning_rate_max"]), log=True),
                "max_depth": trial.suggest_int("max_depth", int(optuna_cfg["max_depth_min"]), int(optuna_cfg["max_depth_max"])),
                "subsample": trial.suggest_float("subsample", float(optuna_cfg["subsample_min"]), float(optuna_cfg["subsample_max"])),
            }
            params = {**base_params, **trial_params}
            model = train_xgb(X_train, y_train, params)

        pred = predict_model(model_name, model, X_val)
        r2 = float(r2_score(y_val, pred))

        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metric("val_r2", r2)

        del model, pred
        gc.collect()
        return r2

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    log.info(f"Optuna on target {target_col} | trials={n_trials}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    METRICS_DIR.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(METRICS_DIR / "optuna_trials.csv", index=False)

    del X_train, y_train, X_val, y_val
    gc.collect()
    return study.best_params


def main():
    all_params = load_params()
    training = all_params.get("training", {})
    model_name = training.get("model", "lgbm").lower().strip()
    horizon = int(training.get("forecast_horizon", 24))
    model_params = all_params.get(model_name, {})

    if model_name not in {"lgbm", "xgboost"}:
        raise ValueError("This train.py supports only lgbm and xgboost.")
    if not model_params:
        raise ValueError(f"No params found for '{model_name}' in params.yaml")

    feature_cols, target_cols = get_feature_cols(horizon, training)

    mlflow.set_tracking_uri(MLFLOW_TRACKING)
    mlflow.set_experiment("AQI_Forecasting")
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("forecast_horizon", horizon)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_end", str(TRAIN_END.date()))
        mlflow.log_param("val_end", str(VAL_END.date()))

        for k in ["sample_frac", "batch_size_rows", "max_train_rows", "max_val_rows", "max_test_rows", "max_features"]:
            if k in training:
                mlflow.log_param(k, training[k])

        optuna_cfg = model_params.get("optuna", {})
        if optuna_cfg.get("enabled", False):
            best_params = run_optuna_for_first_target(model_name, model_params, feature_cols, target_cols, optuna_cfg, training)
            final_params = {**model_params, **best_params}
            all_params[model_name].update(best_params)
            all_params[model_name]["optuna"]["enabled"] = False
            with open(PARAMS_PATH, "w") as f:
                yaml.dump(all_params, f, default_flow_style=False, sort_keys=False)
            mlflow.log_artifact(str(METRICS_DIR / "optuna_trials.csv"))
        else:
            final_params = model_params

        for k, v in final_params.items():
            if k != "optuna":
                mlflow.log_param(k, v)

        model_dir = MODELS_DIR / f"{model_name}_models"
        model_dir.mkdir(exist_ok=True)

        per_target_rows = []
        val_abs_sum = 0.0
        val_sq_sum = 0.0
        val_count = 0
        val_sst_sum = 0.0
        val_sse_sum = 0.0

        test_abs_sum = 0.0
        test_sq_sum = 0.0
        test_count = 0
        test_sst_sum = 0.0
        test_sse_sum = 0.0
        has_test = False

        for i, target_col in enumerate(target_cols, start=1):
            log.info("=" * 70)
            log.info(f"Training {target_col} ({i}/{len(target_cols)})")

            X_train, y_train, X_val, y_val, X_test, y_test = load_single_target_data(feature_cols, target_col, training)

            if model_name == "lgbm":
                model = train_lgb_native(X_train, y_train, X_val, y_val, final_params)
            else:
                model = train_xgb(X_train, y_train, final_params)

            pred_val = predict_model(model_name, model, X_val)
            val_err = y_val - pred_val
            val_mae = float(np.mean(np.abs(val_err)))
            val_rmse = float(np.sqrt(np.mean(val_err ** 2)))
            val_r2 = float(r2_score(y_val, pred_val))

            val_abs_sum += float(np.abs(val_err).sum())
            val_sq_sum += float((val_err ** 2).sum())
            val_count += int(len(y_val))
            val_sse_sum += float((val_err ** 2).sum())
            val_sst_sum += float(((y_val - y_val.mean()) ** 2).sum())

            pred_test = None
            test_mae = test_rmse = test_r2 = None
            if len(X_test):
                has_test = True
                pred_test = predict_model(model_name, model, X_test)
                test_err = y_test - pred_test
                test_mae = float(np.mean(np.abs(test_err)))
                test_rmse = float(np.sqrt(np.mean(test_err ** 2)))
                test_r2 = float(r2_score(y_test, pred_test))

                test_abs_sum += float(np.abs(test_err).sum())
                test_sq_sum += float((test_err ** 2).sum())
                test_count += int(len(y_test))
                test_sse_sum += float((test_err ** 2).sum())
                test_sst_sum += float(((y_test - y_test.mean()) ** 2).sum())

            mlflow.log_metric(f"val_mae_{target_col}", val_mae)
            mlflow.log_metric(f"val_rmse_{target_col}", val_rmse)
            mlflow.log_metric(f"val_r2_{target_col}", val_r2)
            if test_mae is not None:
                mlflow.log_metric(f"test_mae_{target_col}", test_mae)
                mlflow.log_metric(f"test_rmse_{target_col}", test_rmse)
                mlflow.log_metric(f"test_r2_{target_col}", test_r2)

            model_path = model_dir / (f"{target_col}.txt" if model_name == "lgbm" else f"{target_col}.pkl")
            if model_name == "lgbm":
                save_lgb_model(model, model_path)
            else:
                save_xgb_model(model, model_path)

            per_target_rows.append({
                "target": target_col,
                "val_mae": round(val_mae, 4),
                "val_rmse": round(val_rmse, 4),
                "val_r2": round(val_r2, 4),
                "test_mae": None if test_mae is None else round(test_mae, 4),
                "test_rmse": None if test_rmse is None else round(test_rmse, 4),
                "test_r2": None if test_r2 is None else round(test_r2, 4),
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
            })

            pd.DataFrame(per_target_rows).to_csv(METRICS_DIR / "per_target_scores.csv", index=False)

            log.info(f"{target_col} | VAL -> MAE:{val_mae:.4f} RMSE:{val_rmse:.4f} R2:{val_r2:.4f}")
            if test_mae is not None:
                log.info(f"{target_col} | TEST -> MAE:{test_mae:.4f} RMSE:{test_rmse:.4f} R2:{test_r2:.4f}")

            del model, X_train, y_train, X_val, y_val, X_test, y_test, pred_val, pred_test
            gc.collect()

        metrics = {
            "model": model_name,
            "val_mae": round(val_abs_sum / max(val_count, 1), 4),
            "val_rmse": round(float(np.sqrt(val_sq_sum / max(val_count, 1))), 4),
            "val_r2": round(1.0 - (val_sse_sum / val_sst_sum if val_sst_sum > 0 else np.nan), 4),
        }

        per_target_df = pd.read_csv(METRICS_DIR / "per_target_scores.csv")
        for h in [1, 6, 12, 24]:
            tgt = f"aqi_t{h}"
            row = per_target_df[per_target_df["target"] == tgt]
            if not row.empty:
                metrics[f"val_mae_t{h}"] = round(float(row["val_mae"].iloc[0]), 4)
                metrics[f"val_rmse_t{h}"] = round(float(row["val_rmse"].iloc[0]), 4)

        if has_test and test_count > 0:
            metrics.update({
                "test_mae": round(test_abs_sum / test_count, 4),
                "test_rmse": round(float(np.sqrt(test_sq_sum / test_count)), 4),
                "test_r2": round(1.0 - (test_sse_sum / test_sst_sum if test_sst_sum > 0 else np.nan), 4),
            })
            for h in [1, 6, 12, 24]:
                tgt = f"aqi_t{h}"
                row = per_target_df[per_target_df["target"] == tgt]
                if not row.empty and pd.notna(row["test_mae"].iloc[0]):
                    metrics[f"test_mae_t{h}"] = round(float(row["test_mae"].iloc[0]), 4)
                    metrics[f"test_rmse_t{h}"] = round(float(row["test_rmse"].iloc[0]), 4)

        with open(METRICS_DIR / "scores.json", "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame([metrics]).to_csv(METRICS_DIR / "scores.csv", index=False)

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not pd.isna(v):
                mlflow.log_metric(k, v)

        mlflow.log_artifact(str(METRICS_DIR / "scores.json"))
        mlflow.log_artifact(str(METRICS_DIR / "scores.csv"))
        mlflow.log_artifact(str(METRICS_DIR / "per_target_scores.csv"))
        mlflow.log_artifact(str(PARAMS_PATH))

        run_id = mlflow.active_run().info.run_id
        with open(METRICS_DIR / "active_run_id.txt", "w") as f:
            f.write(run_id)

        log.info("=" * 70)
        log.info(f"VAL OVERALL -> MAE:{metrics['val_mae']:.4f} RMSE:{metrics['val_rmse']:.4f} R2:{metrics['val_r2']:.4f}")
        if "test_mae" in metrics:
            log.info(f"TEST OVERALL -> MAE:{metrics['test_mae']:.4f} RMSE:{metrics['test_rmse']:.4f} R2:{metrics['test_r2']:.4f}")
        log.info(f"MLflow Run ID : {run_id}")
        log.info("MLflow UI     : mlflow ui → http://localhost:5000")

        del per_target_rows, per_target_df
        gc.collect()

    log.info("Done")


if __name__ == "__main__":
    main()