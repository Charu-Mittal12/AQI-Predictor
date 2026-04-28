from prometheus_client import Counter, Histogram, Gauge
from prometheus_client.registry import REGISTRY
from prometheus_client.core import GaugeMetricFamily
import sqlite3
from app.utils.config import DB_PATH

# ── Prediction metrics ─────────────────────────────────────────────────────────

PREDICT_REQUESTS = Counter(
    "aqi_predict_requests_total",
    "Total number of prediction requests",
    ["city", "status"],
)

PREDICT_LATENCY = Histogram(
    "aqi_predict_latency_seconds",
    "Prediction latency in seconds",
    ["city"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

LIVE_FETCH_REQUESTS = Counter(
    "aqi_live_fetch_requests_total",
    "Total live fetch attempts",
    ["city", "status"],
)

LIVE_FETCH_ROWS = Counter(
    "aqi_live_fetch_rows_total",
    "Total live rows written to the database",
    ["city"],
)

# ── System health metrics ──────────────────────────────────────────────────────
# LATEST_DB_AGE_HOURS : updated in predict router after every successful prediction
# MODEL_READY         : set to 1 in startup() after model loads successfully

LATEST_DB_AGE_HOURS = Gauge(
    "aqi_latest_db_age_hours",
    "Age of the latest city reading in hours",
    ["city"],
)

MODEL_READY = Gauge(
    "aqi_model_ready",
    "Whether the model is loaded and ready (1=ready, 0=not ready)",
)





class DriftCollector:
    """
    Reads the latest drift_results rows from SQLite and exposes them
    as Prometheus metrics on every /metrics scrape.
    This solves the cross-process problem: Airflow writes to DB,
    FastAPI reads from DB and serves to Prometheus.
    """

    def collect(self):
        psi_metric      = GaugeMetricFamily(
            "aqi_drift_psi",
            "Population Stability Index per feature",
            labels=["feature"],
        )
        detected_metric = GaugeMetricFamily(
            "aqi_drift_detected",
            "Drift detected per feature (1=drift, 0=stable)",
            labels=["feature"],
        )
        count_metric    = GaugeMetricFamily(
            "aqi_drift_features_count",
            "Total features currently drifted",
        )

        try:
            conn = sqlite3.connect(str(DB_PATH))
            # Get the single latest computed_at timestamp
            row = conn.execute(
                "SELECT MAX(computed_at) FROM drift_results"
            ).fetchone()
            latest_ts = row[0] if row else None

            if latest_ts:
                rows = conn.execute(
                    """
                    SELECT feature, psi, drift
                    FROM drift_results
                    WHERE computed_at = ?
                    """,
                    (latest_ts,),
                ).fetchall()

                drifted_count = 0
                for feature, psi, drift in rows:
                    psi_metric.add_metric([feature], psi or 0.0)
                    detected_metric.add_metric([feature], float(drift or 0))
                    if drift:
                        drifted_count += 1

                count_metric.add_metric([], float(drifted_count))
            else:
                count_metric.add_metric([], 0.0)

            conn.close()

        except Exception as e:
            # Never crash the scrape endpoint
            count_metric.add_metric([], 0.0)

        yield psi_metric
        yield detected_metric
        yield count_metric


# Register once at import time
REGISTRY.register(DriftCollector())