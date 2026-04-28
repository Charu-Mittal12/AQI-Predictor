from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.routers.health import router as health_router
from app.routers.cities import router as cities_router
from app.routers.predict import router as predict_router
from app.routers.model_info import router as model_info_router
from app.routers.admin import router as adminrouter
from app.services.db_service import init_db, init_ingestion_tables
from app.services.city_service import get_supported_cities
from app.services.live_pipeline_service import collect_live_city_history
from app.services.model_service import get_model
from app.services.feature_service import get_feature_columns
from app.utils.config import (
    ENABLE_STARTUP_BACKFILL,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    FEATURE_COLUMNS_PATH,
    TRAINED_CITIES_PATH,
    CITY_ENCODER_PATH,
    CITY_COORDS_PATH,
)
from app.utils.logger import get_logger
from app.utils.metrics import MODEL_READY

log = get_logger(__name__)


def _validate_startup_artifacts() -> None:
    required_paths = [
        MODEL_PATH,
        MODEL_METADATA_PATH,
        FEATURE_COLUMNS_PATH,
        TRAINED_CITIES_PATH,
        CITY_ENCODER_PATH,
        CITY_COORDS_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required startup artifacts: " + ", ".join(missing))

    feature_cols = get_feature_columns()
    if not feature_cols:
        raise ValueError("feature_columns artifact is empty")

    log.info("Startup artifact validation passed. feature_count=%s", len(feature_cols))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    init_db()
    init_ingestion_tables()
    _validate_startup_artifacts()

    try:
        get_model()
        MODEL_READY.set(1)   # ← Prometheus: model is loaded and healthy
        log.info("Model loaded successfully. MODEL_READY=1")
    except Exception as exc:
        MODEL_READY.set(0)   # ← Prometheus: model failed to load
        log.error("Model preload failed: %s", exc)
        raise

    if ENABLE_STARTUP_BACKFILL:
        log.info("Backfilling historical data from OpenAQ...")
        for city in get_supported_cities():
            try:
                collect_live_city_history(city, force_refresh=True)
            except Exception as exc:
                log.warning("%s backfill failed: %s", city, exc)
    else:
        log.info("Startup backfill disabled. Serving predictions on demand.")

    yield
    # ── Shutdown ───────────────────────────────────────────────────────────────
    MODEL_READY.set(0)
    log.info("Application shutting down. MODEL_READY=0")


app = FastAPI(
    title="AQI Forecast API",
    version="1.0.0",
    description="FastAPI backend for AQI forecasting and advisory generation.",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

app.include_router(health_router,     tags=["health"])
app.include_router(cities_router,     tags=["cities"])
app.include_router(predict_router,    tags=["predict"])
app.include_router(model_info_router, tags=["model"])
app.include_router(adminrouter)