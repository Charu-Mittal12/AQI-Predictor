import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path.resolve()
    return paths[0].resolve()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEPLOYMENT_BUNDLE_DIR = BACKEND_DIR / "deployment_bundle"


APP_NAME = "AQI Forecast API"
APP_VERSION = "1.0.0"
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")

MODEL_PATH = _env_path(
    "MODEL_PATH",
    _first_existing(
        DEPLOYMENT_BUNDLE_DIR / "model" / "model.pkl",
    ),
)
MODEL_METADATA_PATH = _env_path(
    "MODEL_METADATA_PATH",
    _first_existing(

        DEPLOYMENT_BUNDLE_DIR / "metadata" / "model_metadata.json",
    ),
)


CITY_ENCODER_PATH = _env_path(
    "CITY_ENCODER_PATH",
    DEPLOYMENT_BUNDLE_DIR / "preprocessing" / "city_label_encoder.pkl",
)
FEATURE_COLUMNS_PATH = _env_path(
    "FEATURE_COLUMNS_PATH",
    _first_existing(
        DEPLOYMENT_BUNDLE_DIR / "preprocessing" / "feature_columns.json",
        DEPLOYMENT_BUNDLE_DIR / "preprocessing" / "feature_columnbs.json",
    ),
)
TRAINED_CITIES_PATH = _env_path(
    "TRAINED_CITIES_PATH",
    DEPLOYMENT_BUNDLE_DIR / "preprocessing" / "trained_cities.json",
)
CITY_COORDS_PATH = _env_path(
    "CITY_COORDS_PATH",
    DEPLOYMENT_BUNDLE_DIR / "preprocessing" / "city_coords.json",
)
DB_PATH = _env_path("DB_PATH", PROJECT_ROOT / "database" / "live_aqi_buffer.db")

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")
OPENAQ_BASE_URL = os.getenv("OPENAQ_BASE_URL", "https://api.openaq.org/v3")
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = os.getenv("OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive")
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

MODEL_NAME = os.getenv("MODEL_NAME", MODEL_PATH.name)
ENABLE_STARTUP_BACKFILL = _env_flag("ENABLE_STARTUP_BACKFILL", default=False)
LIVE_HISTORY_HOURS = int(os.getenv("LIVE_HISTORY_HOURS", 200))
