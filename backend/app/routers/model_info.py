from fastapi import APIRouter

from app.utils.config import APP_NAME, APP_VERSION, MODEL_NAME
from app.services.city_service import get_supported_cities
from app.services.model_service import get_model_metadata

router = APIRouter()

@router.get("/model-info")
def model_info():
    cities = get_supported_cities()
    metadata = get_model_metadata()
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "model_name": metadata.get("model_name", MODEL_NAME),
        "model_version": metadata.get("model_version"),
        "model_stage": metadata.get("model_stage"),
        "mlflow_run_id": metadata.get("mlflow_run_id"),
        "model_source": metadata.get("model_source"),
        "model_family": metadata.get("model_family", "unknown"),
        "git_tag": metadata.get("git_tag"),
        "git_commit": metadata.get("git_commit"),
        "primary_metric": metadata.get("primary_metric"),
        "primary_metric_value": metadata.get("primary_metric_value"),
        "forecast_horizon_hours": metadata.get("forecast_horizon", 24),
        "supported_cities_count": len(cities),
        "supported_cities": cities,
    }
