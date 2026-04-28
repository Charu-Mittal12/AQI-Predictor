from fastapi import APIRouter
from app.schemas.common_schema import HealthResponse, ReadyResponse
from app.services.model_service import is_model_ready

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse)
def ready():
    model_loaded = is_model_ready()
    return {
        "status": "ready" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }
