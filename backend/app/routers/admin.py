from fastapi import APIRouter
from app.services.ingestion_service import sync_all_sensors
from app.services.retention_service import prune_old_measurements

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/sync")
def sync():
    return {"results": sync_all_sensors()}

@router.post("/prune")
def prune(retention_days: int = 90):
    deleted = prune_old_measurements(retention_days=retention_days)
    return {"deleted": deleted}