from app.services.db_service import cleanup_old_hourly_measurements
from app.utils.logger import get_logger

log = get_logger(__name__)

def prune_old_measurements(retention_days: int = 10):
    cleanup_old_hourly_measurements(days=retention_days)
    log.info("Pruned hourly_measurements older than %s days.", retention_days)
    return {"pruned": True, "retention_days": retention_days}