from datetime import datetime, timezone
from time import perf_counter

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas.predict_schema import PredictResponse
from app.services.advisory_service import make_advisory
from app.services.city_service import get_supported_cities
from app.services.db_service import get_recent_readings, get_latest_reading, store_prediction
from app.services.feature_service import build_live_payload
from app.services.live_pipeline_service import collect_live_city_history
from app.services.model_service import predict_next_24h
from app.services.openaq_service import get_location_latest_timestamp
from app.utils.logger import get_logger
from app.utils.metrics import (
    PREDICT_REQUESTS,
    PREDICT_LATENCY,
    LIVE_FETCH_REQUESTS,
    LIVE_FETCH_ROWS,
    LATEST_DB_AGE_HOURS,
)

router = APIRouter()
log = get_logger(__name__)

MIN_HISTORY_ROWS = 168   # model minimum — do NOT change to 200
BACKFILL_HOURS   = 200   # how far back we look/fetch


def _to_utc_ts(value) -> pd.Timestamp | None:
    """Safely parse a timestamp string into a UTC pd.Timestamp."""
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _do_backfill(city: str, location_id: int, force_refresh: bool) -> None:
    """Run collect_live_city_history and update Prometheus counters."""
    try:
        fetch_result = collect_live_city_history(
            city=city,
            hours_back=BACKFILL_HOURS,
            force_refresh=force_refresh,
            location_id=location_id,
        )
        LIVE_FETCH_REQUESTS.labels(city=city, status="success").inc()
        LIVE_FETCH_ROWS.labels(city=city).inc(fetch_result.get("rows_written", 0))
    except HTTPException:
        LIVE_FETCH_REQUESTS.labels(city=city, status="failed").inc()
        raise
    except ValueError as exc:
        LIVE_FETCH_REQUESTS.labels(city=city, status="failed").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LIVE_FETCH_REQUESTS.labels(city=city, status="failed").inc()
        raise HTTPException(
            status_code=503,
            detail=f"Live fetch failed for {city} location_id={location_id}: {exc}",
        ) from exc


@router.get("/predict", response_model=PredictResponse)
def predict(city: str, location_id: int = Query(...)):
    start  = perf_counter()
    status = "ok"

    try:
        supported = get_supported_cities()
        if city not in supported:
            raise HTTPException(status_code=400, detail=f"{city} not supported")

        # ── Step 1: fetch latest OpenAQ timestamp for this location ───────────
        openaq_latest_ts = None
        try:
            openaq_latest_ts = get_location_latest_timestamp(location_id)
        except Exception as exc:
            log.warning(
                "Could not fetch latest OpenAQ timestamp for location_id=%s: %s",
                location_id, exc,
            )

        # ── Step 2: check DB freshness for THIS location_id ──────────────────
        # FIX: was get_latest_reading(city) — city-wide, could return wrong station
        db_latest    = get_latest_reading(city=city, location_id=location_id)
        db_latest_ts = _to_utc_ts(db_latest.get("timestamp")) if db_latest else None

        # ── Step 3: count rows for THIS location_id ───────────────────────────
        # FIX: was get_recent_readings(city, hours=BACKFILL_HOURS) — city-wide count
        rows      = get_recent_readings(city=city, hours=BACKFILL_HOURS, location_id=location_id)
        row_count = len(rows)

        # ── Step 4: decide whether backfill is needed ─────────────────────────
        need_backfill = row_count < MIN_HISTORY_ROWS

        if openaq_latest_ts is not None and db_latest_ts is not None:
            diff_hours = (openaq_latest_ts - db_latest_ts).total_seconds() / 3600.0
            if diff_hours > 1.0:
                need_backfill = True
        elif db_latest_ts is None:
            need_backfill = True

        # ── Step 5: first backfill attempt (incremental) ──────────────────────
        if need_backfill:
            _do_backfill(city, location_id, force_refresh=False)

            # FIX: re-read for THIS location_id after backfill
            rows      = get_recent_readings(city=city, hours=BACKFILL_HOURS, location_id=location_id)
            row_count = len(rows)

            # ── Step 6: second backfill attempt (force refresh) ───────────────
            if row_count < MIN_HISTORY_ROWS:
                _do_backfill(city, location_id, force_refresh=True)

                # FIX: re-read for THIS location_id after force refresh
                rows      = get_recent_readings(city=city, hours=BACKFILL_HOURS, location_id=location_id)
                row_count = len(rows)

                if row_count < MIN_HISTORY_ROWS:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Not enough live rows for {city} location_id={location_id} "
                            f"after backfill: {row_count}/{MIN_HISTORY_ROWS}"
                        ),
                    )

        # ── Step 7: build feature payload and predict ─────────────────────────
        try:
            payload = build_live_payload(city=city, hours=BACKFILL_HOURS, location_id=location_id)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        result = predict_next_24h(city, payload["X"], payload["raw_data"])
        store_prediction(city, result)
        result["advisory"] = make_advisory(result["current_aqi"])
        return result

    except HTTPException:
        status = "error"
        raise
    except ValueError as exc:
        status = "error"
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        status = "error"
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        PREDICT_REQUESTS.labels(city=city, status=status).inc()
        PREDICT_LATENCY.labels(city=city).observe(perf_counter() - start)

        # ── Freshness metric — updated every request, even on failure ─────────
        # FIX: was get_latest_reading(city) — city-wide, now scoped to location_id
        try:
            latest = get_latest_reading(city=city, location_id=location_id)
            if latest and latest.get("timestamp"):
                ts = pd.to_datetime(latest["timestamp"], utc=True, errors="coerce")
                if not pd.isna(ts):
                    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
                    LATEST_DB_AGE_HOURS.labels(city=city).set(round(age_hours, 2))
        except Exception as age_exc:
            log.warning(
                "Could not update LATEST_DB_AGE_HOURS for %s location_id=%s: %s",
                city, location_id, age_exc,
            )