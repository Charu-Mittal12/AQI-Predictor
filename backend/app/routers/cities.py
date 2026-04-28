from fastapi import APIRouter
from app.services.city_service import get_supported_cities
from app.schemas.common_schema import CitiesResponse

router = APIRouter()


@router.get("/cities", response_model=CitiesResponse)
def cities():
    return {"cities": get_supported_cities()}