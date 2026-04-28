from pydantic import BaseModel
from typing import List


class PollutantsResponse(BaseModel):
    PM2_5: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    O3: float


class PredictResponse(BaseModel):
    city: str
    station_id: str | None = None
    station_name: str | None = None
    location_id: int | None = None
    current_aqi: int
    primary_pollutant: str
    last_updated: str
    forecast_24h: List[int]
    pollutants: PollutantsResponse
    weekly_trend: List[int]
    advisory: str
