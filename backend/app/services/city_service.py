import json
from app.utils.config import TRAINED_CITIES_PATH

def get_supported_cities():
    with open(TRAINED_CITIES_PATH, "r") as f:
        return json.load(f)