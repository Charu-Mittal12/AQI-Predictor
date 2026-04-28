def make_advisory(aqi: int):
    if aqi <= 50:
        return "Air quality is good. Outdoor activity is safe."
    if aqi <= 100:
        return "Air quality is satisfactory. Sensitive people should stay alert."
    if aqi <= 200:
        return "Moderate pollution. Limit prolonged outdoor exertion."
    if aqi <= 300:
        return "Poor air quality. Consider reducing outdoor activities."
    if aqi <= 400:
        return "Very poor air quality. Avoid outdoor exposure if possible."
    return "Severe air quality. Stay indoors and use protection."