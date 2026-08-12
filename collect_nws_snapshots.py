import os

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET = os.environ["SUPABASE_SECRET"]

NWS_HEADERS = {
    "User-Agent": "WeatherEdge/1.0 weather forecast snapshot collector",
    "Accept": "application/geo+json",
}

LOCATIONS = {
    "New York": {"lat": 40.78335, "lon": -73.96497, "tz": "America/New_York", "station_id": "KNYC"},
    "Chicago": {"lat": 41.78417, "lon": -87.75528, "tz": "America/Chicago", "station_id": "KMDW"},
    "Miami": {"lat": 25.7952, "lon": -80.3254, "tz": "America/New_York", "station_id": "KMIA"},
    "Los Angeles": {"lat": 33.93806, "lon": -118.38889, "tz": "America/Los_Angeles", "station_id": "KLAX"},
    "Denver": {"lat": 39.85, "lon": -104.66, "tz": "America/Denver", "station_id": "KDEN"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "tz": "America/New_York", "station_id": "KATL"},
    "Boston": {"lat": 42.3656, "lon": -71.0096, "tz": "America/New_York", "station_id": "KBOS"},
    "Minneapolis": {"lat": 44.8848, "lon": -93.2223, "tz": "America/Chicago", "station_id": "KMSP"},
    "New Orleans": {"lat": 29.9934, "lon": -90.2580, "tz": "America/Chicago", "station_id": "KMSY"},
    "Dallas": {"lat": 32.8998, "lon": -97.0403, "tz": "America/Chicago", "station_id": "KDFW"},
    "Houston": {"lat": 29.9902, "lon": -95.3368, "tz": "America/Chicago", "station_id": "KIAH"},
    "Oklahoma City": {"lat": 35.3931, "lon": -97.6007, "tz": "America/Chicago", "station_id": "KOKC"},
    "Seattle": {"lat": 47.4502, "lon": -122.3088, "tz": "America/Los_Angeles", "station_id": "KSEA"},
    "San Antonio": {"lat": 29.5337, "lon": -98.4698, "tz": "America/Chicago", "station_id": "KSAT"},
}

def get_json(url):
    response = requests.get(url, headers=NWS_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()

def temperature_f(period):
    value = period.get("temperature")
    if value is None:
        return None
    value = float(value)
    if period.get("temperatureUnit") == "C":
        value = value * 9 / 5 + 32
    return value

def get_hourly_forecast(location):
    point_url = f"https://api.weather.gov/points/{location['lat']},{location['lon']}"
    point_data = get_json(point_url)
    hourly_url = point_data["properties"]["forecastHourly"]
    hourly_data = get_json(hourly_url)
    return hourly_data["properties"]["periods"]

def make_rows(city, location):
    periods = get_hourly_forecast(location)
    tz = ZoneInfo(location["tz"])
    snapshot_time = datetime.now(timezone.utc)
    parsed = []

    for period in periods:
        temp_f = temperature_f(period)
        if temp_f is None:
            continue
        forecast_dt = datetime.fromisoformat(period["startTime"]).astimezone(tz)
        parsed.append({"forecast_dt": forecast_dt, "temperature_f": temp_f})

    highs_by_date = {}
    for item in parsed:
        local_date = item["forecast_dt"].date()
        current = highs_by_date.get(local_date)
        if current is None or item["temperature_f"] > current:
            highs_by_date[local_date] = item["temperature_f"]

    rows = []
    for item in parsed:
        local_date = item["forecast_dt"].date()
        rows.append({
            "city": city,
            "station_id": location["station_id"],
            "contract_date": local_date.isoformat(),
            "snapshot_time": snapshot_time.isoformat(),
            "forecast_time": item["forecast_dt"].astimezone(timezone.utc).isoformat(),
            "temperature_f": round(item["temperature_f"], 1),
            "predicted_daily_high_f": round(highs_by_date[local_date], 1),
            "source": "NWS",
        })
    return rows

def save_rows(rows):
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/weather_forecast_snapshots"
    headers = {
        "apikey": SUPABASE_SECRET,
        "Authorization": f"Bearer {SUPABASE_SECRET}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    chunk_size = 250
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        response = requests.post(url, headers=headers, json=chunk, timeout=30)
        response.raise_for_status()

def main():
    total_rows = 0
    for city, location in LOCATIONS.items():
        print(f"Collecting {city}...")
        try:
            rows = make_rows(city, location)
            save_rows(rows)
            total_rows += len(rows)
            print(f"Saved {len(rows)} rows for {city}")
        except Exception as exc:
            print(f"ERROR for {city}: {exc}")
    print(f"Finished. Saved {total_rows} total rows.")

if __name__ == "__main__":
    main()
