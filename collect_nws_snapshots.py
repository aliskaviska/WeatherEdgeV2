import os

from datetime import datetime, timezone

from zoneinfo import ZoneInfo

import requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")

SUPABASE_SECRET = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

NWS_HEADERS = {

    "User-Agent": "WeatherEdge/1.0 weather forecast archive",

    "Accept": "application/geo+json",

}

LOCATIONS = {

    "New York": {

        "lat": 40.78335,

        "lon": -73.96497,

        "tz": "America/New_York",

        "station_id": "KNYC",

    },

    "Chicago": {

        "lat": 41.78417,

        "lon": -87.75528,

        "tz": "America/Chicago",

        "station_id": "KMDW",

    },

    "Miami": {

        "lat": 25.7952,

        "lon": -80.3254,

        "tz": "America/New_York",

        "station_id": "KMIA",

    },

    "Los Angeles": {

        "lat": 33.93806,

        "lon": -118.38889,

        "tz": "America/Los_Angeles",

        "station_id": "KLAX",

    },

    "Denver": {

        "lat": 39.85,

        "lon": -104.66,

        "tz": "America/Denver",

        "station_id": "KDEN",

    },

}

def get_json(url):

    response = requests.get(

        url,

        headers=NWS_HEADERS,

        timeout=30,

    )

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

    point_url = (

        f"https://api.weather.gov/points/"

        f"{location['lat']},{location['lon']}"

    )

    point_data = get_json(point_url)

    hourly_url = point_data["properties"]["forecastHourly"]

    hourly_data = get_json(hourly_url)

    return hourly_data["properties"]["periods"]

def make_rows(city, location):

    periods = get_hourly_forecast(location)

    tz = ZoneInfo(location["tz"])

    snapshot_time = datetime.now(timezone.utc).isoformat()

    parsed = []

    for period in periods:

        temp_f = temperature_f(period)

        if temp_f is None:

            continue

        forecast_dt = datetime.fromisoformat(

            period["startTime"]

        ).astimezone(tz)

        parsed.append(

            {

                "forecast_dt": forecast_dt,

                "temperature_f": temp_f,

            }

        )

    # Calculate the highest NWS hourly forecast currently

    # available for each local calendar date.

    highs_by_date = {}

    for item in parsed:

        local_date = item["forecast_dt"].date()

        current = highs_by_date.get(local_date)

        if (

            current is None

            or item["temperature_f"] > current

        ):

            highs_by_date[local_date] = item["temperature_f"]

    rows = []

    for item in parsed:

        local_date = item["forecast_dt"].date()

        rows.append(

            {

                "city": city,

                "station_id": location["station_id"],

                "contract_date": local_date.isoformat(),

                "snapshot_time": snapshot_time,

                "forecast_time": item[

                    "forecast_dt"

                ].astimezone(timezone.utc).isoformat(),

                "temperature_f": round(

                    item["temperature_f"], 2

                ),

                "predicted_daily_high_f": round(

                    highs_by_date[local_date], 2

                ),

                "source": "NWS",

            }

        )

    return rows

def save_rows(rows):

    if not rows:

        return

    url = (

        f"{SUPABASE_URL}"

        f"/rest/v1/weather_forecast_snapshots"

    )

    headers = {

        "apikey": SUPABASE_SECRET,

        "Content-Type": "application/json",

        "Prefer": "return=minimal",

    }

    # Upload in chunks so requests stay reasonably small.

    chunk_size = 250

    for start in range(0, len(rows), chunk_size):

        chunk = rows[start:start + chunk_size]

        response = requests.post(

            url,

            headers=headers,

            json=chunk,

            timeout=30,

        )

        response.raise_for_status()

def main():

    total_rows = 0

    for city, location in LOCATIONS.items():

        print(f"Collecting {city}...")

        try:

            rows = make_rows(city, location)

            save_rows(rows)

            total_rows += len(rows)

            print(

                f"Saved {len(rows)} rows for {city}"

            )

        except Exception as exc:

            print(f"ERROR for {city}: {exc}")

    print(

        f"Finished. Saved {total_rows} total rows."

    )

if __name__ == "__main__":

    main()
