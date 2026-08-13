
import math
import os
import re
import time
import threading
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import altair as alt
import pandas as pd
import requests
import streamlit as st

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

# These are convenient starting coordinates for the observing-site area.
# Always verify the settlement source shown in the app before trading.
PRESETS = {
    "New York": {"series":"KXHIGHNY","lat":40.7812,"lon":-73.9665,"tz":"America/New_York","station":"Central Park / NYC settlement area","station_id":"KNYC","wfo":"OKX"},
    "Chicago": {"series":"KXHIGHCHI","lat":41.9742,"lon":-87.9073,"tz":"America/Chicago","station":"Chicago O'Hare area","station_id":"KORD","wfo":"LOT"},
    "Miami": {"series":"KXHIGHMIA","lat":25.7959,"lon":-80.2870,"tz":"America/New_York","station":"Miami International Airport area","station_id":"KMIA","wfo":"MFL"},
    "Los Angeles": {"series":"KXHIGHLAX","lat":33.9416,"lon":-118.4085,"tz":"America/Los_Angeles","station":"Los Angeles International Airport area","station_id":"KLAX","wfo":"LOX"},
    "Denver": {"series":"KXHIGHDEN","lat":39.8561,"lon":-104.6737,"tz":"America/Denver","station":"Denver International Airport area","station_id":"KDEN","wfo":"BOU"},
    "Atlanta": {"series":"KXHIGHTATL","lat":33.6407,"lon":-84.4277,"tz":"America/New_York","station":"Atlanta airport settlement area","station_id":"KATL","wfo":"FFC"},
    "Boston": {"series":"KXHIGHTBOS","lat":42.3656,"lon":-71.0096,"tz":"America/New_York","station":"Boston Logan settlement area","station_id":"KBOS","wfo":"BOX"},
    "Minneapolis": {"series":"KXHIGHTMIN","lat":44.8848,"lon":-93.2223,"tz":"America/Chicago","station":"Minneapolis/St Paul settlement area","station_id":"KMSP","wfo":"MPX"},
    "New Orleans": {"series":"KXHIGHTNOLA","lat":29.9934,"lon":-90.2580,"tz":"America/Chicago","station":"New Orleans airport settlement area","station_id":"KMSY","wfo":"LIX"},
    "Dallas": {"series":"KXHIGHTDAL","lat":32.8998,"lon":-97.0403,"tz":"America/Chicago","station":"Dallas/Fort Worth settlement area","station_id":"KDFW","wfo":"FWD"},
    "Houston": {"series":"KXHIGHTHOU","lat":29.9844,"lon":-95.3414,"tz":"America/Chicago","station":"Houston Intercontinental settlement area","station_id":"KIAH","wfo":"HGX"},
    "Oklahoma City": {"series":"KXHIGHTOKC","lat":35.3931,"lon":-97.6007,"tz":"America/Chicago","station":"Oklahoma City Will Rogers Airport","station_id":"KOKC","wfo":"OUN"},
    "Seattle": {"series":"KXHIGHTSEA","lat":47.4502,"lon":-122.3088,"tz":"America/Los_Angeles","station":"Seattle-Tacoma settlement area","station_id":"KSEA","wfo":"SEW"},
    "San Antonio": {"series":"KXHIGHTSATX","lat":29.5337,"lon":-98.4698,"tz":"America/Chicago","station":"San Antonio airport settlement area","station_id":"KSAT","wfo":"EWX"},
}

# Known Kalshi public-page slugs for the temperature series.
KALSHI_SERIES_SLUGS = {
    "KXHIGHNY": "highest-temperature-in-nyc",
    "KXHIGHCHI": "highest-temperature-in-chicago",
    "KXHIGHMIA": "highest-temperature-in-miami",
    "KXHIGHLAX": "highest-temperature-in-los-angeles",
    "KXHIGHDEN": "highest-temperature-in-denver",
    "KXHIGHTATL": "atlanta-max-temperature",
    "KXHIGHTBOS": "boston-maximum-daily-temperature",
    "KXHIGHTMIN": "minneapolis-daily-high-temperature",
    "KXHIGHTNOLA": "new-orleans-max-temp-daily",
    "KXHIGHTDAL": "dallas-daily-high-temperature",
    "KXHIGHTHOU": "highest-temperature-in-houston",
    "KXHIGHTOKC": "highest-temperature-in-oklahoma-city",
    "KXHIGHTSEA": "seattle-maximum-temperature-daily",
    "KXHIGHTSATX": "san-antonio-daily-maximum-temperature",
}


def nws_climate_url(cfg):
    """Public NWS observed/daily climate page for the city's settlement office."""
    wfo = cfg.get("wfo")
    return f"https://www.weather.gov/wrh/Climate?wfo={wfo.lower()}" if wfo else None


def kalshi_event_url(series_ticker, event_ticker):
    """Build the public Kalshi event page URL for a dated weather event."""
    slug = KALSHI_SERIES_SLUGS.get(series_ticker)
    if slug and event_ticker:
        return (
            f"https://kalshi.com/markets/"
            f"{series_ticker.lower()}/{slug}/{event_ticker.lower()}"
        )
    return "https://kalshi.com/"

HEADERS = {
    "User-Agent": "WeatherEdge/2.0 (personal research dashboard)",
    "Accept": "application/json",
}

_KALSHI_REQUEST_LOCK = threading.Lock()
_KALSHI_LAST_REQUEST_AT = [0.0]
_KALSHI_MIN_REQUEST_GAP_SECONDS = 0.22

def get_json(url, params=None, timeout=25):
    """
    HTTP GET with Kalshi-specific pacing and 429 backoff.

    NWS/Open-Meteo/Supabase requests remain concurrent. Kalshi requests are
    serialized and lightly paced so a parallel city refresh does not produce a
    burst that trips the public API rate limit.
    """
    is_kalshi = str(url).startswith(KALSHI_BASE)

    if not is_kalshi:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()

    backoffs = (0.8, 1.6, 3.2, 5.0)
    last_response = None

    for attempt in range(len(backoffs) + 1):
        with _KALSHI_REQUEST_LOCK:
            elapsed = time.monotonic() - _KALSHI_LAST_REQUEST_AT[0]
            wait_for = _KALSHI_MIN_REQUEST_GAP_SECONDS - elapsed
            if wait_for > 0:
                time.sleep(wait_for)

            r = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )
            _KALSHI_LAST_REQUEST_AT[0] = time.monotonic()

        last_response = r
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()

        retry_after = r.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after is not None else backoffs[min(attempt, len(backoffs)-1)]
        except (TypeError, ValueError):
            delay = backoffs[min(attempt, len(backoffs)-1)]

        # Add a small deterministic cushion so all city workers do not wake up
        # at exactly the same instant.
        time.sleep(max(0.5, delay) + 0.15)

    last_response.raise_for_status()
    return last_response.json()

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

@st.cache_resource
def _kalshi_last_good_store():
    return {"markets": {}, "series": {}, "events": {}}


@st.cache_data(ttl=180, show_spinner=False)
def get_kalshi_markets(series_ticker):
    """
    Fetch open markets with last-good fallback.

    A temporary 429 should not erase a city from the dashboard. If retries are
    exhausted, serve the most recent successful market payload held by this
    Streamlit worker.
    """
    store = _kalshi_last_good_store()
    out, cursor = [], None

    try:
        for _ in range(10):
            params = {
                "series_ticker": series_ticker,
                "status": "open",
                "limit": 1000,
                "mve_filter": "exclude",
            }
            if cursor:
                params["cursor"] = cursor
            data = get_json(f"{KALSHI_BASE}/markets", params=params)
            out.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break

        store["markets"][series_ticker] = {
            "rows": out,
            "saved_at": time.time(),
        }
        return out

    except requests.HTTPError as exc:
        if getattr(exc.response, "status_code", None) == 429:
            stale = store["markets"].get(series_ticker)
            if stale and stale.get("rows"):
                return stale["rows"]
        raise

@st.cache_data(ttl=3600)
def get_series_info(series_ticker):
    store = _kalshi_last_good_store()
    try:
        data = get_json(f"{KALSHI_BASE}/series/{series_ticker}")
        info = data.get("series", {})
        if info:
            store["series"][series_ticker] = info
        return info
    except requests.HTTPError as exc:
        if getattr(exc.response, "status_code", None) == 429:
            return store["series"].get(series_ticker, {})
        raise

@st.cache_data(ttl=300, show_spinner=False)
def get_event(event_ticker):
    if not event_ticker:
        return {}
    store = _kalshi_last_good_store()
    try:
        data = get_json(f"{KALSHI_BASE}/events/{event_ticker}")
        info = data.get("event", {})
        if info:
            store["events"][event_ticker] = info
        return info
    except requests.HTTPError as exc:
        if getattr(exc.response, "status_code", None) == 429:
            return store["events"].get(event_ticker, {})
        raise

@st.cache_data(ttl=86400, show_spinner=False)
def get_nws_forecast_url(lat, lon):
    """Cache the mostly-static NWS point -> forecast endpoint for a day."""
    point = get_json(f"https://api.weather.gov/points/{lat},{lon}")
    return point["properties"]["forecast"]


@st.cache_data(ttl=900, show_spinner=False)
def get_nws_daily(lat, lon):
    forecast_url = get_nws_forecast_url(lat, lon)
    forecast = get_json(forecast_url)
    rows = []
    for p in forecast["properties"]["periods"]:
        if p.get("isDaytime"):
            rows.append({
                "date": datetime.fromisoformat(p["startTime"]).date(),
                "nws_high_f": p.get("temperature"),
                "nws_detail": p.get("shortForecast"),
                "nws_forecast_url": forecast_url,
            })
    return rows


@st.cache_data(ttl=300)
def get_observed_high_so_far(station_id, tz_name, target_date):
    """Highest temperature observed so far on target_date at the settlement-area NWS station."""
    if not station_id:
        return None
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
    now_local = datetime.now(tz)
    if target_date > now_local.date():
        return None
    end_local = now_local if target_date == now_local.date() else datetime.combine(target_date, datetime.max.time(), tzinfo=tz)
    params = {
        "start": start_local.isoformat(),
        "end": end_local.isoformat(),
        "limit": 500,
    }
    data = get_json(f"https://api.weather.gov/stations/{station_id}/observations", params=params)
    vals = []
    for feature in data.get("features", []):
        value_c = (((feature.get("properties") or {}).get("temperature") or {}).get("value"))
        if value_c is not None:
            try:
                vals.append(float(value_c) * 9 / 5 + 32)
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None

def nws_observed_data_url(station_id):
    """Public NWS station page with the latest official observations."""
    if not station_id:
        return None
    return f"https://www.weather.gov/wrh/timeseries?site={station_id}"


@st.cache_data(ttl=900)
def get_gfs_ensemble_daily_highs(lat, lon, tz_name, forecast_days=8):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": "gfs_seamless",
        "temperature_unit": "fahrenheit",
        "timezone": tz_name,
        "forecast_days": forecast_days,
    }
    data = get_json("https://ensemble-api.open-meteo.com/v1/ensemble", params=params)
    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    member_keys = [k for k in hourly if k.startswith("temperature_2m")]
    if not member_keys:
        raise ValueError("No GFS ensemble members returned.")
    df = pd.DataFrame({"time": times})
    for k in member_keys:
        df[k] = pd.to_numeric(hourly[k], errors="coerce")
    df["date"] = df["time"].dt.date
    return df.groupby("date")[member_keys].max()


def infer_market_date(m, tz_name):
    """
    Resolve the canonical Kalshi contract date.

    Daily-temperature markets can settle after midnight, so close/expiration
    timestamps must NOT decide which weather day a contract belongs to. Prefer
    the date encoded in the market/event ticker, then explicit market wording,
    and use timestamps only as a final fallback.
    """
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    # Kalshi weather tickers commonly contain codes such as 26AUG11.
    for key in ("event_ticker", "ticker"):
        blob = str(m.get(key) or "").upper()
        match = re.search(
            r"(?:^|-)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(?:-|$)",
            blob,
        )
        if match:
            yy, mon, dd = match.groups()
            try:
                return datetime(2000 + int(yy), month_map[mon], int(dd)).date()
            except ValueError:
                pass

    # Explicit ISO date in title/subtitle/ticker metadata.
    blob = " ".join(
        str(m.get(k, "")) for k in
        ("title", "subtitle", "yes_sub_title", "ticker", "event_ticker")
    )
    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", blob)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(0), "%Y-%m-%d").date()
        except ValueError:
            pass

    # Human-readable dates such as "Aug 11, 2026".
    human_match = re.search(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2})\b",
        blob,
        flags=re.I,
    )
    if human_match:
        try:
            return datetime.strptime(
                f"{human_match.group(1)} {human_match.group(2)} {human_match.group(3)}",
                "%b %d %Y",
            ).date()
        except ValueError:
            pass

    # Final fallback only. These timestamps can refer to settlement rather than
    # the weather day, which is why they have the lowest priority.
    for key in ("occurrence_datetime", "close_time", "expected_expiration_time", "expiration_time"):
        raw = m.get(key)
        if raw:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                return dt.astimezone(ZoneInfo(tz_name)).date()
            except Exception:
                pass
    return None

def market_condition(m):
    """
    Parse the exact YES outcome wording first, so the forecast logic and the
    label shown to the user refer to the same contract.
    """
    exact = str(m.get("yes_sub_title") or m.get("subtitle") or "").strip()
    text = exact.lower().replace("º", "°")

    # 83° or above / 83 or above / 83°+
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or\s+above|\+|and\s+above)", text)
    if match:
        n = float(match.group(1))
        return "above", n, None, exact or f"{n:g}°F or above"

    # 77° or below
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or\s+below|and\s+below)", text)
    if match:
        n = float(match.group(1))
        return "below_equal", None, n, exact or f"{n:g}°F or below"

    # 78° to 79° / 78-79°
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:to|-|–|—)\s*(-?\d+(?:\.\d+)?)\s*°?",
        text,
    )
    if match:
        lo, hi = sorted((float(match.group(1)), float(match.group(2))))
        return "range", lo, hi, exact or f"{lo:g}–{hi:g}°F"

    # Fallback to API strikes only when exact wording cannot be parsed.
    floor = to_float(m.get("floor_strike"))
    cap = to_float(m.get("cap_strike"))
    strike = str(m.get("functional_strike") or "").lower()

    if floor is not None and cap is not None and cap >= floor:
        return "range", floor, cap, exact or f"{floor:g}–{cap:g}°F"
    if strike in ("greater", "above", "gt") and floor is not None:
        return "above", floor, None, exact or f"{floor:g}°F or above"
    if strike in ("less", "below", "lt") and cap is not None:
        return "below", None, cap, exact or f"below {cap:g}°F"
    if floor is not None:
        return "above", floor, None, exact or f"{floor:g}°F or above"
    if cap is not None:
        return "below", None, cap, exact or f"below {cap:g}°F"
    return None, None, None, exact or "unparsed"


def probability(values, kind, lo, hi):
    s = pd.Series(values).dropna().astype(float)
    if s.empty:
        return None, 0
    if kind == "range":
        hits = ((s >= lo) & (s <= hi)).sum()
    elif kind == "above":
        hits = (s >= lo).sum()
    elif kind == "below":
        hits = (s < hi).sum()
    elif kind == "below_equal":
        hits = (s <= hi).sum()
    else:
        return None, len(s)
    return hits / len(s), len(s)

def wilson_lower(phat, n, z=1.2816):
    # About an 80% one-sided lower bound.
    if phat is None or n <= 0:
        return None
    denom = 1 + z*z/n
    center = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)
    return max(0.0, (center - margin) / denom)



def settlement_rounding_risk(observed_high, kind, lo, hi):
    """
    Fixed settlement-source safety rule for Top Bets.

    Live NWS observations can be decimal values while a settlement source may
    effectively resolve the daily high to a neighboring whole-degree value.
    To avoid betting through that ambiguity, treat BOTH whole degrees that
    bracket the observed high as unsafe strike values.

    Example: observed high 102.2°F -> ambiguous settlement degrees {102, 103}.
    Any contract touching 102 or 103 is excluded from Top Bets, regardless of
    side. This is intentionally conservative and is not user-adjustable.
    """
    if observed_high is None or pd.isna(observed_high):
        return {
            "risk": False,
            "ambiguous_degrees": [],
            "nearest_distance_f": None,
        }

    t = float(observed_high)
    lower_degree = math.floor(t)
    upper_degree = math.ceil(t)

    # If the observation is exactly an integer, keep the current degree plus
    # the next degree as the conservative settlement ambiguity zone.
    if lower_degree == upper_degree:
        ambiguous = {lower_degree, lower_degree + 1}
    else:
        ambiguous = {lower_degree, upper_degree}

    strikes = []
    if kind == "range":
        if lo is not None:
            strikes.append(float(lo))
        if hi is not None:
            strikes.append(float(hi))
    elif kind == "above":
        if lo is not None:
            strikes.append(float(lo))
    elif kind in ("below", "below_equal"):
        if hi is not None:
            strikes.append(float(hi))

    if not strikes:
        return {
            "risk": False,
            "ambiguous_degrees": sorted(ambiguous),
            "nearest_distance_f": None,
        }

    risk = any(
        abs(strike - degree) < 1e-9
        for strike in strikes
        for degree in ambiguous
    )
    nearest = min(abs(t - strike) for strike in strikes)

    return {
        "risk": bool(risk),
        "ambiguous_degrees": sorted(ambiguous),
        "nearest_distance_f": float(nearest),
    }


def _two_piece_normal_cdf(x, mean, sigma_down, sigma_up):
    """
    CDF for a continuous two-piece normal distribution.

    The left and right sides may have different sigmas. This lets WeatherEdge
    represent a final-high distribution whose remaining upside is tighter than
    its pre-peak uncertainty without pretending uncertainty is symmetric.
    """
    mean = float(mean)
    sigma_down = max(0.08, float(sigma_down))
    sigma_up = max(0.08, float(sigma_up))
    x = float(x)

    left_mass = sigma_down / (sigma_down + sigma_up)
    right_mass = 1.0 - left_mass

    if x <= mean:
        z = (x - mean) / sigma_down
        return 2.0 * left_mass * normal_cdf(z, 0.0, 1.0)

    z = (x - mean) / sigma_up
    return left_mass + right_mass * (2.0 * normal_cdf(z, 0.0, 1.0) - 1.0)


def remaining_heating_distribution(
    nws_high,
    adjusted_center,
    observed_high,
    hours_to_peak,
    base_sigma,
    trajectory_live=None,
):
    """
    Build an asymmetric distribution for the eventual official daily high.

    Early in the day this stays close to the calibrated NWS prior. As the
    expected peak approaches, the observed high becomes a stronger anchor and
    the model explicitly estimates how much additional heating remains.

    Large recent forecast misses mainly widen uncertainty elsewhere in the
    calibration system; they do not get permission here to create implausible
    late-day upside.
    """
    raw = float(nws_high)
    center = float(adjusted_center)
    sigma = max(0.35, float(base_sigma))
    obs = None
    if observed_high is not None and not pd.isna(observed_high):
        obs = float(observed_high)

    try:
        htp = float(hours_to_peak)
    except Exception:
        htp = 8.0

    # Same weighting convention as the live trajectory model:
    # 0 well before peak, 1 at/after expected peak.
    peak_weight = _clamp01((5.0 - htp) / 5.0)

    gap = None
    trend = None
    if trajectory_live:
        gap = trajectory_live.get("current_gap_f")
        trend = trajectory_live.get("recent_obs_trend_f")

    # Before observations become informative, preserve the ordinary forecast.
    if obs is None:
        return {
            "center_f": center,
            "sigma_down_f": sigma,
            "sigma_up_f": sigma,
            "observed_floor_f": None,
            "expected_remaining_heating_f": None,
            "upside_p95_f": None,
            "peak_weight": peak_weight,
        }

    # How much heating the point forecast still requires from the already
    # observed daily high.
    forecast_remaining = max(0.0, center - obs)

    # The closer we are to peak, the less of speculative forecast upside we
    # retain. Early-day forecasts are essentially untouched.
    retention = 1.0 - 0.52 * peak_weight

    # Persistent below-forecast observations reduce remaining-heating expectation
    # further. Above-forecast observations are already captured by the hard floor.
    if gap is not None and not pd.isna(gap) and float(gap) < -0.75:
        retention *= max(0.35, 1.0 - 0.07 * min(5.0, abs(float(gap))))

    # Flat/falling observations near peak are strong evidence that little heating
    # remains. Do not use this early in the day.
    if peak_weight >= 0.60 and trend is not None and not pd.isna(trend):
        tr = float(trend)
        if tr <= 0.25:
            retention *= 0.45
        elif tr <= 0.75:
            retention *= 0.65
        elif tr <= 1.25:
            retention *= 0.82

    retention = max(0.0, min(1.0, retention))
    expected_remaining = forecast_remaining * retention

    # If the raw NWS high is already below the observed high, don't invent
    # additional warming just to preserve the adjusted center.
    if raw <= obs:
        expected_remaining = min(expected_remaining, max(0.0, raw + 0.5 - obs))

    final_center = max(obs, obs + expected_remaining)

    # Downside uncertainty still represents uncertainty about where the true
    # maximum sits relative to the center, but it is truncated at observed_high.
    sigma_down = sigma

    # Upside uncertainty contracts much faster near peak. This is the key
    # asymmetry: after much of the heating window is gone, a long hot tail should
    # not survive merely because morning forecast error was wide.
    upside_factor = 1.0 - 0.58 * peak_weight
    if gap is not None and not pd.isna(gap) and float(gap) < -1.0:
        upside_factor *= max(0.50, 1.0 - 0.05 * min(5.0, abs(float(gap))))
    if peak_weight >= 0.65 and trend is not None and not pd.isna(trend):
        if float(trend) <= 0.5:
            upside_factor *= 0.62

    sigma_up = max(0.22, sigma * max(0.28, upside_factor))

    # A diagnostic 95%-ish remaining-heating ceiling. We do not hard truncate
    # here except when the separate daily-high lock fires; keeping a small tail
    # avoids false certainty while still crushing unrealistic late-day upside.
    upside_p95 = max(
        obs,
        final_center + 1.645 * sigma_up,
    )

    return {
        "center_f": float(final_center),
        "sigma_down_f": float(sigma_down),
        "sigma_up_f": float(sigma_up),
        "observed_floor_f": float(obs),
        "expected_remaining_heating_f": float(expected_remaining),
        "upside_p95_f": float(upside_p95),
        "peak_weight": float(peak_weight),
    }


def weatheredge_yes_probability(
    nws_high,
    kind,
    lo,
    hi,
    adjusted_center,
    observed_high=None,
    hours_to_peak=None,
    base_sigma=1.5,
    trajectory_live=None,
):
    """
    Probability of a Kalshi YES outcome under WeatherEdge's asymmetric,
    observed-high-truncated final-temperature distribution.
    """
    params = remaining_heating_distribution(
        nws_high=nws_high,
        adjusted_center=adjusted_center,
        observed_high=observed_high,
        hours_to_peak=hours_to_peak,
        base_sigma=base_sigma,
        trajectory_live=trajectory_live,
    )

    mean = params["center_f"]
    sigma_down = params["sigma_down_f"]
    sigma_up = params["sigma_up_f"]
    floor = params["observed_floor_f"]

    def cdf(x):
        return _two_piece_normal_cdf(
            x, mean, sigma_down, sigma_up
        )

    floor_cdf = cdf(floor) if floor is not None else 0.0
    denom = max(1e-9, 1.0 - floor_cdf)

    # Deterministic floor checks. Once a temperature has already been observed,
    # contracts entirely below that value cannot win.
    if floor is not None:
        if kind == "range" and floor > float(hi) + 0.5:
            return 0.0, params
        if kind == "below" and floor >= float(hi) - 0.5:
            return 0.0, params
        if kind == "below_equal" and floor > float(hi) + 0.5:
            return 0.0, params

    if kind == "range":
        lower = float(lo) - 0.5
        upper = float(hi) + 0.5
        if floor is not None:
            lower = max(lower, floor)
        raw = max(0.0, cdf(upper) - cdf(lower))
    elif kind == "above":
        cutoff = float(lo) - 0.5
        if floor is not None:
            cutoff = max(cutoff, floor)
        raw = max(0.0, 1.0 - cdf(cutoff))
    elif kind == "below":
        cutoff = float(hi) - 0.5
        raw = max(0.0, cdf(cutoff) - floor_cdf) if floor is not None else cdf(cutoff)
    elif kind == "below_equal":
        cutoff = float(hi) + 0.5
        raw = max(0.0, cdf(cutoff) - floor_cdf) if floor is not None else cdf(cutoff)
    else:
        return None, params

    if floor is not None:
        raw /= denom

    return min(1.0, max(0.0, raw)), params


def point_forecast_supports_yes(temp, kind, lo, hi):
    if temp is None or pd.isna(temp):
        return None
    t = float(temp)
    if kind == "range":
        return lo <= t <= hi
    if kind == "above":
        return t >= lo
    if kind == "below":
        return t < hi
    if kind == "below_equal":
        return t <= hi
    return None

def side_supported_by_point(temp, side, kind, lo, hi):
    yes_support = point_forecast_supports_yes(temp, kind, lo, hi)
    if yes_support is None:
        return None
    return yes_support if side == "YES" else (not yes_support)


def infer_market_deadline(m, tz_name):
    """Best available market cutoff in the market's local timezone."""
    tz = ZoneInfo(tz_name)
    candidates = []
    for key in ("close_time", "expected_expiration_time", "expiration_time", "occurrence_datetime"):
        raw = m.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(tz)
            candidates.append(dt)
        except Exception:
            pass
    if not candidates:
        return None
    now = datetime.now(tz)
    future = [dt for dt in candidates if dt >= now]
    return min(future) if future else max(candidates)


def hours_to_deadline(m, tz_name):
    deadline = infer_market_deadline(m, tz_name)
    if deadline is None:
        return None
    now = datetime.now(ZoneInfo(tz_name))
    return max(0.0, (deadline - now).total_seconds() / 3600.0)



def _lead_bucket(hours_to_peak):
    """Lead-time buckets relative to the expected hottest point of the day."""
    if hours_to_peak is None or pd.isna(hours_to_peak):
        return "unknown"
    h = float(hours_to_peak)
    if h <= 0:
        return "after_peak"
    if h <= 2:
        return "0-2h"
    if h <= 4:
        return "2-4h"
    if h <= 8:
        return "4-8h"
    if h <= 16:
        return "8-16h"
    if h <= 30:
        return "16-30h"
    return "30h+"


def heuristic_nws_sigma_f(hours_to_peak):
    """Fallback NWS uncertainty until enough empirical peak-relative history exists."""
    bucket = _lead_bucket(hours_to_peak)
    return {
        "after_peak": 0.70,
        "0-2h": 0.80,
        "2-4h": 0.95,
        "4-8h": 1.20,
        "8-16h": 1.60,
        "16-30h": 2.20,
        "30h+": 2.90,
        "unknown": 2.50,
    }.get(bucket, 2.50)


def nws_sigma_f(hours_to_peak, calibration=None):
    """Use empirical NWS forecast error for the current peak-relative lead bucket."""
    bucket = _lead_bucket(hours_to_peak)
    if calibration and bucket in calibration:
        row = calibration[bucket]
        if row.get("n", 0) >= 8 and row.get("sigma_f") is not None:
            return max(0.55, float(row["sigma_f"]))
    return heuristic_nws_sigma_f(hours_to_peak)


def nws_bias_f(hours_to_peak, calibration=None):
    """
    Historical observed-minus-NWS bias for the current lead bucket.

    Use partial shrinkage for 8-14 samples so a short run of unusual weather
    does not over-correct the latest NWS point forecast.
    """
    bucket = _lead_bucket(hours_to_peak)
    if not calibration or bucket not in calibration:
        return 0.0
    row = calibration[bucket]
    n = int(row.get("n", 0) or 0)
    bias = row.get("bias_f")
    if n < 8 or bias is None:
        return 0.0
    weight = 1.0 if n >= 15 else 0.5
    return float(bias) * weight


def normal_cdf(x, mean, sigma):
    if sigma <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))




def nws_yes_probability(
    nws_high, kind, lo, hi, hours_left=None, observed_high=None,
    calibration=None, hours_to_peak=None, trajectory_adjustment_f=0.0,
    trajectory_sigma_multiplier=1.0, mean_override_f=None
):
    """
    Convert the latest NWS point high into a calibrated probability.

    Same-day markets can additionally condition the distribution on the live
    observed-vs-NWS trajectory. As the expected hottest point approaches, a
    persistent observed shortfall pulls the distribution center down and
    modestly tightens the remaining-upside uncertainty. The observed daily
    high remains a hard lower bound.
    """
    if nws_high is None or pd.isna(nws_high):
        return None

    calibration_lead = hours_to_peak if hours_to_peak is not None else hours_left
    mean = (
        float(mean_override_f)
        if mean_override_f is not None and not pd.isna(mean_override_f)
        else (
            float(nws_high)
            + nws_bias_f(calibration_lead, calibration)
            + float(trajectory_adjustment_f or 0.0)
        )
    )
    sigma = nws_sigma_f(calibration_lead, calibration)
    try:
        sigma *= max(0.55, min(1.35, float(trajectory_sigma_multiplier)))
    except Exception:
        pass
    sigma = max(0.45, sigma)

    floor = None
    if observed_high is not None and not pd.isna(observed_high):
        floor = float(observed_high)
        mean = max(mean, floor)

    def cdf(x):
        return normal_cdf(x, mean, sigma)

    # Condition on the exact highest temperature already observed. Once the
    # station has recorded this value, a lower final daily high is impossible.
    floor_cdf = cdf(floor) if floor is not None else 0.0
    denom = max(1e-9, 1.0 - floor_cdf)

    if kind == "range":
        lower = float(lo) - 0.5
        upper = float(hi) + 0.5
        raw = max(0.0, cdf(upper) - cdf(lower))
    elif kind == "above":
        raw = 1.0 - cdf(float(lo) - 0.5)
    elif kind == "below":
        raw = cdf(float(hi) - 0.5)
    elif kind == "below_equal":
        raw = cdf(float(hi) + 0.5)
    else:
        return None

    if floor is not None:
        if kind == "range":
            lower = max(float(lo) - 0.5, floor)
            upper = float(hi) + 0.5
            raw = max(0.0, cdf(upper) - cdf(lower))
        elif kind in ("below", "below_equal"):
            cutoff = (float(hi) - 0.5) if kind == "below" else (float(hi) + 0.5)
            raw = max(0.0, cdf(cutoff) - floor_cdf)
        else:
            raw = max(0.0, 1.0 - cdf(max(float(lo) - 0.5, floor)))
        raw /= denom

    return min(1.0, max(0.0, raw))


def market_yes_probability(m):
    """Approximate the market-implied YES probability from the live bid/ask."""
    bid = to_float(m.get("yes_bid_dollars"))
    ask = to_float(m.get("yes_ask_dollars"))
    if bid is not None and ask is not None and 0 <= bid <= ask <= 1:
        return (bid + ask) / 2
    if ask is not None and 0 < ask < 1:
        return ask
    last = to_float(m.get("last_price_dollars"))
    if last is not None and 0 < last < 1:
        return last
    return None


def market_implied_temperature(event_markets):
    """Approximate Kalshi's event-level implied temperature from contract prices.

    This is an internal approximation from bracket mid-prices, not Kalshi's
    proprietary displayed forecast number.
    """
    weighted = []
    for m in event_markets:
        kind, lo, hi, _ = market_condition(m)
        prob = market_yes_probability(m)
        if prob is None or kind is None:
            continue
        if kind == "range" and lo is not None and hi is not None:
            center = (float(lo) + float(hi)) / 2
        elif kind == "above" and lo is not None:
            center = float(lo) + 1.5
        elif kind in ("below", "below_equal") and hi is not None:
            center = float(hi) - 1.5
        else:
            continue
        weighted.append((center, prob))
    total = sum(w for _, w in weighted)
    if total <= 0:
        return None
    return sum(v * w for v, w in weighted) / total


def opportunity_score(edge, temp_gap, hours_left):
    """Ranking score: contract edge first, then forecast mismatch and time urgency."""
    if edge is None:
        return None
    mismatch_bonus = min(abs(temp_gap), 6.0) * 2.0 if temp_gap is not None else 0.0
    if hours_left is None:
        time_bonus = 0.0
    elif hours_left <= 3:
        time_bonus = 7.0
    elif hours_left <= 6:
        time_bonus = 6.0
    elif hours_left <= 12:
        time_bonus = 4.0
    elif hours_left <= 24:
        time_bonus = 2.0
    else:
        time_bonus = 0.0
    return edge * 100.0 + mismatch_bonus + time_bonus

def agreement_label(nws_support, median_support):
    if nws_support is True and median_support is True:
        return "✅ NWS + ensemble agree"
    if nws_support is False and median_support is False:
        return "❌ Both forecasts oppose"
    return "⚠️ Forecasts conflict"

def pretty_date(d):
    try:
        return d.strftime("%a, %b %-d")
    except Exception:
        return d.strftime("%a, %b %d").replace(" 0", " ")

def classify(edge, suspicious):
    if suspicious:
        return "⚠️ CHECK"
    if edge >= 0.15:
        return "🟢 STRONG"
    if edge >= 0.08:
        return "🟢 GOOD"
    if edge >= 0.04:
        return "🟡 MAYBE"
    return "⚪ PASS"

def source_names(series_info, event_info):
    sources = event_info.get("settlement_sources") or series_info.get("settlement_sources") or []
    names = [s.get("name") for s in sources if s.get("name")]
    return ", ".join(names) if names else "Check Kalshi contract rules"




@st.cache_data(ttl=1800, show_spinner=False)
@st.cache_resource(ttl=86400, show_spinner=False)
def build_nws_error_calibration(city, station_id, tz_name, cache_date=None):
    """
    Learn both ordinary NWS forecast error and how forecast revisions themselves
    relate to the eventual observed daily high.

    Historical evolution samples retain, for each stored forecast snapshot:
      - hours relative to the day's actual hottest point
      - current predicted high
      - 3h / 6h / 12h / 24h changes in the predicted high
      - recent revision volatility
      - final observed high minus the then-current prediction

    Live markets can compare today's revision pattern with these historical
    examples. Sparse histories are automatically shrunk toward zero adjustment.
    """
    cache_dir = Path("/tmp/weatheredge_persistent_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_city = re.sub(r"[^A-Za-z0-9_-]+", "_", str(city))
    cache_key_date = str(cache_date or datetime.now(ZoneInfo(tz_name)).date().isoformat())
    calibration_cache_path = cache_dir / f"calibration_v456_{safe_city}_{cache_key_date}.pkl"

    try:
        if calibration_cache_path.exists():
            age = time.time() - calibration_cache_path.stat().st_mtime
            # Historical calibration does not need minute-by-minute rebuilding.
            if age <= 24 * 3600:
                with calibration_cache_path.open("rb") as fh:
                    cached = pickle.load(fh)
                if isinstance(cached, dict):
                    return cached
    except Exception:
        pass

    snap, norm_err = get_normalized_snapshot_rows(city)
    err = norm_err
    if err or snap.empty or not station_id:
        return {}

    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    work = snap.copy()
    work["valid_local"] = work["valid_at"].dt.tz_convert(tz_name)
    work["snapshot_local"] = work["snapshot_at"].dt.tz_convert(tz_name)
    work["target_date"] = work["valid_local"].dt.date
    work = work[work["target_date"] < now_local.date()].copy()
    if work.empty:
        return {}

    recent_dates = sorted(work["target_date"].unique())[-60:]
    work = work[work["target_date"].isin(recent_dates)].copy()

    projected = (
        work.groupby(["snapshot_key", "target_date"], as_index=False)
        .agg(
            projected_high_f=("temp_f", "max"),
            snapshot_at=("snapshot_at", "min"),
            valid_min=("valid_local", "min"),
            valid_max=("valid_local", "max"),
        )
        .sort_values(["target_date", "snapshot_at"])
    )

    observed_stats = {}
    peak_minutes_by_date = {}
    for d in recent_dates:
        obs = get_station_observations(station_id, tz_name, d)
        if obs is None or obs.empty:
            continue

        x = obs.copy()
        x["time_local"] = pd.to_datetime(
            x["time"], utc=True, errors="coerce"
        ).dt.tz_convert(tz)
        x["temp_f"] = pd.to_numeric(x["temp_f"], errors="coerce")
        x = x.dropna(subset=["time_local", "temp_f"]).sort_values("time_local")
        if x.empty:
            continue

        high = float(x["temp_f"].max())
        peak_rows = x[x["temp_f"] >= high - 0.15].copy()
        minutes = (
            peak_rows["time_local"].dt.hour * 60
            + peak_rows["time_local"].dt.minute
            + peak_rows["time_local"].dt.second / 60.0
        )
        peak_minutes = float(minutes.median())
        peak_dt = (
            pd.Timestamp(datetime.combine(d, datetime.min.time(), tzinfo=tz))
            + pd.Timedelta(minutes=peak_minutes)
        )

        observed_stats[d] = {"high_f": high, "peak_dt": peak_dt}
        peak_minutes_by_date[d.isoformat()] = peak_minutes

    # Only retain snapshots that actually covered the day's observed peak
    # temperature window. This prevents late-day residual-hour maxima from
    # masquerading as full-day predicted highs in calibration and Best Bets.
    valid_projected_parts = []
    for d, grp in projected.groupby("target_date"):
        stat = observed_stats.get(d)
        if stat is None:
            continue
        peak_dt = stat["peak_dt"]
        g = grp[
            grp["snapshot_at"].apply(
                lambda x: pd.Timestamp(x).tz_convert(tz) <= peak_dt
            )
            & (grp["valid_min"] <= peak_dt)
            & (grp["valid_max"] >= peak_dt)
        ].copy()
        if not g.empty:
            valid_projected_parts.append(g)

    projected = (
        pd.concat(valid_projected_parts, ignore_index=True)
        if valid_projected_parts
        else projected.iloc[0:0].copy()
    )

    # Ordinary error calibration.
    samples = []
    for _, row in projected.iterrows():
        d = row["target_date"]
        stat = observed_stats.get(d)
        if stat is None:
            continue

        snapshot_local = pd.Timestamp(row["snapshot_at"]).tz_convert(tz)
        hours_to_peak = (
            stat["peak_dt"] - snapshot_local
        ).total_seconds() / 3600.0
        error = stat["high_f"] - float(row["projected_high_f"])
        samples.append({
            "bucket": _lead_bucket(hours_to_peak),
            "error_f": error,
            "hours_to_peak": hours_to_peak,
        })

    calibration = {}
    if samples:
        sdf = pd.DataFrame(samples)
        for bucket, grp in sdf.groupby("bucket"):
            errors = grp["error_f"].dropna()
            n = int(len(errors))
            if n == 0:
                continue
            calibration[bucket] = {
                "n": n,
                "sigma_f": float((errors.pow(2).mean()) ** 0.5),
                "bias_f": float(errors.mean()),
                "mae_f": float(errors.abs().mean()),
            }

    # Historical forecast-evolution samples.
    evolution_samples = []

    def value_near_prior_time(day_df, current_time, hours_back):
        target = pd.Timestamp(current_time) - pd.Timedelta(hours=hours_back)
        earlier = day_df[day_df["snapshot_at"] <= target]
        if earlier.empty:
            return None
        row = earlier.iloc[-1]
        age_h = (
            pd.Timestamp(current_time) - pd.Timestamp(row["snapshot_at"])
        ).total_seconds() / 3600.0
        # Do not use a wildly stale substitute for the requested lookback.
        if age_h > hours_back + 3.0:
            return None
        return float(row["projected_high_f"])

    for d, grp in projected.groupby("target_date"):
        stat = observed_stats.get(d)
        if stat is None:
            continue

        g = grp.sort_values("snapshot_at").reset_index(drop=True)
        for idx, row in g.iterrows():
            current_time = pd.Timestamp(row["snapshot_at"])
            current = float(row["projected_high_f"])
            snapshot_local = current_time.tz_convert(tz)
            hours_to_peak = (
                stat["peak_dt"] - snapshot_local
            ).total_seconds() / 3600.0

            p3 = value_near_prior_time(g, current_time, 3)
            p6 = value_near_prior_time(g, current_time, 6)
            p12 = value_near_prior_time(g, current_time, 12)
            p24 = value_near_prior_time(g, current_time, 24)

            recent = g[
                (g["snapshot_at"] <= current_time)
                & (g["snapshot_at"] >= current_time - pd.Timedelta(hours=12))
            ]["projected_high_f"].astype(float)
            volatility = float(recent.std(ddof=0)) if len(recent) >= 2 else 0.0

            evolution_samples.append({
                "target_date": d.isoformat(),
                "bucket": _lead_bucket(hours_to_peak),
                "hours_to_peak": float(hours_to_peak),
                "current_f": current,
                "change_3h_f": None if p3 is None else current - p3,
                "change_6h_f": None if p6 is None else current - p6,
                "change_12h_f": None if p12 is None else current - p12,
                "change_24h_f": None if p24 is None else current - p24,
                "volatility_12h_f": volatility,
                "final_error_f": float(stat["high_f"] - current),
            })

    peak_values = list(peak_minutes_by_date.values())
    typical_peak = (
        float(pd.Series(peak_values).median())
        if peak_values else 15 * 60.0
    )
    recent_daily_errors = []
    for d, grp in projected.groupby("target_date"):
        stat = observed_stats.get(d)
        if stat is None:
            continue
        g = grp.sort_values("snapshot_at").copy()
        eligible = g[
            g["snapshot_at"].apply(
                lambda x: pd.Timestamp(x).tz_convert(tz) <= stat["peak_dt"]
            )
        ]
        chosen = eligible.iloc[-1] if not eligible.empty else g.iloc[-1]
        projected_high = float(chosen["projected_high_f"])
        recent_daily_errors.append({
            "date": d.isoformat(),
            "projected_high_f": projected_high,
            "observed_high_f": float(stat["high_f"]),
            "error_f": float(stat["high_f"] - projected_high),
        })

    calibration["__meta__"] = {
        "typical_peak_minutes": typical_peak,
        "peak_minutes_by_date": peak_minutes_by_date,
        "n_peak_days": len(peak_values),
        "evolution_samples": evolution_samples,
        "n_evolution_samples": len(evolution_samples),
        "recent_daily_errors": sorted(recent_daily_errors, key=lambda x: x["date"]),
    }
    try:
        tmp_path = calibration_cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(calibration, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(calibration_cache_path)
    except Exception:
        pass

    return calibration


def _clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0





def recent_completed_day_adjustment(target_date, hours_to_peak, calibration):
    """
    Use recent completed-day forecast errors primarily to widen uncertainty,
    not to drag today's center.

    Philosophy:
      - one large previous-day miss is evidence that forecast error can be large;
      - repeated same-direction misses can earn a modest center correction;
      - magnitude of recent misses has a much stronger effect on sigma than mean.
    """
    meta = (calibration or {}).get("__meta__", {})
    recent = meta.get("recent_daily_errors", []) or []

    prior = []
    for item in recent:
        try:
            d = datetime.fromisoformat(str(item.get("date"))).date()
        except Exception:
            continue
        if d < target_date:
            prior.append(item)

    prior = prior[-5:]
    if not prior:
        return {
            "adjustment_f": 0.0,
            "sigma_multiplier": 1.0,
            "weighted_error_f": 0.0,
            "n_days": 0,
            "consistency": 0.0,
            "yesterday_error_f": None,
        }

    errors = [float(x.get("error_f", 0.0)) for x in prior]
    weights = list(range(1, len(errors) + 1))
    weighted = sum(e*w for e, w in zip(errors, weights)) / sum(weights)

    # Same-direction consistency: 1.0 means all misses share a sign.
    nonzero = [e for e in errors if abs(e) >= 0.25]
    if nonzero:
        pos = sum(e > 0 for e in nonzero)
        neg = sum(e < 0 for e in nonzero)
        consistency = max(pos, neg) / len(nonzero)
    else:
        consistency = 0.0

    yesterday_error = errors[-1] if errors else None

    # Center correction is intentionally conservative.
    # A single large miss should mostly widen the curve, not move the mean.
    base_center = weighted * 0.18
    if consistency >= 0.8 and len(nonzero) >= 3:
        base_center += weighted * 0.10
    elif consistency < 0.6:
        base_center *= 0.5

    adjustment = max(-0.85, min(0.85, base_center))

    # Uncertainty widens much more aggressively with recent misses.
    mae = sum(abs(e) for e in errors) / len(errors)
    yesterday_mag = abs(yesterday_error) if yesterday_error is not None else 0.0

    sigma_extra = 0.10 * mae + 0.06 * yesterday_mag
    if yesterday_mag >= 3.0:
        sigma_extra += 0.20
    if mae >= 2.0:
        sigma_extra += 0.15
    if consistency >= 0.8 and mae >= 1.5:
        sigma_extra += 0.08

    sigma_multiplier = max(1.0, min(1.85, 1.0 + sigma_extra))

    # Far from the expected peak, retain uncertainty. Near/after peak, live
    # observations should dominate and historical widening should fade.
    if hours_to_peak is not None:
        hp = float(hours_to_peak)
        if hp <= 0:
            sigma_multiplier = 1.0 + (sigma_multiplier - 1.0) * 0.25
            adjustment *= 0.25
        elif hp <= 2:
            sigma_multiplier = 1.0 + (sigma_multiplier - 1.0) * 0.55
            adjustment *= 0.55

    return {
        "adjustment_f": float(adjustment),
        "sigma_multiplier": float(sigma_multiplier),
        "weighted_error_f": float(weighted),
        "n_days": int(len(errors)),
        "consistency": float(consistency),
        "yesterday_error_f": None if yesterday_error is None else float(yesterday_error),
    }


def forecast_evolution_adjustment(
    city, cfg, contract_date, hours_to_peak, calibration, snapshot_df=None
):
    """
    Match today's NWS revision pattern to historical revision patterns that had
    known observed outcomes.

    Returns a data-driven adjustment to the current NWS high plus an uncertainty
    multiplier. The adjustment is intentionally conservative:
      - fewer than 6 comparable samples -> no adjustment
      - 6-11 samples -> heavily shrunk
      - 12+ samples -> progressively more trust, capped well below 100%
    """
    neutral = {
        "adjustment_f": 0.0,
        "sigma_multiplier": 1.0,
        "n_matches": 0,
        "raw_historical_error_f": None,
        "change_3h_f": None,
        "change_6h_f": None,
        "change_12h_f": None,
        "change_24h_f": None,
        "volatility_12h_f": None,
        "confidence": 0.0,
    }

    try:
        samples = (calibration or {}).get("__meta__", {}).get(
            "evolution_samples", []
        ) or []
        if not samples:
            return neutral

        if snapshot_df is None:
            snap, norm_err = get_recent_normalized_snapshot_rows(city)
        else:
            snap, norm_err = snapshot_df, None
        if norm_err or snap is None or snap.empty:
            return neutral

        tz = ZoneInfo(cfg["tz"])
        work = snap.copy()
        work["valid_local"] = work["valid_at"].dt.tz_convert(tz)
        work["snapshot_local"] = work["snapshot_at"].dt.tz_convert(tz)
        work["target_date"] = work["valid_local"].dt.date
        work = work[work["target_date"] == contract_date].copy()
        if work.empty:
            return neutral

        projected = (
            work.groupby("snapshot_key", as_index=False)
            .agg(
                projected_high_f=("temp_f", "max"),
                snapshot_at=("snapshot_at", "min"),
            )
            .sort_values("snapshot_at")
        )
        if projected.empty:
            return neutral

        latest = projected.iloc[-1]
        current_time = pd.Timestamp(latest["snapshot_at"])
        current = float(latest["projected_high_f"])

        def prior_value(hours_back):
            target = current_time - pd.Timedelta(hours=hours_back)
            earlier = projected[projected["snapshot_at"] <= target]
            if earlier.empty:
                return None
            row = earlier.iloc[-1]
            age_h = (
                current_time - pd.Timestamp(row["snapshot_at"])
            ).total_seconds() / 3600.0
            if age_h > hours_back + 3.0:
                return None
            return float(row["projected_high_f"])

        p3, p6, p12, p24 = (
            prior_value(3),
            prior_value(6),
            prior_value(12),
            prior_value(24),
        )

        recent = projected[
            projected["snapshot_at"] >= current_time - pd.Timedelta(hours=12)
        ]["projected_high_f"].astype(float)
        volatility = float(recent.std(ddof=0)) if len(recent) >= 2 else 0.0

        current_features = {
            "hours_to_peak": float(hours_to_peak)
            if hours_to_peak is not None else 6.0,
            "change_3h_f": None if p3 is None else current - p3,
            "change_6h_f": None if p6 is None else current - p6,
            "change_12h_f": None if p12 is None else current - p12,
            "change_24h_f": None if p24 is None else current - p24,
            "volatility_12h_f": volatility,
        }

        # Prefer the same lead-time bucket, then nearby lead-time examples.
        bucket = _lead_bucket(current_features["hours_to_peak"])
        candidate = [s for s in samples if s.get("bucket") == bucket]
        if len(candidate) < 8:
            candidate = samples

        # Distances are normalized by practical Fahrenheit/time scales rather than
        # fitting an unstable regression to a still-growing dataset.
        scales = {
            "hours_to_peak": 4.0,
            "change_3h_f": 1.5,
            "change_6h_f": 2.0,
            "change_12h_f": 2.5,
            "change_24h_f": 3.0,
            "volatility_12h_f": 1.0,
        }

        ranked = []
        for sample in candidate:
            parts = []
            for key, scale in scales.items():
                a = current_features.get(key)
                b = sample.get(key)
                if a is None or b is None:
                    continue
                try:
                    parts.append(((float(a) - float(b)) / scale) ** 2)
                except Exception:
                    continue

            # Require at least lead time plus two revision/volatility features.
            if len(parts) < 3:
                continue

            distance = math.sqrt(sum(parts) / len(parts))
            weight = math.exp(-0.5 * distance * distance)
            if weight <= 0.02:
                continue
            ranked.append((distance, weight, sample))

        ranked.sort(key=lambda x: x[0])
        ranked = ranked[:30]
        if len(ranked) < 6:
            return {**neutral, **current_features, "n_matches": len(ranked)}

        weights = [x[1] for x in ranked]
        errors = [float(x[2]["final_error_f"]) for x in ranked]
        wsum = sum(weights)
        raw_error = sum(w * e for w, e in zip(weights, errors)) / max(wsum, 1e-9)

        # Weighted residual spread around the matched historical error.
        variance = sum(
            w * (e - raw_error) ** 2
            for w, e in zip(weights, errors)
        ) / max(wsum, 1e-9)
        matched_sigma = math.sqrt(max(0.0, variance))

        n = len(ranked)
        confidence = min(0.75, max(0.0, (n - 5) / 25.0))
        # Additional shrinkage when the nearest matches are not especially close.
        mean_distance = sum(x[0] for x in ranked[:10]) / min(10, n)
        closeness = max(0.25, min(1.0, 1.35 - 0.35 * mean_distance))
        confidence *= closeness

        adjustment = max(-4.0, min(4.0, raw_error * confidence))

        # If similar histories had tight residuals, modestly tighten uncertainty.
        # If they were noisy, widen it slightly. Never let this dominate.
        sigma_multiplier = 1.0
        if matched_sigma < 1.2 and confidence >= 0.25:
            sigma_multiplier = max(0.82, 1.0 - 0.18 * confidence)
        elif matched_sigma > 2.5:
            sigma_multiplier = min(1.18, 1.0 + 0.12 * confidence)

        return {
            "adjustment_f": float(adjustment),
            "sigma_multiplier": float(sigma_multiplier),
            "n_matches": int(n),
            "raw_historical_error_f": float(raw_error),
            "change_3h_f": current_features["change_3h_f"],
            "change_6h_f": current_features["change_6h_f"],
            "change_12h_f": current_features["change_12h_f"],
            "change_24h_f": current_features["change_24h_f"],
            "volatility_12h_f": current_features["volatility_12h_f"],
            "confidence": float(confidence),
        }
    except Exception:
        return neutral


def live_trajectory_adjustment(
    city, cfg, contract_date, hours_to_peak,
    snapshot_df=None, observed_df=None,
):
    """
    Estimate how much the live observed trajectory should move today's final-high
    distribution away from the stored NWS trajectory.

    The adjustment deliberately becomes stronger near/after the expected daily
    peak. Early-day misses are mostly ignored because substantial heating time
    remains. Near the peak, a persistent shortfall plus weak recent warming is
    treated as evidence that the original NWS high is becoming less reachable.

    Returns a dict containing center adjustment, sigma multiplier, current
    observed-minus-forecast gap, and diagnostic fields.
    """
    neutral = {
        "adjustment_f": 0.0,
        "sigma_multiplier": 1.0,
        "current_gap_f": None,
        "peak_weight": 0.0,
        "recent_obs_trend_f": None,
        "matched_points": 0,
        "agreement_score": 0.5,
    }
    try:
        tz = ZoneInfo(cfg["tz"])
        now_local = pd.Timestamp.now(tz=tz)

        # Only condition same-day markets on live trajectory.
        if now_local.date() != contract_date:
            return neutral

        if snapshot_df is None:
            snap, norm_err = get_recent_normalized_snapshot_rows(city)
        else:
            snap, norm_err = snapshot_df, None
        if norm_err or snap is None or snap.empty:
            return neutral

        obs = (
            observed_df
            if observed_df is not None
            else get_station_observations(
                cfg.get("station_id"), cfg["tz"], contract_date
            )
        )
        if obs is None or obs.empty:
            return neutral

        latest_key = snap["snapshot_key"].max()
        fc = snap[snap["snapshot_key"] == latest_key].copy()
        fc["time_local"] = fc["valid_at"].dt.tz_convert(tz)
        fc = fc[
            (fc["time_local"].dt.date == contract_date)
            & (fc["time_local"] <= now_local)
        ][["time_local", "temp_f"]].dropna().sort_values("time_local")
        if fc.empty:
            return neutral

        ob = obs.copy()
        ob["time_local"] = pd.to_datetime(
            ob["time"], utc=True, errors="coerce"
        ).dt.tz_convert(tz)
        ob = ob[
            (ob["time_local"].dt.date == contract_date)
            & (ob["time_local"] <= now_local)
        ][["time_local", "temp_f"]].dropna().sort_values("time_local")
        if ob.empty:
            return neutral

        merged = pd.merge_asof(
            fc.sort_values("time_local"),
            ob.sort_values("time_local"),
            on="time_local",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=75),
            suffixes=("_fc", "_obs"),
        ).dropna(subset=["temp_f_fc", "temp_f_obs"])
        if merged.empty:
            return neutral

        # Compute the trajectory-agreement score here from the SAME matched frame.
        # V465 performed a second snapshot/observation merge just to get this score.
        if len(merged) >= 2:
            mae = float(
                (merged["temp_f_obs"] - merged["temp_f_fc"]).abs().mean()
            )
            mae_component = _clamp01(1.0 - (mae - 0.5) / 3.5)

            if len(merged) >= 3:
                fc_delta = float(
                    merged["temp_f_fc"].iloc[-1]
                    - merged["temp_f_fc"].iloc[-3]
                )
                obs_delta = float(
                    merged["temp_f_obs"].iloc[-1]
                    - merged["temp_f_obs"].iloc[-3]
                )
                if abs(fc_delta) < 0.4 and abs(obs_delta) < 0.4:
                    trend_component = 1.0
                elif fc_delta == 0 or obs_delta == 0:
                    trend_component = 0.7
                else:
                    trend_component = (
                        1.0
                        if (fc_delta > 0) == (obs_delta > 0)
                        else 0.25
                    )
            else:
                trend_component = 0.6

            agreement_score = _clamp01(
                0.75 * mae_component + 0.25 * trend_component
            )
        else:
            agreement_score = 0.5

        # Emphasize the most recent ~3 matched hours.
        recent = merged.tail(3)
        gaps = recent["temp_f_obs"] - recent["temp_f_fc"]
        current_gap = float(gaps.iloc[-1])
        weighted_gap = float(
            sum((i + 1) * float(v) for i, v in enumerate(gaps))
            / sum(range(1, len(gaps) + 1))
        )

        # 0 early in the day, ~0.5 two hours before peak, 1 at/after peak.
        htp = float(hours_to_peak) if hours_to_peak is not None else 6.0
        peak_weight = _clamp01((5.0 - htp) / 5.0)

        recent_obs_trend = None
        if len(ob) >= 2:
            tail = ob.tail(3)
            recent_obs_trend = float(
                tail["temp_f"].iloc[-1] - tail["temp_f"].iloc[0]
            )

        # Only chase meaningful trajectory errors. A small mismatch is noise.
        if abs(weighted_gap) < 0.75:
            effective_gap = 0.0
        else:
            effective_gap = weighted_gap

        # Near peak, use most of a persistent miss. Before peak, use much less.
        adjustment = effective_gap * (0.15 + 0.75 * peak_weight)

        # If we are near/after peak, observations are below forecast, and recent
        # warming has flattened, make the downward correction more decisive.
        if (
            peak_weight >= 0.65
            and effective_gap < -1.0
            and recent_obs_trend is not None
            and recent_obs_trend <= 1.0
        ):
            adjustment += 0.20 * effective_gap

        # Avoid a single bad sensor/forecast point producing absurd corrections.
        adjustment = max(-7.0, min(5.0, adjustment))

        # Once near peak with a confirmed shortfall, remaining upside uncertainty
        # should shrink rather than stay as wide as the morning distribution.
        sigma_multiplier = 1.0
        if peak_weight >= 0.5 and effective_gap < -1.0:
            sigma_multiplier = max(
                0.62,
                1.0 - 0.25 * peak_weight - 0.04 * min(5.0, abs(effective_gap)),
            )

        return {
            "adjustment_f": float(adjustment),
            "sigma_multiplier": float(sigma_multiplier),
            "current_gap_f": current_gap,
            "peak_weight": float(peak_weight),
            "recent_obs_trend_f": recent_obs_trend,
            "matched_points": int(len(merged)),
            "agreement_score": float(agreement_score),
        }
    except Exception:
        return neutral


def trajectory_agreement_score(city, cfg, contract_date, nws_high):
    """Compatibility wrapper; live scans use live_trajectory_adjustment directly."""
    try:
        result = live_trajectory_adjustment(
            city,
            cfg,
            contract_date,
            hours_to_peak=None,
        )
        return float(result.get("agreement_score", 0.5))
    except Exception:
        return 0.5


def bet_quality_score(edge, p_nws, hours_left, temp_gap, trajectory_score,
                      sigma_source=None, sigma_samples=0):
    """
    0-100 composite ranking score.

    Weights:
      25% market edge
      40% NWS confidence
      15% observed/NWS trajectory agreement
      10% time to settlement
       7% NWS-vs-Kalshi temperature mismatch
       3% secondary pricing mismatch

    Includes penalties for weak calibration and contradictory recent observations.
    """
    # Market edge: full credit at +30 pp, zero at <=0.
    edge_component = _clamp01((edge or 0.0) / 0.30)

    # NWS confidence: 55% starts earning credit; 90%+ is full credit.
    confidence_component = _clamp01(((p_nws or 0.0) - 0.55) / 0.35)

    # Observation / NWS trajectory agreement already 0..1.
    trajectory_component = _clamp01(trajectory_score)

    # Time: closer settlement is more informative, but not an automatic win.
    if hours_left is None:
        time_component = 0.35
    elif hours_left <= 3:
        time_component = 1.0
    elif hours_left <= 6:
        time_component = 0.9
    elif hours_left <= 12:
        time_component = 0.78
    elif hours_left <= 24:
        time_component = 0.62
    elif hours_left <= 48:
        time_component = 0.42
    elif hours_left <= 72:
        time_component = 0.28
    else:
        time_component = 0.15

    # Temperature mismatch: 0°F = none, 4°F+ = full credit.
    temp_component = _clamp01(abs(temp_gap or 0.0) / 4.0)

    # Small secondary pricing term to avoid double-counting edge too strongly.
    pricing_component = _clamp01((edge or 0.0) / 0.20)

    raw = 100.0 * (
        0.25 * edge_component
        + 0.40 * confidence_component
        + 0.15 * trajectory_component
        + 0.10 * time_component
        + 0.07 * temp_component
        + 0.03 * pricing_component
    )

    # Penalty if empirical NWS error calibration is not ready yet.
    if sigma_source != "historical":
        raw -= 5.0
    elif sigma_samples is not None and sigma_samples < 15:
        raw -= 2.0

    # Strong disagreement between observations and NWS trajectory.
    if trajectory_component < 0.30:
        raw -= 12.0
    elif trajectory_component < 0.50:
        raw -= 5.0

    return max(0.0, min(100.0, raw))


def bet_quality_label(score):
    if score >= 85:
        return "EXCELLENT"
    if score >= 75:
        return "STRONG"
    if score >= 65:
        return "GOOD"
    if score >= 55:
        return "WATCH"
    return "WEAK"



def expected_peak_context(calibration, target_date, tz_name):
    """
    Expected hottest time for a live target day.

    Anchor on the historical median local peak time, then move 35% of the way
    toward the previous day's actual peak. Cap the one-day adjustment at 90 min.
    """
    tz = ZoneInfo(tz_name)
    meta = (calibration or {}).get("__meta__", {})
    typical = float(meta.get("typical_peak_minutes", 15 * 60.0))
    peak_by_date = meta.get("peak_minutes_by_date", {}) or {}

    previous = peak_by_date.get((target_date - timedelta(days=1)).isoformat())
    expected = typical
    if previous is not None:
        adjustment = 0.35 * (float(previous) - typical)
        adjustment = max(-90.0, min(90.0, adjustment))
        expected += adjustment

    expected = max(10 * 60.0, min(20 * 60.0, expected))
    peak_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=tz) + timedelta(minutes=expected)
    now_local = datetime.now(tz)
    hours_to_peak = (peak_dt - now_local).total_seconds() / 3600.0
    return {
        "expected_peak_dt": peak_dt,
        "hours_to_peak": hours_to_peak,
        "typical_peak_minutes": typical,
        "previous_peak_minutes": previous,
        "n_peak_days": int(meta.get("n_peak_days", 0) or 0),
    }



def constrain_adjusted_high_center(
    raw_nws_high,
    proposed_center,
    observed_high,
    hours_to_peak,
    trajectory_live=None,
):
    """
    Keep the blended WeatherEdge center physically plausible for the current day.

    Historical signals may move the center, but near/after the expected daily
    peak they cannot imply implausibly large additional heating when today's
    actual observations and NWS forecast do not support it.
    """
    center = float(proposed_center)
    raw = float(raw_nws_high)
    obs = None
    if observed_high is not None and not pd.isna(observed_high):
        obs = float(observed_high)

    # Broad all-day guardrail: historical corrections can matter, but the center
    # should not wander arbitrarily far from today's live NWS anchor.
    center = max(raw - 4.0, min(raw + 4.0, center))

    if obs is not None:
        center = max(center, obs)

        htp = float(hours_to_peak) if hours_to_peak is not None else 8.0

        # As the expected peak approaches, limit how much unexplained warming can
        # remain above the higher of NWS and the observed high.
        live_anchor = max(raw, obs)
        if htp <= 0:
            upside_allowance = 0.75
        elif htp <= 1:
            upside_allowance = 1.25
        elif htp <= 2:
            upside_allowance = 1.75
        elif htp <= 4:
            upside_allowance = 2.75
        else:
            upside_allowance = 4.0

        # If observations are already running at/below the NWS trajectory and the
        # day is near peak, be stricter about speculative upside.
        gap = None
        trend = None
        if trajectory_live:
            gap = trajectory_live.get("current_gap_f")
            trend = trajectory_live.get("recent_obs_trend_f")
        if (
            htp <= 2
            and gap is not None
            and not pd.isna(gap)
            and float(gap) <= 0.75
        ):
            upside_allowance = min(upside_allowance, 1.75)
        if (
            htp <= 2
            and trend is not None
            and not pd.isna(trend)
            and float(trend) <= 1.0
        ):
            upside_allowance = min(upside_allowance, 1.25)

        if htp <= 0 and trend is not None and not pd.isna(trend):
            if float(trend) <= 0.25:
                upside_allowance = min(upside_allowance, 0.50)
            elif float(trend) <= 0.75:
                upside_allowance = min(upside_allowance, 0.75)
            else:
                upside_allowance = min(upside_allowance, 1.25)

        center = min(center, live_anchor + upside_allowance)

    return float(center)



def clearly_past_daily_peak(hours_to_peak, trajectory_live, observed_high):
    """
    Decide when the day's high should be treated as effectively locked.

    Once the expected hottest point is clearly behind us AND the recent observed
    trajectory is flat/falling, WeatherEdge stops assigning probability to a new
    higher daily temperature. A longer time past peak can lock even without a
    usable recent-trend estimate.
    """
    if observed_high is None or pd.isna(observed_high):
        return False

    try:
        htp = float(hours_to_peak)
    except Exception:
        return False

    trend = None
    if trajectory_live:
        trend = trajectory_live.get("recent_obs_trend_f")

    # Strongest case: at least an hour past peak and recent observations have
    # essentially stopped warming.
    if htp <= -1.0 and trend is not None and not pd.isna(trend):
        if float(trend) <= 0.50:
            return True

    # By two hours past the expected peak, treat the observed high as locked even
    # if a recent trend estimate is unavailable/noisy.
    if htp <= -2.0:
        return True

    return False


def build_city_rows(city, cfg):
    # Cheap/live inputs first. If there are no open markets, skip calibration.
    markets = get_kalshi_markets(cfg["series"])
    if not markets:
        return []

    nws = {r["date"]: r for r in get_nws_daily(cfg["lat"], cfg["lon"])}
    if not nws:
        return []

    calibration = build_nws_error_calibration(
        city,
        cfg.get("station_id"),
        cfg["tz"],
        cache_date=datetime.now(ZoneInfo(cfg["tz"])).date().isoformat(),
    )

    markets_by_event = {}
    for market in markets:
        markets_by_event.setdefault(market.get("event_ticker"), []).append(market)
    implied_by_event = {
        event_ticker: market_implied_temperature(group)
        for event_ticker, group in markets_by_event.items()
    }

    rows = []
    live_context_cache = {}
    observed_context_cache = {}

    # One compact normalized snapshot dataframe per city scan. All same-day
    # trajectory/revision calculations share it.
    recent_snap, recent_snap_err = get_recent_normalized_snapshot_rows(city)
    if recent_snap_err:
        recent_snap = pd.DataFrame()

    for m in markets:
        d = infer_market_date(m, cfg["tz"])
        if d is None or d not in nws:
            continue

        kind, lo, hi, bracket = market_condition(m)
        if kind is None:
            continue

        nrow = nws.get(d, {})
        nws_high = nrow.get("nws_high_f")
        if nws_high is None:
            continue

        peak_context = expected_peak_context(calibration, d, cfg["tz"])
        hours_to_peak = peak_context["hours_to_peak"]

        if d not in observed_context_cache:
            try:
                obs_df = get_station_observations(
                    cfg.get("station_id"), cfg["tz"], d
                )
                if obs_df is None:
                    obs_df = pd.DataFrame(columns=["time", "temp_f"])
            except Exception:
                obs_df = pd.DataFrame(columns=["time", "temp_f"])

            try:
                obs_high, obs_high_time = observed_high_details_from_df(
                    obs_df, cfg["tz"], d
                )
            except Exception:
                obs_high, obs_high_time = None, None

            observed_context_cache[d] = {
                "frame": obs_df,
                "high": obs_high,
                "high_time": obs_high_time,
            }

        obs_ctx = observed_context_cache[d]

        if d not in live_context_cache:
            trajectory_live = live_trajectory_adjustment(
                city,
                cfg,
                d,
                hours_to_peak,
                snapshot_df=recent_snap,
                observed_df=obs_ctx["frame"],
            )
            live_context_cache[d] = {
                "trajectory_score": trajectory_live.get(
                    "agreement_score", 0.5
                ),
                "trajectory_live": trajectory_live,
                "forecast_evolution": forecast_evolution_adjustment(
                    city,
                    cfg,
                    d,
                    hours_to_peak,
                    calibration,
                    snapshot_df=recent_snap,
                ),
                "recent_days": recent_completed_day_adjustment(
                    d, hours_to_peak, calibration
                ),
            }

        trajectory_score = live_context_cache[d]["trajectory_score"]
        trajectory_live = live_context_cache[d]["trajectory_live"]
        forecast_evolution = live_context_cache[d]["forecast_evolution"]
        recent_days = live_context_cache[d]["recent_days"]

        # Current-day observations affect betting probabilities and stay in the
        # fast scan. Previous-day history remains lazy/display-only.
        observed_high = obs_ctx["high"]
        observed_high_time = obs_ctx["high_time"]

        previous_day_high = None
        previous_day_high_time = None
        previous_3day_avg_high = None
        previous_day_prediction = None
        previous_day_predicted = None

        hours_left = hours_to_deadline(m, cfg["tz"])

        historical_bias_adj = float(nws_bias_f(hours_to_peak, calibration) or 0.0)
        live_adj = float(trajectory_live["adjustment_f"] or 0.0)
        evolution_adj = float(forecast_evolution["adjustment_f"] or 0.0)
        recent_adj = float(recent_days["adjustment_f"] or 0.0)

        # Conservative consensus correction:
        # live observations matter most, forecast-evolution next, while historical
        # and recent-day effects are deliberately shrunk to avoid double-counting.
        center_correction = (
            0.30 * historical_bias_adj
            + 0.55 * evolution_adj
            + 1.00 * live_adj
            + 0.45 * recent_adj
        )

        # Ordinary corrections are capped at +/-2°F. Larger historical misses
        # should widen sigma instead of producing a giant center shift.
        center_correction = max(-2.0, min(2.0, center_correction))

        unconstrained_nws_center = float(nws_high) + center_correction
        adjusted_nws_center = constrain_adjusted_high_center(
            raw_nws_high=nws_high,
            proposed_center=unconstrained_nws_center,
            observed_high=observed_high,
            hours_to_peak=hours_to_peak,
            trajectory_live=trajectory_live,
        )

        daily_high_locked = clearly_past_daily_peak(
            hours_to_peak,
            trajectory_live,
            observed_high,
        )
        if daily_high_locked:
            adjusted_nws_center = float(observed_high)

        post_peak_sigma_factor = 1.0
        if hours_to_peak is not None and float(hours_to_peak) <= 0:
            recent_trend = trajectory_live.get("recent_obs_trend_f")
            if recent_trend is not None and not pd.isna(recent_trend):
                post_peak_sigma_factor = 0.58 if float(recent_trend) <= 0.5 else 0.72
            else:
                post_peak_sigma_factor = 0.68

        combined_sigma_multiplier = (
            trajectory_live["sigma_multiplier"]
            * forecast_evolution["sigma_multiplier"]
            * recent_days["sigma_multiplier"]
            * post_peak_sigma_factor
        )

        base_final_sigma = max(
            0.35,
            nws_sigma_f(hours_to_peak, calibration)
            * combined_sigma_multiplier,
        )

        if daily_high_locked:
            # Hard post-peak rule: once the day is clearly past its hottest
            # period, the observed high is the final-high distribution.
            locked_yes = point_forecast_supports_yes(
                float(observed_high), kind, lo, hi
            )
            p_yes = None if locked_yes is None else (1.0 if locked_yes else 0.0)
            combined_sigma_multiplier = 0.05
            heating_params = {
                "center_f": float(observed_high),
                "sigma_down_f": 0.12,
                "sigma_up_f": 0.12,
                "observed_floor_f": float(observed_high),
                "expected_remaining_heating_f": 0.0,
                "upside_p95_f": float(observed_high),
                "peak_weight": 1.0,
            }
        else:
            p_yes, heating_params = weatheredge_yes_probability(
                nws_high=nws_high,
                kind=kind,
                lo=lo,
                hi=hi,
                adjusted_center=adjusted_nws_center,
                observed_high=observed_high,
                hours_to_peak=hours_to_peak,
                base_sigma=base_final_sigma,
                trajectory_live=trajectory_live,
            )
            # The canonical displayed center should match the distribution
            # actually used for betting probabilities.
            adjusted_nws_center = heating_params["center_f"]

        if p_yes is None:
            continue

        # GFS stays lazy/display-only; do not spend live-scan time computing it.
        ensemble_median = ensemble_low = ensemble_high = None
        ensemble_daily_highs = []

        event_ticker = m.get("event_ticker")
        event_info = {}

        title = m.get("title") or f"{city} high temperature"
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or bracket
        settlement = "Open contract for settlement source"
        contract_url = None
        implied_temp = implied_by_event.get(event_ticker)
        temp_gap = float(nws_high) - implied_temp if implied_temp is not None else None
        nws_support_yes = point_forecast_supports_yes(
            adjusted_nws_center, kind, lo, hi
        )
        settlement_safety = settlement_rounding_risk(
            observed_high, kind, lo, hi
        )
        rounding_distance = settlement_safety["nearest_distance_f"]
        settlement_rounding_risk_flag = settlement_safety["risk"]
        settlement_ambiguous_degrees = settlement_safety["ambiguous_degrees"]

        side_data = [
            ("YES", to_float(m.get("yes_ask_dollars")), p_yes),
            ("NO", to_float(m.get("no_ask_dollars")), 1 - p_yes),
        ]

        for side, ask, p_nws in side_data:
            if ask is None or not (0 < ask < 1):
                continue

            edge = p_nws - ask
            score = opportunity_score(edge, temp_gap, hours_left)
            sigma_source = "historical" if calibration.get(_lead_bucket(hours_to_peak), {}).get("n", 0) >= 8 else "fallback"
            sigma_samples = calibration.get(_lead_bucket(hours_to_peak), {}).get("n", 0)
            quality_score = bet_quality_score(
                edge=edge,
                p_nws=p_nws,
                hours_left=hours_left,
                temp_gap=temp_gap,
                trajectory_score=trajectory_score,
                sigma_source=sigma_source,
                sigma_samples=sigma_samples,
            )
            nws_support = nws_support_yes if side == "YES" else (not nws_support_yes if nws_support_yes is not None else None)
            qualifies = (
                p_nws >= 0.55
                and edge >= 0.05
                and nws_support is True
            )
            suspicious = edge >= 0.30

            rows.append({
                "city": city,
                "station_hint": cfg["station"],
                "date": d,
                "date_label": pretty_date(d),
                "series_ticker": cfg["series"],
                "event_ticker": event_ticker,
                "market_ticker": m.get("ticker"),
                "event_title": title,
                "market_subtitle": subtitle,
                "bracket": bracket,
                "condition_kind": kind,
                "condition_low_f": lo,
                "condition_high_f": hi,
                "side": side,
                "ask": ask,
                "yes_ask": to_float(m.get("yes_ask_dollars")),
                "no_ask": to_float(m.get("no_ask_dollars")),
                # Legacy field names kept for the rest of the app, but these are NWS-only.
                "model_prob": p_nws,
                "conservative_prob": p_nws,
                "edge": edge,
                "conservative_edge": edge,
                "expected_roi": edge / ask,
                "n_members": None,
                "nws_high_f": nws_high,
                "observed_high_f": observed_high,
                "observed_high_time_local": observed_high_time,
                "rounding_risk_distance_f": rounding_distance,
                "settlement_rounding_risk": settlement_rounding_risk_flag,
                "settlement_ambiguous_degrees": settlement_ambiguous_degrees,
                "previous_day_high_f": previous_day_high,
                "previous_day_high_time_local": previous_day_high_time,
                "previous_3day_avg_high_f": previous_3day_avg_high,
                "previous_day_prediction_avg_f": (
                    previous_day_prediction.get("average_f") if previous_day_prediction else None
                ),
                "previous_day_prediction_low_f": (
                    previous_day_prediction.get("low_f") if previous_day_prediction else None
                ),
                "previous_day_prediction_high_f": (
                    previous_day_prediction.get("high_f") if previous_day_prediction else None
                ),
                "previous_day_prediction_latest_f": (
                    previous_day_prediction.get("latest_f") if previous_day_prediction else None
                ),
                "previous_day_prediction_n": (
                    previous_day_prediction.get("n_snapshots") if previous_day_prediction else 0
                ),
                "previous_day_prediction_source": (
                    previous_day_prediction.get("source_label") if previous_day_prediction else None
                ),
                "previous_day_predicted_high_f": (
                    previous_day_predicted.get("predicted_high_f")
                    if previous_day_predicted else None
                ),
                "previous_day_predicted_high_source": (
                    previous_day_predicted.get("source")
                    if previous_day_predicted else None
                ),
                "previous_day_prediction_consistent": (
                    True
                    if (
                        previous_day_prediction is None
                        or previous_day_predicted is None
                    )
                    else (
                        float(previous_day_prediction.get("low_f"))
                        <= float(previous_day_predicted.get("predicted_high_f"))
                        <= float(previous_day_prediction.get("high_f"))
                    )
                ),
                "observed_data_url": nws_climate_url(cfg),
                "nws_forecast": nrow.get("nws_detail"),
                "nws_forecast_url": nrow.get("nws_forecast_url"),
                "nws_sigma_f": (
                    0.12
                    if daily_high_locked
                    else max(
                        float(heating_params.get("sigma_down_f") or base_final_sigma),
                        float(heating_params.get("sigma_up_f") or base_final_sigma),
                    )
                ),
                "nws_bias_f": nws_bias_f(hours_to_peak, calibration),
                # Canonical WeatherEdge final-high center. The daily high can never
                # finish below a temperature that has already been observed.
                "adjusted_nws_center_f": adjusted_nws_center,
                "unconstrained_adjusted_center_f": unconstrained_nws_center,
                "expected_remaining_heating_f": heating_params.get(
                    "expected_remaining_heating_f"
                ),
                "final_high_sigma_down_f": heating_params.get("sigma_down_f"),
                "final_high_sigma_up_f": heating_params.get("sigma_up_f"),
                "final_high_upside_p95_f": heating_params.get("upside_p95_f"),
                "final_high_distribution_model": "asymmetric_remaining_heating_v1",
                "daily_high_locked": daily_high_locked,
                "trajectory_adjustment_f": trajectory_live["adjustment_f"],
                "trajectory_current_gap_f": trajectory_live["current_gap_f"],
                "trajectory_peak_weight": trajectory_live["peak_weight"],
                "trajectory_sigma_multiplier": trajectory_live["sigma_multiplier"],
                "forecast_evolution_adjustment_f": forecast_evolution["adjustment_f"],
                "forecast_evolution_sigma_multiplier": forecast_evolution["sigma_multiplier"],
                "forecast_evolution_matches": forecast_evolution["n_matches"],
                "forecast_evolution_raw_error_f": forecast_evolution["raw_historical_error_f"],
                "forecast_evolution_confidence": forecast_evolution["confidence"],
                "forecast_change_3h_f": forecast_evolution["change_3h_f"],
                "forecast_change_6h_f": forecast_evolution["change_6h_f"],
                "forecast_change_12h_f": forecast_evolution["change_12h_f"],
                "forecast_change_24h_f": forecast_evolution["change_24h_f"],
                "forecast_volatility_12h_f": forecast_evolution["volatility_12h_f"],
                "recent_days_adjustment_f": recent_days["adjustment_f"],
                "recent_days_sigma_multiplier": recent_days["sigma_multiplier"],
                "post_peak_sigma_factor": post_peak_sigma_factor,
                "recent_days_weighted_error_f": recent_days["weighted_error_f"],
                "recent_days_n": recent_days["n_days"],
                "recent_days_consistency": recent_days["consistency"],
                "yesterday_forecast_error_f": recent_days["yesterday_error_f"],
                "nws_sigma_source": "historical" if calibration.get(_lead_bucket(hours_to_peak), {}).get("n", 0) >= 8 else "fallback",
                "nws_sigma_samples": calibration.get(_lead_bucket(hours_to_peak), {}).get("n", 0),
                "hours_to_expected_peak": hours_to_peak,
                "expected_peak_local": peak_context["expected_peak_dt"],
                "peak_history_days": peak_context["n_peak_days"],
                "hours_to_settlement": hours_left,
                "kalshi_implied_temp_f": implied_temp,
                "temperature_mismatch_f": temp_gap,
                "trajectory_agreement_score": trajectory_score,
                "bet_quality_score": quality_score,
                "bet_quality_label": bet_quality_label(quality_score),
                "opportunity_score": score,
                # GFS fields are display-only from here down.
                "ensemble_median_f": ensemble_median,
                "ensemble_low_f": ensemble_low,
                "ensemble_high_f": ensemble_high,
                "ensemble_daily_highs_f": ensemble_daily_highs,
                "nws_support": nws_support,
                "median_support": None,
                "forecasts_agree": nws_support is True,
                "qualifies": qualifies,
                "agreement": "✅ NWS supports this side" if nws_support is True else "❌ NWS opposes this side",
                "settlement_source": settlement,
                "contract_url": contract_url,
                "kalshi_event_url": kalshi_event_url(cfg["series"], event_ticker),
                "volume": to_float(m.get("volume_fp")) or 0,
                "open_interest": to_float(m.get("open_interest_fp")) or 0,
                "suspicious": suspicious,
            })
    return rows




_PERSISTENT_SCAN_PATH = Path("/tmp/weatheredge_last_good_scan_schema4.pkl")

def load_persistent_scan(max_age_seconds=900):
    """Load the last successful universe scan for a near-instant startup."""
    try:
        if not _PERSISTENT_SCAN_PATH.exists():
            return None
        age = time.time() - _PERSISTENT_SCAN_PATH.stat().st_mtime
        if age > max_age_seconds:
            return None
        with _PERSISTENT_SCAN_PATH.open("rb") as fh:
            payload = pickle.load(fh)
        if (
            isinstance(payload, tuple)
            and len(payload) == 3
            and payload[0]
        ):
            return payload
    except Exception:
        pass
    return None


def save_persistent_scan(payload):
    try:
        tmp = _PERSISTENT_SCAN_PATH.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(_PERSISTENT_SCAN_PATH)
    except Exception:
        pass


@st.cache_data(ttl=300, show_spinner=False)
def scan_live_market_universe():
    """
    Refresh all supported cities concurrently, then cache the combined result
    for five minutes. Network-bound city scans benefit substantially from a
    modest thread pool.
    """
    rows = []
    errors = []
    # Kalshi calls are separately serialized/paced by get_json, while NWS and
    # cached local processing can safely use a larger worker pool.
    max_workers = min(10, max(1, len(PRESETS)))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(build_city_rows, city, cfg): city
            for city, cfg in PRESETS.items()
        }
        for future in as_completed(futures):
            city = futures[future]
            try:
                city_rows = future.result()
                if city_rows:
                    rows.extend(city_rows)
            except Exception as exc:
                errors.append(f"{city}: {exc}")

    refreshed_at = pd.Timestamp.now(tz="UTC").isoformat()
    payload = (rows, errors, refreshed_at)
    if rows:
        save_persistent_scan(payload)
    return payload




def clear_live_data_caches():
    """
    Force the next scan to fetch fresh live market/weather inputs.

    Function names are resolved from globals at click-time so a missing or
    reordered optional cache helper can never crash the app.
    """
    try:
        scan_live_market_universe.clear()
    except Exception:
        pass

    cache_names = (
        "get_kalshi_markets",
        "get_nws_daily",
        "get_live_station_observations",
        "get_observed_high_so_far",
    )
    for name in cache_names:
        func = globals().get(name)
        if func is None:
            continue
        try:
            func.clear()
        except Exception:
            pass



def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{100*x:.1f}%"


# -----------------------------------------------------------------------------
# Forecast snapshot history stored in Supabase
# -----------------------------------------------------------------------------
SNAPSHOT_TABLE = "weather_forecast_snapshots"


def _secret(name):
    """Read secrets robustly from Streamlit (flat or [supabase]) or env vars."""
    aliases = {
        "SUPABASE_URL": ["SUPABASE_URL", "supabase_url", "url"],
        "SUPABASE_SERVICE_ROLE_KEY": [
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_SECRET_KEY",
            "supabase_service_role_key",
            "supabase_secret_key",
            "service_role_key",
            "secret_key",
        ],
        "SUPABASE_ANON_KEY": ["SUPABASE_ANON_KEY", "supabase_anon_key", "anon_key"],
    }
    names = aliases.get(name, [name])

    # 1) Flat Streamlit secrets, e.g. SUPABASE_URL = "..."
    try:
        for candidate in names:
            value = st.secrets.get(candidate)
            if value:
                return str(value).strip()
    except Exception:
        pass

    # 2) Nested TOML, e.g. [supabase] url = "..."
    try:
        section = st.secrets.get("supabase")
        if section:
            for candidate in names:
                short = candidate.lower().replace("supabase_", "")
                value = section.get(candidate) or section.get(short)
                if value:
                    return str(value).strip()
    except Exception:
        pass

    # 3) Environment variables.
    for candidate in names:
        value = os.getenv(candidate)
        if value:
            return str(value).strip()
    return None


def _first_present(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_snapshot_rows(city, max_rows=10000):
    """Fetch only recent snapshot rows needed by live same-day calculations."""
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_SERVICE_ROLE_KEY") or _secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return [], "Snapshot data unavailable."

    endpoint = f"{url.rstrip('/')}/rest/v1/{SNAPSHOT_TABLE}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    params = {
        "select": "*",
        "city": f"eq.{city}",
        "order": "id.desc",
        "limit": int(max_rows),
    }
    try:
        response = requests.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=18,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            return [], "Unexpected snapshot response."
        return rows, None
    except Exception as exc:
        return [], f"Could not read recent forecast snapshots: {exc}"


@st.cache_resource(ttl=300, show_spinner=False)
def get_recent_normalized_snapshot_rows(city):
    rows, err = get_recent_snapshot_rows(city)
    if err:
        return pd.DataFrame(), err
    return normalize_snapshot_rows(rows)


@st.cache_data(ttl=60, show_spinner=False)
def get_snapshot_rows(city, contract_date):
    """
    Read stored hourly projections for the city across collector runs.

    We intentionally do NOT filter by contract_date here. Historical cron runs may
    contain forecast-valid hours needed immediately before the selected contract day.

    The collector writes 156 hourly rows per city per run, so this function
    paginates instead of relying on Supabase's usual 1,000-row response cap.
    """
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_SERVICE_ROLE_KEY") or _secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return [], (
            "Snapshot charts cannot see the Supabase credentials for this Streamlit deployment. "
            "Add SUPABASE_URL plus SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SECRET_KEY) in Manage app → Secrets."
        )

    endpoint = f"{url.rstrip('/')}/rest/v1/{SNAPSHOT_TABLE}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    # IMPORTANT: do not filter the snapshot table to only the selected contract_date.
    # The collector stores forecast hours under the contract/date context of each run.
    # If we filter here, forecasts that were captured yesterday for hours leading into
    # the selected day disappear from the chart even though cron successfully saved them.
    # Pull the city's most recent stored history, then let chart/calibration code
    # select the exact valid-time window. Recent-first ordering is essential because
    # the table can be larger than our bounded pagination window.
    params = {
        "select": "*",
        "city": f"eq.{city}",
        # Fetch newest rows first. The table can exceed the pagination cap;
        # oldest-first caused yesterday's snapshots to disappear once enough
        # collector history accumulated.
        "order": "id.desc",
    }

    rows = []
    page_size = 1000
    for start in range(0, 40000, page_size):
        page_headers = dict(headers)
        page_headers["Range"] = f"{start}-{start + page_size - 1}"
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=page_headers,
                timeout=20,
            )
            response.raise_for_status()
            batch = response.json()
        except Exception as exc:
            return [], f"Could not read forecast snapshots from Supabase: {exc}"

        if not isinstance(batch, list):
            return [], "Supabase returned an unexpected response for forecast snapshots."
        rows.extend(batch)
        if len(batch) < page_size:
            break

    return rows, None


def normalize_snapshot_rows(rows):
    """Normalize collector column names into snapshot_at, valid_at, and temp_f."""
    if not rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(rows)
    columns = set(df.columns)

    snapshot_col = _first_present(columns, [
        "snapshot_time", "snapshot_at", "collected_at", "captured_at",
        "retrieved_at", "run_at", "created_at",
    ])
    valid_col = _first_present(columns, [
        "forecast_time", "forecast_at", "valid_time", "valid_at",
        "period_start", "start_time", "forecast_start", "hour_start",
    ])
    temp_col = _first_present(columns, [
        "temperature_f", "temp_f", "forecast_temp_f",
        "forecast_temperature_f", "temperature",
    ])

    missing = []
    if snapshot_col is None:
        missing.append("snapshot timestamp")
    if valid_col is None:
        missing.append("forecast/valid timestamp")
    if temp_col is None:
        missing.append("forecast temperature")
    if missing:
        return pd.DataFrame(), (
            "I found the snapshot table, but could not identify "
            + ", ".join(missing)
            + ". Columns returned: "
            + ", ".join(map(str, df.columns))
        )

    out = df.copy()
    out["snapshot_at"] = pd.to_datetime(out[snapshot_col], utc=True, errors="coerce")
    out["valid_at"] = pd.to_datetime(out[valid_col], utc=True, errors="coerce")
    out["temp_f"] = pd.to_numeric(out[temp_col], errors="coerce")
    out = out.dropna(subset=["snapshot_at", "valid_at", "temp_f"]).copy()

    # Rows from one collector run may have insertion timestamps a few seconds apart.
    # Bucketing by minute correctly reconstructs the :07 / :37 snapshots.
    out["snapshot_key"] = out["snapshot_at"].dt.floor("min")
    out = out.sort_values(["snapshot_at", "valid_at"]).reset_index(drop=True)
    return out, None



@st.cache_resource(ttl=900, show_spinner=False)
def get_normalized_snapshot_rows(city):
    """
    Return normalized stored forecast snapshots once per city.

    Live trajectory, forecast evolution, calibration, and City Explorer all use
    the same dataframe instead of repeatedly parsing tens of thousands of rows.
    """
    rows, err = get_snapshot_rows(city, None)
    if err:
        return pd.DataFrame(), err
    return normalize_snapshot_rows(rows)



def _fetch_station_observations(station_id, tz_name, target_date):
    """Observed temperatures for the selected local calendar day."""
    if not station_id:
        return pd.DataFrame(columns=["time", "temp_f"])

    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
    now_local = datetime.now(tz)
    if target_date > now_local.date():
        return pd.DataFrame(columns=["time", "temp_f"])

    if target_date == now_local.date():
        end_local = now_local
    else:
        end_local = datetime.combine(target_date, datetime.max.time(), tzinfo=tz)

    params = {
        "start": start_local.isoformat(),
        "end": end_local.isoformat(),
        "limit": 500,
    }
    try:
        data = get_json(
            f"https://api.weather.gov/stations/{station_id}/observations",
            params=params,
        )
    except Exception:
        return pd.DataFrame(columns=["time", "temp_f"])

    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        raw_time = props.get("timestamp")
        value_c = ((props.get("temperature") or {}).get("value"))
        if raw_time is None or value_c is None:
            continue
        try:
            rows.append({
                "time": pd.Timestamp(raw_time),
                "temp_f": float(value_c) * 9 / 5 + 32,
            })
        except (TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame(columns=["time", "temp_f"])
    return pd.DataFrame(rows).sort_values("time")

@st.cache_data(ttl=45, show_spinner=False)
def get_live_station_observations(station_id, tz_name, target_date):
    return _fetch_station_observations(station_id, tz_name, target_date)


@st.cache_data(ttl=86400, show_spinner=False)
def get_historical_station_observations(station_id, tz_name, target_date):
    return _fetch_station_observations(station_id, tz_name, target_date)


def get_station_observations(station_id, tz_name, target_date):
    """Use a short cache for today and a long cache for completed days."""
    if not station_id:
        return pd.DataFrame(columns=["time", "temp_f"])
    today = datetime.now(ZoneInfo(tz_name)).date()
    if target_date < today:
        return get_historical_station_observations(station_id, tz_name, target_date)
    return get_live_station_observations(station_id, tz_name, target_date)









def previous_day_predicted_high(city, tz_name, target_date, calibration=None):
    """
    Return the latest NWS-predicted high for the previous calendar day.

    This delegates to previous_day_forecast_summary so the dotted line can never
    disagree with the shaded min/max range.
    """
    summary = previous_day_forecast_summary(
        city, tz_name, target_date, calibration=calibration
    )
    if summary is None:
        return None
    return {
        "predicted_high_f": float(summary["latest_f"]),
        "snapshot_at": None,
        "source": summary.get("source_label") or "stored NWS snapshots",
    }




def previous_day_forecast_summary(city, tz_name, target_date, calibration=None):
    """
    Canonical previous-day NWS forecast summary.

    IMPORTANT: only use snapshots that were captured while the previous day's
    high-temperature forecast was still meaningful. Late-day snapshots can contain
    only the remaining evening hours; treating their evening maximum as the day's
    predicted high creates an artificially huge historical range.

    The dotted line and shaded range always come from the same eligible snapshot set.
    """
    previous_day = target_date - timedelta(days=1)
    tz = ZoneInfo(tz_name)

    snap, norm_err = get_normalized_snapshot_rows(city)
    if norm_err or snap.empty:
        return None

    work = snap.copy()
    work["valid_local"] = work["valid_at"].dt.tz_convert(tz)
    work["snapshot_local"] = work["snapshot_at"].dt.tz_convert(tz)
    work["valid_date"] = work["valid_local"].dt.date

    target = work[work["valid_date"] == previous_day].copy()
    if target.empty:
        return None

    # Determine the actual previous-day peak time directly from observations when
    # possible. This makes the chart independent of whether calibration metadata
    # happened to be available in the lazy City Explorer path.
    peak_dt = None
    try:
        cfg = PRESETS.get(city) or {}
        station_id = cfg.get("station_id")
        if station_id:
            obs = get_station_observations(station_id, tz_name, previous_day)
            if obs is not None and not obs.empty:
                obs2 = obs.copy()
                obs2["time_local"] = pd.to_datetime(
                    obs2["time"], utc=True, errors="coerce"
                ).dt.tz_convert(tz)
                obs2["temp_f"] = pd.to_numeric(obs2["temp_f"], errors="coerce")
                obs2 = obs2.dropna(subset=["time_local", "temp_f"]).sort_values("time_local")
                if not obs2.empty:
                    high = float(obs2["temp_f"].max())
                    tied = obs2[obs2["temp_f"] >= high - 0.15]
                    if not tied.empty:
                        peak_dt = pd.Timestamp(tied.iloc[0]["time_local"])
    except Exception:
        peak_dt = None

    # Calibration metadata is a secondary source for peak time.
    if peak_dt is None:
        try:
            peak_minutes = (
                (calibration or {})
                .get("__meta__", {})
                .get("peak_minutes_by_date", {})
                .get(previous_day.isoformat())
            )
            if peak_minutes is not None:
                peak_dt = (
                    pd.Timestamp(
                        datetime.combine(
                            previous_day,
                            datetime.min.time(),
                            tzinfo=tz,
                        )
                    )
                    + pd.Timedelta(minutes=float(peak_minutes))
                )
        except Exception:
            peak_dt = None

    # Group each collector run and retain coverage metadata. A valid daily-high
    # snapshot must still include the day's peak window in its forecast horizon.
    projected = (
        target.groupby("snapshot_key", as_index=False)
        .agg(
            projected_high_f=("temp_f", "max"),
            snapshot_at=("snapshot_at", "min"),
            valid_min=("valid_local", "min"),
            valid_max=("valid_local", "max"),
        )
        .dropna(subset=["projected_high_f", "snapshot_at"])
        .sort_values("snapshot_at")
    )
    if projected.empty:
        return None

    now_local = pd.Timestamp.now(tz=tz_name)

    if peak_dt is not None:
        eligible = projected[
            projected["snapshot_at"].apply(
                lambda x: pd.Timestamp(x).tz_convert(tz) <= peak_dt
            )
            & (projected["valid_min"] <= peak_dt)
            & (projected["valid_max"] >= peak_dt)
        ].copy()
    else:
        # Conservative fallback when no observed peak time is available:
        # only use snapshots captured before 6 PM local, avoiding the common
        # failure mode where evening-only residual hours masquerade as the day's high.
        fallback_cutoff = pd.Timestamp(
            datetime.combine(
                previous_day,
                datetime.min.time(),
                tzinfo=tz,
            ) + timedelta(hours=18)
        )
        if previous_day == now_local.date():
            fallback_cutoff = min(fallback_cutoff, now_local)
        eligible = projected[
            projected["snapshot_at"].apply(
                lambda x: pd.Timestamp(x).tz_convert(tz) <= fallback_cutoff
            )
        ].copy()

    if eligible.empty:
        # Last resort: use the earliest snapshot set rather than late-day residual
        # forecasts, because the latter systematically understate the day's predicted high.
        eligible = projected.head(max(1, min(6, len(projected)))).copy()

    values = eligible["projected_high_f"].astype(float)
    latest = float(eligible.iloc[-1]["projected_high_f"])
    low = float(values.min())
    high = float(values.max())

    low = min(low, latest)
    high = max(high, latest)

    return {
        "average_f": float(values.mean()),
        "low_f": low,
        "high_f": high,
        "latest_f": latest,
        "n_snapshots": int(len(values)),
        "source_label": (
            "stored NWS snapshots covering previous-day peak"
            if peak_dt is not None
            else "stored NWS pre-evening snapshots"
        ),
    }



def observed_high_details_from_df(obs, tz_name, target_date=None):
    """Compute daily high/time from an already-loaded observation dataframe."""
    if obs is None or obs.empty:
        return None, None
    tz = ZoneInfo(tz_name)
    work = obs.copy()
    work["time_local"] = pd.to_datetime(
        work["time"], utc=True, errors="coerce"
    ).dt.tz_convert(tz)
    work["temp_f"] = pd.to_numeric(work["temp_f"], errors="coerce")
    work = work.dropna(subset=["time_local", "temp_f"]).sort_values("time_local")
    if target_date is not None:
        work = work[work["time_local"].dt.date == target_date]
    if work.empty:
        return None, None
    high = float(work["temp_f"].max())
    tied = work[work["temp_f"] >= high - 1e-9]
    peak_time = tied.iloc[0]["time_local"] if not tied.empty else None
    return high, peak_time


def observed_daily_high_details(station_id, tz_name, target_date):
    """Return the observed daily high and first local timestamp tied for that high."""
    obs = get_station_observations(station_id, tz_name, target_date)
    return observed_high_details_from_df(obs, tz_name, target_date)


def recent_observed_high_summary(station_id, tz_name, target_date):
    """Return previous-day high/time and prior-3-day average high when available."""
    prev_high, prev_high_time = observed_daily_high_details(
        station_id, tz_name, target_date - timedelta(days=1)
    )
    highs = []
    for days_back in (1, 2, 3):
        high, _ = observed_daily_high_details(
            station_id, tz_name, target_date - timedelta(days=days_back)
        )
        if high is None:
            return prev_high, prev_high_time, None
        highs.append(float(high))
    return prev_high, prev_high_time, sum(highs) / 3.0


def _chart_base_properties(chart, height=300):
    return chart.properties(height=height).configure_view(strokeWidth=0)





@st.cache_data(ttl=300, show_spinner=False)
def lazy_city_explorer_weather(city, target_date):
    """
    Load chart-only weather context for one opened city/date.

    Previous-day NWS prediction history is reconstructed DIRECTLY from stored
    snapshot rows first. It does not depend on historical calibration succeeding.
    """
    cfg = PRESETS.get(city)
    if not cfg:
        return {}

    result = {}

    # Previous-day observed context.
    try:
        previous_day_high, previous_day_high_time, previous_3day_avg = (
            recent_observed_high_summary(
                cfg.get("station_id"), cfg["tz"], target_date
            )
        )
        result["previous_day_high_f"] = previous_day_high
        result["previous_day_high_time_local"] = previous_day_high_time
        result["previous_3day_avg_high_f"] = previous_3day_avg
    except Exception:
        pass

    # FIRST CHOICE: raw stored NWS snapshots for the previous calendar day.
    # This is the canonical source for both the last predicted high and the range.
    summary = None
    try:
        summary = previous_day_forecast_summary(
            city,
            cfg["tz"],
            target_date,
            calibration=None,
        )
    except Exception:
        summary = None

    if summary:
        latest = float(summary["latest_f"])
        low = float(summary["low_f"])
        high = float(summary["high_f"])

        # Defensive consistency guarantee.
        low = min(low, latest)
        high = max(high, latest)

        result.update({
            "previous_day_prediction_avg_f": summary.get("average_f"),
            "previous_day_prediction_low_f": low,
            "previous_day_prediction_high_f": high,
            "previous_day_prediction_latest_f": latest,
            "previous_day_prediction_n": summary.get("n_snapshots", 0),
            "previous_day_prediction_source": (
                summary.get("source_label") or "stored NWS snapshots"
            ),
            "previous_day_predicted_high_f": latest,
            "previous_day_predicted_high_source": (
                summary.get("source_label") or "stored NWS snapshots"
            ),
            "previous_day_prediction_consistent": True,
        })

    # SECOND CHOICE: calibration metadata already built for ranking.
    # This is fallback-only and cannot suppress the raw snapshot result above.
    if summary is None:
        try:
            calibration = build_nws_error_calibration(
                city,
                cfg.get("station_id"),
                cfg["tz"],
                cache_date=datetime.now(
                    ZoneInfo(cfg["tz"])
                ).date().isoformat(),
            )
            prev_date = target_date - timedelta(days=1)
            recent_errors = (
                calibration.get("__meta__", {})
                .get("recent_daily_errors", [])
                or []
            )
            prior_match = next(
                (
                    item for item in reversed(recent_errors)
                    if str(item.get("date")) == prev_date.isoformat()
                ),
                None,
            )
            if prior_match is not None:
                f = float(prior_match["projected_high_f"])
                result.update({
                    "previous_day_prediction_avg_f": f,
                    "previous_day_prediction_low_f": f,
                    "previous_day_prediction_high_f": f,
                    "previous_day_prediction_latest_f": f,
                    "previous_day_prediction_n": 1,
                    "previous_day_prediction_source": (
                        "historical calibration fallback"
                    ),
                    "previous_day_predicted_high_f": f,
                    "previous_day_predicted_high_source": (
                        "historical calibration fallback"
                    ),
                    "previous_day_prediction_consistent": True,
                })
        except Exception:
            pass

    # THIRD CHOICE: if "previous day" is actually today for a future-date market,
    # the live NWS daily forecast can supply the single current value.
    if result.get("previous_day_predicted_high_f") is None:
        try:
            prev_date = target_date - timedelta(days=1)
            nws_rows = {
                r["date"]: r
                for r in get_nws_daily(cfg["lat"], cfg["lon"])
            }
            fallback = (nws_rows.get(prev_date) or {}).get("nws_high_f")
            if fallback is not None:
                f = float(fallback)
                result.update({
                    "previous_day_prediction_avg_f": f,
                    "previous_day_prediction_low_f": f,
                    "previous_day_prediction_high_f": f,
                    "previous_day_prediction_latest_f": f,
                    "previous_day_prediction_n": 1,
                    "previous_day_prediction_source": "current NWS daily forecast",
                    "previous_day_predicted_high_f": f,
                    "previous_day_predicted_high_source": "current NWS daily forecast",
                    "previous_day_prediction_consistent": True,
                })
        except Exception:
            pass

    # GFS remains display-only and lazy.
    try:
        ens = get_gfs_ensemble_daily_highs(
            cfg["lat"], cfg["lon"], cfg["tz"]
        )
        if ens is not None and target_date in ens.index:
            members = pd.Series(ens.loc[target_date].values).dropna().astype(float)
            if not members.empty:
                result["ensemble_daily_highs_f"] = [
                    float(v) for v in members.tolist()
                ]
                result["ensemble_median_f"] = float(members.median())
                result["ensemble_low_f"] = float(members.quantile(0.10))
                result["ensemble_high_f"] = float(members.quantile(0.90))
    except Exception:
        pass

    return result


def hydrate_city_explorer_row(row):
    """Merge lazy chart data and display metadata into a selected market row."""
    try:
        out = row.copy()

        # Kalshi metadata is display-only. Fetch it here rather than during the
        # all-city ranking scan, which saves dozens of rate-limited requests.
        try:
            series_info = get_series_info(str(row.get("series_ticker")))
        except Exception:
            series_info = {}
        try:
            event_info = get_event(str(row.get("event_ticker")))
        except Exception:
            event_info = {}

        if event_info.get("title"):
            out["event_title"] = event_info.get("title")
        elif series_info.get("title") and not out.get("event_title"):
            out["event_title"] = series_info.get("title")

        out["settlement_source"] = source_names(series_info, event_info)
        if series_info.get("contract_url"):
            out["contract_url"] = series_info.get("contract_url")
        extra = lazy_city_explorer_weather(
            str(row.get("city")),
            row.get("date"),
        )
        for key, value in extra.items():
            out[key] = value

        # GFS final-high values cannot finish below an already observed high.
        observed = out.get("observed_high_f")
        members = out.get("ensemble_daily_highs_f") or []
        if observed is not None and not pd.isna(observed) and members:
            clipped = [max(float(observed), float(v)) for v in members]
            out["ensemble_daily_highs_f"] = clipped
            s = pd.Series(clipped, dtype=float)
            out["ensemble_median_f"] = float(s.median())
            out["ensemble_low_f"] = float(s.quantile(0.10))
            out["ensemble_high_f"] = float(s.quantile(0.90))
        return out
    except Exception:
        return row


def forecast_range_summary_chart(row, show_bet_overlay=False):
    """
    Daily-high probability view.

    NWS: the exact calibrated normal distribution used by the NWS probability
    engine, including historical bias, peak-relative sigma, and truncation at an
    already-observed daily high.

    GFS: an empirical kernel-density estimate from the GFS ensemble member daily
    highs. GFS remains visual/reference-only and does not enter NWS probability
    or Bet Quality calculations.
    """
    def clean_number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    nws_raw = clean_number(row.get("nws_high_f"))
    nws_bias = clean_number(row.get("nws_bias_f")) or 0.0
    trajectory_adjustment = clean_number(row.get("trajectory_adjustment_f")) or 0.0
    evolution_adjustment = clean_number(row.get("forecast_evolution_adjustment_f")) or 0.0
    recent_days_adjustment = clean_number(row.get("recent_days_adjustment_f")) or 0.0
    nws_sigma = clean_number(row.get("nws_sigma_f"))
    observed = clean_number(row.get("observed_high_f"))
    implied = clean_number(row.get("kalshi_implied_temp_f"))

    # Selected Kalshi contract metadata used by the Top Bet overlay.
    bet_kind = row.get("condition_kind")
    bet_low = clean_number(row.get("condition_low_f"))
    bet_high = clean_number(row.get("condition_high_f"))
    bet_side = str(row.get("side") or "").upper()

    gfs_values = [
        float(v) for v in (row.get("ensemble_daily_highs_f") or [])
        if v is not None and math.isfinite(float(v))
    ]

    if nws_raw is None and not gfs_values:
        return None

    # Prefer the canonical center calculated during the live scan. For older
    # rows, rebuild it here and enforce the observed-high floor.
    nws_center = clean_number(row.get("adjusted_nws_center_f"))
    if nws_center is None and nws_raw is not None:
        nws_center = (
            nws_raw + nws_bias + trajectory_adjustment
            + evolution_adjustment + recent_days_adjustment
        )
        if observed is not None:
            nws_center = max(nws_center, observed)

    # Adaptive x-axis. Use the decision-relevant values plus the central GFS
    # ensemble range rather than letting one far-out ensemble member or a wide
    # theoretical tail force a huge temperature scale.
    core_anchors = [
        v for v in [nws_raw, nws_center, observed, implied]
        if v is not None
    ]

    gfs_axis_values = []
    if gfs_values:
        gs = pd.Series(gfs_values, dtype=float)
        if len(gs) >= 5:
            gfs_axis_values = [
                float(gs.quantile(0.10)),
                float(gs.quantile(0.50)),
                float(gs.quantile(0.90)),
            ]
        else:
            gfs_axis_values = [float(gs.min()), float(gs.max())]

    anchors = core_anchors + gfs_axis_values
    if not anchors:
        return None

    # For tight clusters, zoom in aggressively so a 1°F contract bucket and
    # small differences among observed/NWS/Kalshi values are actually readable.
    anchor_span = max(anchors) - min(anchors)
    sigma_pad = max(0.9, min(2.5, 2.2 * (nws_sigma or 1.0)))

    if anchor_span <= 3.0:
        pad = max(1.25, sigma_pad)
        min_span = 6.0
    elif anchor_span <= 6.0:
        pad = max(1.5, sigma_pad)
        min_span = 8.0
    else:
        pad = max(2.0, sigma_pad)
        min_span = 10.0

    x_min = math.floor((min(anchors) - pad) * 2) / 2
    x_max = math.ceil((max(anchors) + pad) * 2) / 2

    if x_max - x_min < min_span:
        mid = (x_min + x_max) / 2.0
        x_min = math.floor((mid - min_span / 2.0) * 2) / 2
        x_max = math.ceil((mid + min_span / 2.0) * 2) / 2

    step = 0.05 if (x_max - x_min) <= 8 else 0.10
    n_steps = max(2, int(round((x_max - x_min) / step)) + 1)
    xs = [x_min + i * step for i in range(n_steps)]

    distribution_rows = []

    # NWS density, truncated below already-observed high exactly as the probability
    # engine conditions the final daily high.
    if nws_center is not None and nws_sigma is not None and nws_sigma > 0:
        normalizer = nws_sigma * math.sqrt(2.0 * math.pi)
        nws_pdf = []
        for x in xs:
            if observed is not None and x < observed:
                density = 0.0
            else:
                z = (x - nws_center) / nws_sigma
                density = math.exp(-0.5 * z * z) / normalizer
            nws_pdf.append(density)

        area = sum(nws_pdf) * step
        if area > 0:
            nws_pdf = [v / area for v in nws_pdf]
        for x, density in zip(xs, nws_pdf):
            distribution_rows.append({
                "temperature": x,
                "density": density,
                "Series": "NWS calibrated probability",
            })

    # GFS ensemble KDE. Bandwidth is deliberately modest so the actual ensemble
    # shape is visible without becoming a comb of individual member spikes.
    if gfs_values:
        n = len(gfs_values)
        std = float(pd.Series(gfs_values).std(ddof=1)) if n > 1 else 0.8
        if not math.isfinite(std) or std <= 0:
            std = 0.8
        bandwidth = max(0.35, min(1.5, 1.06 * std * (n ** (-1 / 5)) if n > 1 else 0.8))
        denom = n * bandwidth * math.sqrt(2.0 * math.pi)

        gfs_pdf = []
        for x in xs:
            if observed is not None and x < observed:
                val = 0.0
            else:
                val = sum(
                    math.exp(-0.5 * ((x - member) / bandwidth) ** 2)
                    for member in gfs_values
                ) / denom
            gfs_pdf.append(val)

        area = sum(gfs_pdf) * step
        if area > 0:
            gfs_pdf = [v / area for v in gfs_pdf]
        for x, density in zip(xs, gfs_pdf):
            distribution_rows.append({
                "temperature": x,
                "density": density,
                "Series": "GFS ensemble probability",
            })

    dist_df = pd.DataFrame(distribution_rows)
    layers = []


    if not dist_df.empty:
        color_scale = alt.Scale(
            domain=["NWS calibrated probability", "GFS ensemble probability"],
            range=["#9B7BFF", "#00D4FF"],
        )
        layers.append(
            alt.Chart(dist_df).mark_area(opacity=0.18).encode(
                x=alt.X(
                    "temperature:Q",
                    title="Final daily high temperature (°F)",
                    scale=alt.Scale(domain=[x_min, x_max]),
                    axis=alt.Axis(
                        tickCount=min(
                            24,
                            max(
                                9,
                                int((x_max - x_min) * (2 if (x_max - x_min) <= 8 else 1)) + 1,
                            ),
                        ),
                        tickSize=7,
                        grid=True,
                        gridOpacity=0.42,
                        labelExpr="datum.value + '°'",
                        labelFlush=False,
                    ),
                ),
                y=alt.Y("density:Q", title="Probability density"),
                color=alt.Color(
                    "Series:N",
                    scale=color_scale,
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Series:N", title="Distribution"),
                    alt.Tooltip("temperature:Q", title="Temperature", format=".1f"),
                    alt.Tooltip("density:Q", title="Density", format=".3f"),
                ],
            )
        )
        layers.append(
            alt.Chart(dist_df).mark_line(strokeWidth=3).encode(
                x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                y=alt.Y("density:Q"),
                color=alt.Color("Series:N", scale=color_scale, legend=None),
            )
        )

    previous_day_high = clean_number(row.get("previous_day_high_f"))
    previous_day_high_time = row.get("previous_day_high_time_local")
    previous_3day_avg = clean_number(row.get("previous_3day_avg_high_f"))
    observed_time = row.get("observed_high_time_local")

    # Historical references should stay visible when they are reasonably close,
    # but a distant prior-day value should not flatten today's decision region.
    prev_pred_avg = clean_number(row.get("previous_day_prediction_avg_f"))
    prev_pred_low = clean_number(row.get("previous_day_prediction_low_f"))
    prev_pred_high = clean_number(row.get("previous_day_prediction_high_f"))
    prev_pred_latest = clean_number(row.get("previous_day_prediction_latest_f"))
    previous_day_predicted_high = clean_number(
        row.get("previous_day_predicted_high_f")
    )
    if previous_day_predicted_high is not None:
        prev_pred_latest = previous_day_predicted_high
    prev_pred_n = int(row.get("previous_day_prediction_n") or 0)

    nearby_refs = [
        v for v in [
            previous_day_high,
            previous_3day_avg,
            prev_pred_avg,
            prev_pred_low,
            prev_pred_high,
            prev_pred_latest,
        ]
        if v is not None and (x_min - 2.0) <= v <= (x_max + 2.0)
    ]
    if nearby_refs:
        x_min = min(x_min, math.floor((min(nearby_refs) - 0.75) * 2) / 2)
        x_max = max(x_max, math.ceil((max(nearby_refs) + 0.75) * 2) / 2)

    # Historical reference: forecasts made the day before for the previous day's high.
    # IMPORTANT: do not render distant historical marks outside today's chart domain.
    # Vega-Lite otherwise unions the layered x scales and creates a detached empty
    # panel on mobile (the bug visible in V460). The legend still reports those
    # historical values even when they are too far away to plot usefully.
    history_is_nearby = (
        prev_pred_latest is not None
        and x_min <= prev_pred_latest <= x_max
    )
    history_range_is_nearby = (
        prev_pred_low is not None
        and prev_pred_high is not None
        and prev_pred_high >= x_min
        and prev_pred_low <= x_max
    )

    if prev_pred_avg is not None and history_range_is_nearby:
        history_color = "#36C2FF"

        # If the prediction changed during that day, shade the full min/max range.
        if (
            prev_pred_low is not None
            and prev_pred_high is not None
            and abs(prev_pred_high - prev_pred_low) >= 0.05
        ):
            history_band = pd.DataFrame([{
                "x1": prev_pred_low,
                "x2": prev_pred_high,
                "label": f"Previous-day forecast range · {prev_pred_n} snapshots",
            }])
            layers.append(
                alt.Chart(history_band)
                .mark_rect(
                    color=history_color,
                    opacity=0.16,
                    stroke=history_color,
                    strokeWidth=1.8,
                    strokeDash=[5, 4],
                )
                .encode(
                    x=alt.X("x1:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    x2=alt.X2("x2:Q"),
                    tooltip=[
                        alt.Tooltip("label:N", title="Historical forecast"),
                        alt.Tooltip("x1:Q", title="Forecast low", format=".1f"),
                        alt.Tooltip("x2:Q", title="Forecast high", format=".1f"),
                    ],
                )
            )


    # Most recent NWS predicted high for the previous day. Keep this as a
    # simple dotted vertical reference so yesterday's forecast is immediately
    # comparable with yesterday's observed high and today's distribution.
    if prev_pred_latest is not None and history_is_nearby:
        prev_nws_line = pd.DataFrame([{
            "temperature": prev_pred_latest,
            "label": f"Previous day NWS predicted high · {prev_pred_latest:.1f}°F",
        }])
        layers.append(
            alt.Chart(prev_nws_line)
            .mark_rule(
                color="#63D8FF",
                strokeWidth=3,
                strokeDash=[2, 5],
            )
            .encode(
                x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                tooltip=[
                    alt.Tooltip("label:N", title="Historical NWS reference"),
                    alt.Tooltip("temperature:Q", title="Predicted high", format=".1f"),
                ],
            )
        )

    marker_rows = []
    marker_specs = []

    if nws_raw is not None:
        marker_specs.append(("Latest NWS high", nws_raw, ""))
    if nws_center is not None:
        marker_specs.append(("WeatherEdge adjusted high", nws_center, ""))
    if observed is not None:
        observed_detail = ""
        if observed_time is not None and not pd.isna(observed_time):
            try:
                observed_detail = pd.Timestamp(observed_time).strftime("%-I:%M %p")
            except Exception:
                observed_detail = ""
        observed_label = "High observed so far"
        if observed_detail:
            observed_label += f" · {observed_detail}"
        marker_specs.append((observed_label, observed, observed_detail))
    if implied is not None:
        marker_specs.append(("Kalshi implied high", implied, ""))
    if prev_pred_latest is not None and x_min <= prev_pred_latest <= x_max:
        marker_specs.append((
            "NWS predicted high for previous day",
            prev_pred_latest,
            "",
        ))

    if previous_day_high is not None and x_min <= previous_day_high <= x_max:
        previous_day_detail = ""
        if previous_day_high_time is not None and not pd.isna(previous_day_high_time):
            try:
                previous_day_detail = pd.Timestamp(previous_day_high_time).strftime("%-I:%M %p")
            except Exception:
                previous_day_detail = ""
        previous_day_label = "Previous day high"
        if previous_day_detail:
            previous_day_label += f" · {previous_day_detail}"
        marker_specs.append((previous_day_label, previous_day_high, previous_day_detail))
    if previous_3day_avg is not None and x_min <= previous_3day_avg <= x_max:
        marker_specs.append(("Average high, previous 3 days", previous_3day_avg, ""))

    for label, value, detail in marker_specs:
        marker_rows.append({
            "temperature": value,
            "Marker": label,
            "detail": detail,
        })

    if marker_rows:
        marker_df = pd.DataFrame(marker_rows)
        def marker_color(label):
            if label.startswith("Latest NWS high"):
                return "#FFFFFF"
            if label.startswith("WeatherEdge adjusted high"):
                return "#9B7BFF"
            if label.startswith("High observed so far"):
                return "#FF2D8D"
            if label.startswith("Kalshi implied high"):
                return "#FFD400"
            if label.startswith("NWS predicted high for previous day"):
                return "#36C2FF"
            if label.startswith("Previous day high"):
                return "#FF6B5E"
            if label.startswith("Average high, previous 3 days"):
                return "#FF8A00"
            return "#BFC5D2"

        marker_domain = [r["Marker"] for r in marker_rows]
        marker_scale = alt.Scale(
            domain=marker_domain,
            range=[marker_color(label) for label in marker_domain],
        )
        layers.append(
            alt.Chart(marker_df).mark_rule(strokeWidth=2, strokeDash=[6, 4]).encode(
                x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                color=alt.Color(
                    "Marker:N",
                    scale=marker_scale,
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Marker:N", title="Reference"),
                    alt.Tooltip("temperature:Q", title="Temperature", format=".1f"),
                    alt.Tooltip("detail:N", title="Observed time"),
                ],
            )
        )


    # Draw the Top Bet overlay LAST so it cannot be hidden by the NWS/GFS fills.
    # The full-height shaded band represents the actual contract bucket.
    if show_bet_overlay and bet_side in ("YES", "NO"):
        band_lo = band_hi = None

        if bet_kind == "range" and bet_low is not None and bet_high is not None:
            # Visual contract highlight is exactly 1°F wide.
            midpoint = (bet_low + bet_high) / 2.0
            band_lo, band_hi = midpoint - 0.5, midpoint + 0.5
        elif bet_kind == "above" and bet_low is not None:
            # Threshold contracts get a compact 1°F visual reference band.
            band_lo, band_hi = bet_low - 0.5, bet_low + 0.5
        elif bet_kind in ("below", "below_equal") and bet_high is not None:
            band_lo, band_hi = bet_high - 0.5, bet_high + 0.5

        if band_lo is not None and band_hi is not None:
            if band_hi <= band_lo:
                midpoint = (band_lo + band_hi) / 2.0
                band_lo, band_hi = midpoint - 0.5, midpoint + 0.5

            # Expand the x-axis if necessary so the recommended bucket is always visible.
            x_min = min(x_min, math.floor(band_lo - 1))
            x_max = max(x_max, math.ceil(band_hi + 1))

            top_color = "#18D47B" if bet_side == "YES" else "#FF4050"
            top_label = f"TOP BET · {bet_side}"

            # Use DATA-SPACE y coordinates rather than pixel coordinates.
            # Vega-Lite/Altair reliably renders x/x2 + y/y2 quantitative rectangles
            # across layered charts; the prior pixel-value version could disappear.
            chart_y_max = (
                max(0.35, float(dist_df["density"].max()) * 1.18)
                if not dist_df.empty else 0.35
            )

            bet_df = pd.DataFrame([{
                "x1": band_lo,
                "x2": band_hi,
                "y1": 0.0,
                "y2": chart_y_max,
                "label": top_label,
            }])

            layers.append(
                alt.Chart(bet_df)
                .mark_rect(
                    color=top_color,
                    # Slightly stronger fill so the recommended contract bucket
                    # remains obvious on mobile without obscuring distributions.
                    opacity=0.24,
                    stroke=top_color,
                    strokeWidth=2.5,
                )
                .encode(
                    x=alt.X("x1:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    x2=alt.X2("x2:Q"),
                    y=alt.Y("y1:Q", scale=alt.Scale(domain=[0, chart_y_max])),
                    y2=alt.Y2("y2:Q"),
                    tooltip=[
                        alt.Tooltip("label:N", title="Recommended bet"),
                        alt.Tooltip("x1:Q", title="Bucket from", format=".1f"),
                        alt.Tooltip("x2:Q", title="Bucket to", format=".1f"),
                    ],
                )
            )

            # Put the YES/NO label inside the band near the top of the data area.
            label_df = pd.DataFrame([{
                "temperature": (band_lo + band_hi) / 2.0,
                "density": chart_y_max * 0.93,
                "label": top_label,
            }])
            layers.append(
                alt.Chart(label_df)
                .mark_text(
                    angle=0,
                    fontSize=11,
                    fontWeight="bold",
                    color=top_color,
                    baseline="bottom",
                    dy=-3,
                )
                .encode(
                    x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    y=alt.Y("density:Q", scale=alt.Scale(domain=[0, chart_y_max])),
                    text="label:N",
                    tooltip=[alt.Tooltip("label:N", title="Recommended bet")],
                )
            )


    if not layers:
        return None

    subtitle = (
        "NWS = probability model used for NWS chance. GFS = reference-only ensemble distribution."
    )

    main_chart = (
        alt.layer(*layers)
        .resolve_scale(color="independent", y="shared")
        .properties(
            height=280,
            background="#0D0F16",
            title=alt.TitleParams(
                text="Daily high probability distributions",
                subtitle=subtitle,
                anchor="start",
            ),
        )
    )

    return (
        main_chart
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            labelColor="#F1EDF7",
            titleColor="#FAF7FF",
            gridColor="#777185",
            gridOpacity=.38,
            domainColor="#AAA3B5",
            tickColor="#AAA3B5",
        )
        .configure_legend(
            title=None,
            labelFontSize=12,
            labelColor="#F1EDF7",
            symbolSize=150,
            padding=8,
        )
        .configure_title(
            fontSize=19,
            subtitleFontSize=12,
            color="#FAF7FF",
            subtitleColor="#D4CEDD",
        )
        .configure_view(strokeWidth=0)
    )


def render_probability_chart_legend(row):
    """Single high-contrast chart key for the daily-high probability figure."""
    def present(value):
        try:
            return value is not None and not pd.isna(value)
        except Exception:
            return value is not None

    items = [
        ("NWS probability distribution", "#9B7BFF", "solid"),
        ("GFS ensemble distribution", "#00D4FF", "solid"),
    ]

    if present(row.get("nws_high_f")):
        raw_nws = float(row.get("nws_high_f"))
        adjusted = row.get("adjusted_nws_center_f")
        if not present(adjusted):
            adjusted = (
                raw_nws
                + float(row.get("nws_bias_f") or 0.0)
                + float(row.get("trajectory_adjustment_f") or 0.0)
            )
            observed_floor = row.get("observed_high_f")
            if present(observed_floor):
                adjusted = max(float(adjusted), float(observed_floor))
        adjusted = float(adjusted)
        items.append((f"Latest NWS high · {raw_nws:.1f}°F", "#FFFFFF", "dash"))
        items.append((f"WeatherEdge adjusted high · {adjusted:.1f}°F", "#9B7BFF", "dash"))

    if present(row.get("observed_high_f")):
        obs_label = "High observed so far"
        obs_time = row.get("observed_high_time_local")
        if present(obs_time):
            try:
                obs_label += f" · {pd.Timestamp(obs_time).strftime('%-I:%M %p')}"
            except Exception:
                pass
        obs_label += f" · {float(row.get('observed_high_f')):.1f}°F"
        items.append((obs_label, "#FF2D8D", "dash"))

    if present(row.get("kalshi_implied_temp_f")):
        items.append((
            f"Kalshi implied high · {float(row.get('kalshi_implied_temp_f')):.1f}°F",
            "#FFD400",
            "dash",
        ))

    prev_pred_avg = row.get("previous_day_prediction_avg_f")
    prev_pred_low = row.get("previous_day_prediction_low_f")
    prev_pred_high = row.get("previous_day_prediction_high_f")
    prev_pred_latest = row.get("previous_day_prediction_latest_f")
    exact_prev_pred = row.get("previous_day_predicted_high_f")
    if present(exact_prev_pred):
        prev_pred_latest = exact_prev_pred
    prev_pred_n = int(row.get("previous_day_prediction_n") or 0)

    if present(prev_pred_latest):
        items.append((
            f"Previous day NWS predicted high · {float(prev_pred_latest):.1f}°F",
            "#63D8FF",
            "dot",
        ))

    if (
        present(prev_pred_low)
        and present(prev_pred_high)
        and abs(float(prev_pred_high) - float(prev_pred_low)) >= 0.05
    ):
        items.append((
            f"Previous day NWS prediction range · "
            f"{float(prev_pred_low):.1f}–{float(prev_pred_high):.1f}°F",
            "#36C2FF",
            "history_band",
        ))

    if present(row.get("previous_day_high_f")):
        prev_label = "Previous day high"
        prev_time = row.get("previous_day_high_time_local")
        if present(prev_time):
            try:
                prev_label += f" · {pd.Timestamp(prev_time).strftime('%-I:%M %p')}"
            except Exception:
                pass
        prev_label += f" · {float(row.get('previous_day_high_f')):.1f}°F"
        items.append((prev_label, "#FF6B5E", "dash"))

    if present(row.get("previous_3day_avg_high_f")):
        items.append((
            f"Average high, previous 3 days · {float(row.get('previous_3day_avg_high_f')):.1f}°F",
            "#FF8A00",
            "dash",
        ))

    status_text, status_kind = contract_status(row)
    if status_kind == "best":
        bet_side = str(row.get("side") or "").upper()
        top_color = "#18D47B" if bet_side == "YES" else "#FF4050"
        items.append((f"{status_text} · {bet_side} · shaded contract range", top_color, "band"))

    rows = []
    for label, color, kind in items:
        if kind == "solid":
            swatch = (
                f"<span style='width:30px;height:5px;border-radius:999px;"
                f"background:{color};display:inline-block;flex:0 0 30px'></span>"
            )
        elif kind == "circle":
            swatch = (
                f"<span style='width:14px;height:14px;border-radius:50%;"
                f"background:{color};border:2px solid #FFFFFF;"
                f"display:inline-block;flex:0 0 14px;margin-left:8px;margin-right:8px'></span>"
            )
        elif kind == "diamond":
            swatch = (
                f"<span style='width:14px;height:14px;background:{color};"
                f"border:2px solid #FFFFFF;transform:rotate(45deg);"
                f"display:inline-block;flex:0 0 14px;margin-left:8px;margin-right:8px'></span>"
            )
        elif kind == "band":
            swatch = (
                f"<span style='width:30px;height:16px;background:{color}2E;"
                f"border:2px solid {color};border-radius:3px;"
                f"display:inline-block;flex:0 0 30px'></span>"
            )
        elif kind == "history_band":
            swatch = (
                f"<span style='width:30px;height:16px;background:{color}1F;"
                f"border:2px dashed {color};border-radius:3px;"
                f"display:inline-block;flex:0 0 30px'></span>"
            )
        elif kind == "dot":
            swatch = (
                f"<span style='width:30px;height:0;border-top:3px dotted {color};"
                f"display:inline-block;flex:0 0 30px'></span>"
            )
        else:
            swatch = (
                f"<span style='width:30px;height:0;border-top:3px dashed {color};"
                f"display:inline-block;flex:0 0 30px'></span>"
            )

        rows.append(
            f"<div style='display:flex;align-items:center;gap:.7rem;'>"
            f"{swatch}"
            f"<span style='color:#F7F4FA;font-size:.92rem;line-height:1.3'>{label}</span>"
            f"</div>"
        )

    st.markdown(
        "<div style='margin:.45rem 0 1.1rem;padding:.85rem .95rem;border-radius:14px;"
        "background:#11131C;border:1px solid rgba(255,255,255,.14)'>"
        "<div style='font-size:.72rem;font-weight:850;letter-spacing:.12em;"
        "color:#CFC4FF;margin-bottom:.7rem'>CHART KEY</div>"
        f"<div style='display:grid;grid-template-columns:1fr;gap:.6rem'>{''.join(rows)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )



def latest_projection_chart(snapshot_df, observed_df, previous_observed_df, tz_name, target_date):
    """
    Observed vs NWS trajectory across previous + current day.

    Direct labels are deliberately minimal:
      - current-day max observed
      - current-day max NWS prediction
      - previous-day max observed
      - previous-day last predicted high

    Labels on the figure show temperature only. No minimum/end-point labels.
    """
    if snapshot_df.empty:
        return None, None

    tz = ZoneInfo(tz_name)
    latest_key = snapshot_df["snapshot_key"].max()

    work = snapshot_df.copy()
    work["time"] = work["valid_at"].dt.tz_convert(tz)
    work = work[work["snapshot_at"] <= work["valid_at"]].copy()

    display_start = pd.Timestamp(
        datetime.combine(
            target_date - timedelta(days=1),
            datetime.min.time(),
            tzinfo=tz,
        )
    )
    display_end = pd.Timestamp(
        datetime.combine(
            target_date + timedelta(days=1),
            datetime.min.time(),
            tzinfo=tz,
        )
    )

    window_rows = work[
        (work["time"] >= display_start) & (work["time"] < display_end)
    ].copy()
    if window_rows.empty:
        return None, latest_key

    # Stitched NWS forecast: latest stored forecast available for each valid hour.
    window_rows = window_rows.sort_values(["valid_at", "snapshot_at"])
    forecast = window_rows.groupby("valid_at", as_index=False).tail(1).copy()
    forecast = forecast.sort_values("valid_at")
    forecast["time"] = forecast["valid_at"].dt.tz_convert(tz)
    forecast = forecast[["time", "temp_f", "snapshot_at"]].copy()
    forecast["Series"] = "NWS prediction"

    obs_parts = []
    for frame in (previous_observed_df, observed_df):
        if frame is None or frame.empty:
            continue
        x = frame.copy()
        x["time"] = pd.to_datetime(
            x["time"], utc=True, errors="coerce"
        ).dt.tz_convert(tz)
        x["temp_f"] = pd.to_numeric(x["temp_f"], errors="coerce")
        x = x.dropna(subset=["time", "temp_f"])
        obs_parts.append(x[["time", "temp_f"]])

    observations = (
        pd.concat(obs_parts, ignore_index=True).sort_values("time")
        if obs_parts
        else pd.DataFrame(columns=["time", "temp_f"])
    )
    if not observations.empty:
        observations = observations[
            (observations["time"] >= display_start)
            & (observations["time"] < display_end)
        ].copy()
        observations["Series"] = "Observed"

    now_local = pd.Timestamp.now(tz=tz_name)

    # Keep the visible x domain limited to actual data + a small pad.
    all_times = list(forecast["time"])
    if not observations.empty:
        all_times += list(observations["time"])
    x_min = min(all_times) if all_times else display_start
    x_max = max(all_times) if all_times else display_end
    x_pad = pd.Timedelta(hours=1.5)
    x_min = max(display_start, x_min - x_pad)
    x_max = min(display_end, x_max + x_pad)

    all_temps = list(pd.to_numeric(forecast["temp_f"], errors="coerce").dropna())
    if not observations.empty:
        all_temps += list(pd.to_numeric(observations["temp_f"], errors="coerce").dropna())
    if all_temps:
        y_min = math.floor(min(all_temps) - 2)
        y_max = math.ceil(max(all_temps) + 2)
    else:
        y_min, y_max = 60, 100

    domain = ["Observed", "NWS prediction"]
    rng = ["#FF8FCB", "#B79CFF"]
    color_scale = alt.Scale(domain=domain, range=rng)

    common_x = alt.X(
        "time:T",
        title="Local time",
        axis=alt.Axis(
            format="%b %-d, %-I %p",
            labelAngle=-30,
            tickCount=10,
            grid=True,
            gridOpacity=0.42,
            tickSize=7,
        ),
        scale=alt.Scale(domain=[x_min, x_max]),
    )
    common_y = alt.Y(
        "temp_f:Q",
        title="Temperature (°F)",
        scale=alt.Scale(domain=[y_min, y_max], zero=False),
        axis=alt.Axis(
            tickCount=min(15, max(7, y_max - y_min + 1)),
            grid=True,
            gridOpacity=0.45,
            tickSize=7,
            format=".0f",
        ),
    )

    layers = []

    if not observations.empty:
        layers.append(
            alt.Chart(observations)
            .mark_line(
                point=alt.OverlayMarkDef(filled=True, size=34),
                strokeWidth=3,
            )
            .encode(
                x=common_x,
                y=common_y,
                color=alt.Color(
                    "Series:N",
                    scale=color_scale,
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    alt.Tooltip("time:T", title="Observed time", format="%b %-d, %-I:%M %p"),
                    alt.Tooltip("temp_f:Q", title="Observed", format=".1f"),
                ],
            )
        )

    layers.append(
        alt.Chart(forecast)
        .mark_line(
            point=alt.OverlayMarkDef(filled=True, size=34),
            strokeWidth=3,
            strokeDash=[7, 4],
        )
        .encode(
            x=common_x,
            y=common_y,
            color=alt.Color(
                "Series:N",
                scale=color_scale,
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("time:T", title="Forecast valid time", format="%b %-d, %-I:%M %p"),
                alt.Tooltip("temp_f:Q", title="NWS prediction", format=".1f"),
                alt.Tooltip("snapshot_at:T", title="Stored from snapshot", format="%b %-d, %-I:%M %p"),
            ],
        )
    )

    # Temperature-only labels for four maxima/reference points.
    label_rows = []

    # Current-day observed max.
    if not observations.empty:
        cur_obs = observations[observations["time"].dt.date == target_date]
        if not cur_obs.empty:
            cur_obs_max = float(cur_obs["temp_f"].max())
            p = cur_obs[cur_obs["temp_f"] >= cur_obs_max - 0.05].sort_values("time").iloc[0]
            label_rows.append({
                "time": p["time"],
                "temp_f": cur_obs_max,
                "label": f"{cur_obs_max:.1f}°",
                "kind": "Observed",
                "dy": -12,
            })

    # Current-day NWS predicted max from stitched forecast.
    cur_fc = forecast[forecast["time"].dt.date == target_date]
    if not cur_fc.empty:
        cur_fc_max = float(cur_fc["temp_f"].max())
        p = cur_fc[cur_fc["temp_f"] >= cur_fc_max - 0.05].sort_values("time").iloc[0]
        label_rows.append({
            "time": p["time"],
            "temp_f": cur_fc_max,
            "label": f"{cur_fc_max:.1f}°",
            "kind": "NWS prediction",
            "dy": -12,
        })

    prev_day = target_date - timedelta(days=1)

    # Previous-day observed max.
    if not observations.empty:
        prev_obs = observations[observations["time"].dt.date == prev_day]
        if not prev_obs.empty:
            prev_obs_max = float(prev_obs["temp_f"].max())
            p = prev_obs[prev_obs["temp_f"] >= prev_obs_max - 0.05].sort_values("time").iloc[0]
            label_rows.append({
                "time": p["time"],
                "temp_f": prev_obs_max,
                "label": f"{prev_obs_max:.1f}°",
                "kind": "Previous observed high",
                "dy": -16,
            })

    # Previous-day last valid predicted high. Use only snapshots that still covered
    # the observed previous-day peak so an evening residual cannot masquerade as a
    # daily-high prediction.
    try:
        prev_snap = snapshot_df.copy()
        prev_snap["valid_local"] = prev_snap["valid_at"].dt.tz_convert(tz)
        prev_snap = prev_snap[prev_snap["valid_local"].dt.date == prev_day].copy()

        peak_time = None
        if not observations.empty:
            prev_obs2 = observations[observations["time"].dt.date == prev_day]
            if not prev_obs2.empty:
                high2 = float(prev_obs2["temp_f"].max())
                peak_time = (
                    prev_obs2[prev_obs2["temp_f"] >= high2 - 0.05]
                    .sort_values("time")
                    .iloc[0]["time"]
                )

        if not prev_snap.empty:
            projected_prev = (
                prev_snap.groupby("snapshot_key", as_index=False)
                .agg(
                    projected_high_f=("temp_f", "max"),
                    snapshot_at=("snapshot_at", "min"),
                    valid_min=("valid_local", "min"),
                    valid_max=("valid_local", "max"),
                )
                .sort_values("snapshot_at")
            )

            chosen = None
            if peak_time is not None:
                eligible = projected_prev[
                    projected_prev["snapshot_at"].apply(
                        lambda x: pd.Timestamp(x).tz_convert(tz) <= peak_time
                    )
                    & (projected_prev["valid_min"] <= peak_time)
                    & (projected_prev["valid_max"] >= peak_time)
                ]
                if not eligible.empty:
                    chosen = eligible.iloc[-1]

            if chosen is None and not projected_prev.empty:
                chosen = projected_prev.iloc[0]

            if chosen is not None:
                pred_val = float(chosen["projected_high_f"])
                # Put label at the predicted peak-valid hour from that snapshot,
                # not at the collector timestamp, so it sits on the trajectory.
                chosen_key = chosen["snapshot_key"]
                chosen_rows = prev_snap[prev_snap["snapshot_key"] == chosen_key]
                chosen_peak_rows = chosen_rows[
                    chosen_rows["temp_f"] >= pred_val - 0.05
                ].sort_values("valid_local")
                label_time = (
                    chosen_peak_rows.iloc[0]["valid_local"]
                    if not chosen_peak_rows.empty
                    else pd.Timestamp(chosen["snapshot_at"]).tz_convert(tz)
                )
                label_rows.append({
                    "time": label_time,
                    "temp_f": pred_val,
                    "label": f"{pred_val:.1f}°",
                    "kind": "Previous NWS predicted high",
                    "dy": 16,
                })
    except Exception:
        pass

    if label_rows:
        label_df = pd.DataFrame(label_rows)
        label_scale = alt.Scale(
            domain=[
                "Observed",
                "NWS prediction",
                "Previous observed high",
                "Previous NWS predicted high",
            ],
            range=[
                "#FF8FCB",
                "#B79CFF",
                "#FF6B5E",
                "#36C2FF",
            ],
        )

        layers.append(
            alt.Chart(label_df)
            .mark_point(
                filled=True,
                size=115,
                stroke="#11121B",
                strokeWidth=1.5,
            )
            .encode(
                x=alt.X("time:T"),
                y=alt.Y("temp_f:Q"),
                color=alt.Color("kind:N", scale=label_scale, legend=None),
                tooltip=[
                    alt.Tooltip("kind:N", title="Value"),
                    alt.Tooltip("temp_f:Q", title="Temperature", format=".1f"),
                    alt.Tooltip("time:T", title="Time", format="%b %-d, %-I:%M %p"),
                ],
            )
        )

        # Use dy from data via two layers so labels don't stack directly on peaks.
        up_labels = label_df[label_df["dy"] < 0]
        down_labels = label_df[label_df["dy"] >= 0]

        if not up_labels.empty:
            layers.append(
                alt.Chart(up_labels)
                .mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-8,
                    fontSize=13,
                    fontWeight="bold",
                    stroke="#11121B",
                    strokeWidth=3,
                    color="#FFFFFF",
                )
                .encode(
                    x=alt.X("time:T"),
                    y=alt.Y("temp_f:Q"),
                    text="label:N",
                )
            )

        if not down_labels.empty:
            layers.append(
                alt.Chart(down_labels)
                .mark_text(
                    align="center",
                    baseline="top",
                    dy=8,
                    fontSize=13,
                    fontWeight="bold",
                    stroke="#11121B",
                    strokeWidth=3,
                    color="#FFFFFF",
                )
                .encode(
                    x=alt.X("time:T"),
                    y=alt.Y("temp_f:Q"),
                    text="label:N",
                )
            )

    if x_min <= now_local <= x_max:
        now_df = pd.DataFrame({"time": [now_local]})
        layers.append(
            alt.Chart(now_df)
            .mark_rule(
                color="#8CEAF2",
                strokeWidth=2,
                strokeDash=[3, 3],
            )
            .encode(x="time:T")
        )
        layers.append(
            alt.Chart(now_df)
            .mark_text(
                text="NOW",
                color="#8CEAF2",
                fontSize=12,
                fontWeight="bold",
                align="left",
                baseline="top",
                dx=5,
                dy=5,
            )
            .encode(x="time:T", y=alt.value(5))
        )

    chart = (
        alt.layer(*layers)
        .resolve_scale(color="independent")
        .properties(
            height=360,
            background="#11121B",
            title=alt.TitleParams(
                text="Observed vs NWS prediction",
                subtitle="",
                anchor="start",
            ),
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=15,
            labelColor="#F1EDF7",
            titleColor="#FAF7FF",
            gridColor="#777185",
            gridOpacity=.42,
            domainColor="#AAA3B5",
            tickColor="#AAA3B5",
        )
        .configure_legend(
            orient="top",
            direction="horizontal",
            title=None,
            labelFontSize=14,
            labelColor="#F1EDF7",
            symbolSize=150,
        )
        .configure_title(
            fontSize=18,
            subtitleFontSize=13,
            color="#FAF7FF",
            subtitleColor="#D4CEDD",
        )
        .configure_view(strokeWidth=0)
    )
    return chart, latest_key


def max_projection_history_chart(
    snapshot_df,
    tz_name,
    target_date,
    observed_df=None,
):
    """
    One point per collector run: that run's predicted high for target_date.

    Only forecast-valid hours on the selected calendar day are used. For a
    completed day, snapshots must also cover the actual observed peak window,
    preventing late-evening residual forecasts from appearing as fake low daily
    highs.
    """
    if snapshot_df.empty:
        return None, pd.DataFrame()

    tz = ZoneInfo(tz_name)
    work = snapshot_df.copy()
    work["valid_local"] = work["valid_at"].dt.tz_convert(tz)
    work["snapshot_local"] = work["snapshot_at"].dt.tz_convert(tz)
    work = work[work["valid_local"].dt.date == target_date].copy()
    if work.empty:
        return None, pd.DataFrame()

    history = (
        work.groupby("snapshot_key", as_index=False)
        .agg(
            predicted_high_f=("temp_f", "max"),
            snapshot_time=("snapshot_at", "min"),
            valid_min=("valid_local", "min"),
            valid_max=("valid_local", "max"),
        )
        .sort_values("snapshot_time")
    )

    # When an actual daily peak is available, a true full-day-high forecast
    # snapshot must both precede and cover that peak time.
    peak_dt = None
    try:
        if observed_df is not None and not observed_df.empty:
            obs = observed_df.copy()
            obs["time_local"] = pd.to_datetime(
                obs["time"], utc=True, errors="coerce"
            ).dt.tz_convert(tz)
            obs["temp_f"] = pd.to_numeric(obs["temp_f"], errors="coerce")
            obs = obs.dropna(subset=["time_local", "temp_f"]).sort_values("time_local")
            obs = obs[obs["time_local"].dt.date == target_date]
            if not obs.empty:
                high = float(obs["temp_f"].max())
                peak_rows = obs[obs["temp_f"] >= high - 0.15]
                if not peak_rows.empty:
                    peak_dt = pd.Timestamp(peak_rows.iloc[0]["time_local"])
    except Exception:
        peak_dt = None

    now_local = pd.Timestamp.now(tz=tz_name)
    # Apply peak coverage only to a completed day or once we're safely past the
    # observed peak by at least two hours. Before then, keep today's evolving
    # forecasts visible.
    if peak_dt is not None and (
        target_date < now_local.date()
        or now_local >= peak_dt + pd.Timedelta(hours=2)
    ):
        eligible = history[
            history["snapshot_time"].apply(
                lambda x: pd.Timestamp(x).tz_convert(tz) <= peak_dt
            )
            & (history["valid_min"] <= peak_dt)
            & (history["valid_max"] >= peak_dt)
        ].copy()
        if not eligible.empty:
            history = eligible

    history["snapshot_time"] = pd.to_datetime(
        history["snapshot_time"], utc=True, errors="coerce"
    ).dt.tz_convert(tz)
    history = history.dropna(subset=["snapshot_time", "predicted_high_f"])
    if history.empty:
        return None, history

    chart = (
        alt.Chart(history)
        .mark_line(
            point=alt.OverlayMarkDef(filled=True, size=58),
            strokeWidth=3,
            color="#B79CFF",
        )
        .encode(
            x=alt.X(
                "snapshot_time:T",
                title="Forecast snapshot time",
                axis=alt.Axis(
                    format="%b %-d, %-I:%M %p",
                    labelAngle=-30,
                    tickCount=7,
                    grid=True,
                    gridOpacity=0.42,
                    tickSize=7,
                ),
            ),
            y=alt.Y(
                "predicted_high_f:Q",
                title="Predicted daily high (°F)",
                scale=alt.Scale(zero=False, padding=12),
                axis=alt.Axis(tickCount=8, format=".1f"),
            ),
            tooltip=[
                alt.Tooltip(
                    "snapshot_time:T",
                    title="Snapshot",
                    format="%b %-d, %-I:%M %p",
                ),
                alt.Tooltip(
                    "predicted_high_f:Q",
                    title="Predicted high",
                    format=".1f",
                ),
            ],
        )
        .properties(
            height=260,
            background="#11121B",
            title=alt.TitleParams(
                text="How the predicted high has changed",
                subtitle=(
                    "Each point is that snapshot's predicted high for the selected day"
                ),
                anchor="start",
            ),
        )
        .configure_axis(
            labelFontSize=14,
            titleFontSize=15,
            labelColor="#DED9EA",
            titleColor="#F5F1FA",
            gridColor="#5D5870",
            gridOpacity=0.32,
            domainColor="#6E687E",
            tickColor="#6E687E",
        )
        .configure_title(
            fontSize=18,
            subtitleFontSize=14,
            color="#FAF7FF",
            subtitleColor="#C9C3D5",
        )
        .configure_view(strokeWidth=0)
    )
    return chart, history


def render_bet_forecast(city, contract_date):
    """Render snapshot-driven weather context directly inside a selected bet."""
    cfg = PRESETS[city]
    rows, snapshot_error = get_snapshot_rows(city, contract_date)
    if snapshot_error:
        st.info(snapshot_error)
        return
    if not rows:
        st.info(f"No stored forecast snapshots yet for {city} on {contract_date:%b %-d}.")
        return

    snapshot_df, normalize_error = normalize_snapshot_rows(rows)
    if normalize_error:
        st.warning(normalize_error)
        return
    if snapshot_df.empty:
        st.info("Snapshot rows were found, but none had usable timestamps and temperatures.")
        return

    observed = get_station_observations(cfg.get("station_id"), cfg["tz"], contract_date)
    previous_observed = get_station_observations(
        cfg.get("station_id"), cfg["tz"], contract_date - timedelta(days=1)
    )
    latest_chart, latest_key = latest_projection_chart(
        snapshot_df, observed, previous_observed, cfg["tz"], contract_date
    )
    history_chart, history = max_projection_history_chart(
        snapshot_df,
        cfg["tz"],
        contract_date,
        observed_df=observed,
    )

    latest_high = None
    if latest_key is not None:
        latest_slice = snapshot_df[snapshot_df["snapshot_key"] == latest_key]
        if not latest_slice.empty:
            latest_high = latest_slice["temp_f"].max()
    observed_high = None if observed.empty else observed["temp_f"].max()

    st.markdown("<div class='section-kicker'>WEATHER FIGURES</div>", unsafe_allow_html=True)
    st.caption("Forecast range, observed-vs-predicted trajectory, and forecast history are grouped together below.")

    if latest_chart is not None:
        st.altair_chart(latest_chart, use_container_width=True)
        if latest_key is not None:
            latest_local = latest_key.tz_convert(ZoneInfo(cfg["tz"]))
            st.caption(f"Latest stored projection: {latest_local:%b %-d at %-I:%M %p %Z}.")

    if history_chart is not None:
        st.altair_chart(history_chart, use_container_width=True)
    else:
        st.info("Forecast-history will appear as additional snapshots are collected.")

    st.metric("Stored snapshots", f"{history.shape[0]:,}")




st.set_page_config(page_title="WeatherEdge", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --space:#030817;
  --space2:#071129;
  --panel:rgba(6,16,39,.84);
  --panel2:rgba(13,20,51,.83);
  --ink:#F8FAFF;
  --muted:#A8B5D7;
  --violet:#9A67FF;
  --magenta:#F24BC7;
  --cyan:#22D6FF;
  --blue:#407BFF;
  --sun:#FFC247;
  --storm:#6FA7FF;
  --good:#29E6A6;
  --danger:#FF607D;
  --border:rgba(108,142,255,.30);
  --border-hot:rgba(230,78,225,.52);
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  color:var(--ink);
  background:
    radial-gradient(circle at 12% 0%, rgba(145,67,255,.18), transparent 25rem),
    radial-gradient(circle at 90% 8%, rgba(0,200,255,.14), transparent 28rem),
    radial-gradient(circle at 55% 55%, rgba(242,75,199,.05), transparent 30rem),
    linear-gradient(165deg, #020610 0%, #061028 42%, #030817 100%);
  font-size:16px;
  line-height:1.55;
}
.stApp:before {
  content:"";
  position:fixed;
  inset:0;
  pointer-events:none;
  opacity:.42;
  z-index:0;
  background-image:
    radial-gradient(circle at 18% 20%, rgba(255,255,255,.85) 0 1px, transparent 1.6px),
    radial-gradient(circle at 82% 36%, rgba(110,194,255,.9) 0 1px, transparent 1.6px),
    radial-gradient(circle at 64% 8%, rgba(242,75,199,.9) 0 1px, transparent 1.6px);
  background-size:180px 180px, 240px 240px, 300px 300px;
}
[data-testid="stAppViewContainer"] > .main {position:relative; z-index:1;}
.block-container {
  max-width:1180px;
  padding-top:.7rem;
  padding-bottom:4rem;
}
[data-testid="stHeader"] {
  background:rgba(2,6,16,.72)!important;
  border-bottom:1px solid rgba(90,126,232,.18);
  backdrop-filter:blur(16px);
}
[data-testid="stToolbar"] {right:.75rem;}

h1,h2,h3,h4,h5,h6 {
  color:#FFFFFF!important;
  font-weight:780!important;
  letter-spacing:-.02em!important;
}
h1 {font-size:2.65rem!important;}
h2 {font-size:1.75rem!important;}
h3 {font-size:1.28rem!important;}
p, li, span, label, [data-testid="stMarkdownContainer"] {
  color:#E9EEFF;
}
small, .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color:#9EADD0!important;
  font-size:.89rem!important;
}
a {color:#6DE8FF!important;}

/* orbital hero */
.we-hero {
  position:relative;
  min-height:215px;
  overflow:hidden;
  border:1px solid rgba(83,132,255,.36);
  border-radius:22px;
  padding:1.35rem 1.55rem 1.25rem;
  margin:.15rem 0 1rem;
  background:
    linear-gradient(90deg, rgba(2,7,24,.96) 0%, rgba(2,7,24,.73) 46%, rgba(2,7,24,.26) 100%),
    url("data:image/webp;base64,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") center 42% / cover no-repeat;
  box-shadow:0 18px 60px rgba(0,0,0,.38), inset 0 0 55px rgba(47,103,255,.06);
}
.we-hero:after {
  content:"";
  position:absolute;
  left:0; right:0; bottom:0; height:3px;
  background:linear-gradient(90deg, #7F57FF, #F24BC7 48%, #22D6FF);
  opacity:.85;
}
.we-orbit {
  position:absolute; right:3.2rem; top:1.1rem;
  width:150px; height:74px;
  border:1px solid rgba(99,232,255,.22);
  border-radius:50%;
  transform:rotate(-12deg);
}
.we-sat {
  position:absolute; right:5.25rem; top:2.0rem;
  font-size:2.2rem;
  filter:drop-shadow(0 0 12px rgba(82,213,255,.8));
}
.we-brand {
  font-size:2.1rem;
  font-weight:900;
  letter-spacing:-.045em;
  color:#fff;
  text-shadow:0 0 22px rgba(154,103,255,.45);
}
.we-brand span {
  color:#F45BD4;
  font-size:.72rem;
  font-weight:780;
  margin-left:.55rem;
  letter-spacing:.04em;
}
.we-tag {
  margin-top:.25rem;
  color:#D955D4;
  font-size:.83rem;
  letter-spacing:.095em;
  text-transform:uppercase;
  font-weight:790;
}
.we-sub {
  max-width:460px;
  color:#B8C6E8;
  font-size:.94rem;
  margin-top:.5rem;
}
.we-status {
  display:flex;
  flex-wrap:wrap;
  gap:.5rem;
  margin-top:1rem;
}
.we-chip {
  display:inline-flex;
  align-items:center;
  gap:.4rem;
  padding:.38rem .63rem;
  border:1px solid rgba(88,126,220,.35);
  border-radius:999px;
  background:rgba(3,10,28,.66);
  color:#CFD8F7;
  font-size:.73rem;
  font-weight:700;
  backdrop-filter:blur(8px);
}
.we-live-dot {
  width:7px;height:7px;border-radius:50%;background:#29E6A6;
  box-shadow:0 0 9px #29E6A6;
}
.we-weather-strip {
  display:flex; align-items:center; gap:.65rem;
  font-size:1.25rem; margin-top:.85rem;
  opacity:.92;
}
.we-weather-strip span {
  min-width:34px; text-align:center;
  filter:drop-shadow(0 0 8px rgba(115,186,255,.35));
}
.we-weather-strip .palm {filter:drop-shadow(0 0 8px rgba(242,75,199,.4));}

/* section language */
.section-kicker {
  color:#BCA7FF!important;
  font-size:.72rem;
  letter-spacing:.18em;
  text-transform:uppercase;
  font-weight:900;
  margin:1.2rem 0 .45rem;
}
.section-kicker:before {
  content:"✦";
  color:#F24BC7;
  margin-right:.45rem;
}
.card-title {font-size:1.38rem;font-weight:850;color:white;}
.card-sub {font-size:.93rem;color:#AEBBDD;line-height:1.5;}

/* universal glass cards */
[data-testid="stMetric"],
[data-testid="stExpander"],
.bet-callout,
.quality-card,
.signal-tile {
  background:
    linear-gradient(145deg, rgba(18,28,64,.88), rgba(5,13,33,.88))!important;
  border:1px solid var(--border)!important;
  box-shadow:0 10px 32px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.025)!important;
  backdrop-filter:blur(14px);
}
[data-testid="stMetric"] {
  border-radius:16px!important;
  padding:14px 15px!important;
}
[data-testid="stMetric"]:hover {
  border-color:rgba(83,208,255,.55)!important;
  transform:translateY(-1px);
}
[data-testid="stMetricLabel"] p {
  color:#9DADD2!important;
  text-transform:uppercase;
  letter-spacing:.075em;
  font-size:.72rem!important;
  font-weight:800!important;
}
[data-testid="stMetricValue"] {
  color:#FFFFFF!important;
  font-weight:850!important;
  font-size:1.62rem!important;
}
[data-testid="stMetricDelta"] {color:#49E2B0!important;}

.bet-callout {
  border-radius:18px;
  padding:1rem 1.05rem;
  border-color:rgba(242,75,199,.42)!important;
  background:
    radial-gradient(circle at 88% 12%, rgba(242,75,199,.16), transparent 9rem),
    linear-gradient(145deg, rgba(24,21,66,.93), rgba(5,16,39,.91))!important;
}
.bet-callout-label,.quality-label,.signal-label {
  color:#B69DFF!important;
  text-transform:uppercase;
  letter-spacing:.13em;
  font-weight:900;
  font-size:.68rem;
}
.bet-callout-main {font-size:1.22rem;color:#fff;font-weight:850;}
.bet-callout-sub,.quality-sub,.signal-sub {color:#AAB7D9!important;}
.quality-card {
  border-color:rgba(34,214,255,.38)!important;
  border-radius:18px;
  padding:1rem 1.05rem;
  background:
    radial-gradient(circle at 82% 20%, rgba(34,214,255,.12), transparent 9rem),
    linear-gradient(145deg, rgba(12,30,61,.92), rgba(15,13,49,.92))!important;
}
.quality-top {display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;}
.quality-value {
  font-size:2.25rem;
  color:#F85BD6!important;
  text-shadow:0 0 18px rgba(248,91,214,.28);
  font-weight:900;
}
.quality-grade {color:#69E9FF!important;font-weight:800;}
.signal-strip {display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin:.7rem 0 .9rem;}
.signal-tile {padding:.78rem .86rem;border-radius:14px;}
.signal-value {font-size:1.22rem;font-weight:850;color:#fff;}

/* Sidebar = mission control */
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 10% 80%, rgba(242,75,199,.17), transparent 19rem),
    linear-gradient(180deg, #060B1D 0%, #09112B 55%, #050914 100%)!important;
  border-right:1px solid rgba(104,129,255,.24)!important;
}
[data-testid="stSidebar"] > div {background:transparent!important;}
[data-testid="stSidebar"] * {color:#EEF3FF!important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  color:#F25BD2!important;
  text-transform:uppercase;
  letter-spacing:.08em!important;
  font-size:.95rem!important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  color:#92A3CB!important;
}
.sidebar-brand {
  margin:.15rem 0 1rem;
  padding:.9rem .8rem;
  border:1px solid rgba(127,87,255,.38);
  border-radius:15px;
  background:linear-gradient(135deg,rgba(114,61,222,.18),rgba(8,18,42,.70));
}
.sidebar-brand .orb {font-size:1.5rem;}
.sidebar-brand strong {font-size:1.02rem;letter-spacing:.03em;}
.sidebar-brand small {
  display:block;
  margin-top:.18rem;
  color:#9CB0D8!important;
  font-size:.72rem!important;
  letter-spacing:.07em;
  text-transform:uppercase;
}

/* controls */
.stButton button, [data-testid="stLinkButton"] a {
  color:#F8FBFF!important;
  background:linear-gradient(135deg,rgba(42,28,86,.92),rgba(10,27,55,.94))!important;
  border:1px solid rgba(112,131,242,.46)!important;
  border-radius:12px!important;
  font-weight:780!important;
  min-height:2.7rem;
  box-shadow:0 5px 18px rgba(0,0,0,.18)!important;
}
.stButton button:hover, [data-testid="stLinkButton"] a:hover {
  border-color:#D350D3!important;
  box-shadow:0 0 18px rgba(211,80,211,.20)!important;
  transform:translateY(-1px);
}
[data-testid="stRadio"] > div {gap:.45rem;}
[data-testid="stRadio"] label {
  background:rgba(8,17,42,.86);
  border:1px solid rgba(92,118,197,.32);
  border-radius:12px;
  padding:.5rem .78rem;
}
[data-testid="stRadio"] label:hover {
  border-color:#D354D5;
  background:rgba(40,20,72,.88);
}
[data-testid="stRadio"] label p {font-weight:760!important;}
[data-testid="stSlider"] div[role="slider"] {
  background:#F35BCD!important;
  box-shadow:0 0 12px rgba(243,91,205,.55)!important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
  background-color:#23304E!important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
  background:linear-gradient(90deg,#7F57FF,#F35BCD,#22D6FF)!important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background:#081229!important;
  color:#fff!important;
  border-color:rgba(94,123,213,.35)!important;
}
[data-baseweb="menu"] {background:#071025!important;}
[data-baseweb="menu"] li {color:#fff!important;}

/* expanders / alerts / dataframe */
[data-testid="stExpander"] {
  border-radius:15px!important;
}
[data-testid="stExpander"] summary {
  background:rgba(10,19,45,.90)!important;
  color:#fff!important;
  border-radius:14px!important;
}
[data-testid="stExpander"] summary:hover {
  background:rgba(34,25,67,.92)!important;
}
[data-testid="stAlert"] {
  background:rgba(11,22,48,.91)!important;
  border:1px solid rgba(90,126,224,.34)!important;
  border-radius:14px!important;
}
[data-testid="stDataFrame"] {
  border:1px solid rgba(91,120,208,.30)!important;
  border-radius:14px!important;
  overflow:hidden;
}

/* chart frames */
[data-testid="stVegaLiteChart"] {
  background:linear-gradient(145deg,rgba(3,9,25,.96),rgba(6,13,32,.94));
  border:1px solid rgba(82,115,204,.27);
  border-radius:16px;
  padding:.25rem;
  box-shadow:0 10px 28px rgba(0,0,0,.20);
  overflow:hidden;
}

/* tiny satellite/weather ornamental rail */
.we-orbit-rail {
  display:flex;
  gap:.55rem;
  align-items:center;
  justify-content:space-between;
  padding:.72rem .9rem;
  margin:.55rem 0 1rem;
  border:1px solid rgba(89,119,214,.25);
  border-radius:14px;
  background:rgba(4,12,31,.64);
}
.we-orbit-copy {
  font-size:.75rem;
  color:#94A7D4;
  letter-spacing:.08em;
  text-transform:uppercase;
  font-weight:750;
}
.we-orbit-icons {display:flex;gap:.6rem;font-size:1.22rem;}
.we-orbit-icons span {filter:drop-shadow(0 0 7px rgba(77,182,255,.35));}

code {color:#65F0C5!important;background:#061B1A!important;}
hr {border-color:rgba(83,115,204,.22)!important;}

@media (max-width:760px) {
  .block-container {padding-left:.82rem;padding-right:.82rem;padding-top:.45rem;}
  .we-hero {min-height:190px;padding:1.05rem 1rem;background-position:56% 45%;}
  .we-brand {font-size:1.7rem;}
  .we-sat {right:1.55rem;top:1.25rem;font-size:1.75rem;}
  .we-orbit {right:.65rem;top:.5rem;width:120px;height:60px;}
  .we-sub {max-width:72%;font-size:.84rem;}
  .we-weather-strip {font-size:1.1rem;gap:.38rem;}
  .we-status {max-width:73%;}
  .we-chip {font-size:.65rem;padding:.28rem .48rem;}
  .signal-strip {grid-template-columns:1fr 1fr;gap:.45rem;}
  .signal-tile {padding:.64rem .68rem;}
  [data-testid="stMetricValue"] {font-size:1.42rem!important;}
  h1 {font-size:2rem!important;}
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="we-hero">
      <div class="we-orbit"></div>
      <div class="we-sat">🛰️</div>
      <div class="we-brand">WEATHEREDGE <span>ORBITAL</span></div>
      <div class="we-tag">Real-time weather intelligence · probability from the edge of space</div>
      <div class="we-sub">
        Live weather markets, calibrated forecasts, and satellite-minded probability modeling
        in one streamlined mission-control view.
      </div>
      <div class="we-status">
        <div class="we-chip"><span class="we-live-dot"></span> SYSTEM LIVE</div>
        <div class="we-chip">🛰️ SATELLITE WEATHER</div>
        <div class="we-chip">🌡️ NWS + OBSERVATIONS</div>
        <div class="we-chip">◈ PROBABILITY ENGINE</div>
      </div>
      <div class="we-weather-strip">
        <span>☀️</span><span>⛅</span><span>🌧️</span><span>⚡</span>
        <span>❄️</span><span>🌬️</span><span class="palm">🌴</span>
      </div>
    </div>
    <div class="we-orbit-rail">
      <div class="we-orbit-copy">Weather on Earth · signals from orbit · markets in motion</div>
      <div class="we-orbit-icons"><span>🛰️</span><span>◌</span><span>🌎</span><span>✦</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation state is initialized before the sidebar so Bet Settings can appear
# only while Best Bets is active.
if "main_view" not in st.session_state:
    st.session_state.main_view = "Best Bets"
if "explorer_city" not in st.session_state:
    st.session_state.explorer_city = list(PRESETS.keys())[0]
if "explorer_contract" not in st.session_state:
    st.session_state.explorer_contract = None
if "watched_bets" not in st.session_state:
    st.session_state.watched_bets = []
# Placeholder for future read-only Kalshi account integration.
if "placed_bets" not in st.session_state:
    st.session_state.placed_bets = []
if "top_n_setting" not in st.session_state:
    st.session_state.top_n_setting = 5
if "min_nws_setting" not in st.session_state:
    st.session_state.min_nws_setting = 70
if "min_gap_setting" not in st.session_state:
    st.session_state.min_gap_setting = 5
if "min_market_price_setting" not in st.session_state:
    st.session_state.min_market_price_setting = 5

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div><span class="orb">🪐</span> <strong>WEATHEREDGE</strong></div>
          <small>Orbital market intelligence · mission control</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Top Bet Settings")
    st.caption(
        "Adjust the safety cutoffs and how many ranked Top Bets WeatherEdge shows. "
        "Settlement-source rounding protection is fixed and always on."
    )

    with st.expander("Adjust Top Bet filters", expanded=False):
        top_n = st.slider(
            "Number of Top Bets",
            min_value=3,
            max_value=20,
            value=int(st.session_state.top_n_setting),
            step=1,
            key="top_n_setting",
            help="Maximum number of qualifying bets shown in the Top Bets ranking.",
        )
        min_nws_pct = st.slider(
            "Minimum NWS-based chance",
            min_value=55,
            max_value=95,
            value=int(st.session_state.min_nws_setting),
            step=1,
            key="min_nws_setting",
            help=(
                "Hard safety cutoff. Contracts below this WeatherEdge/NWS-based "
                "win probability are excluded from Top Bets."
            ),
        )
        min_gap_pct = st.slider(
            "Minimum Weather Edge",
            min_value=0,
            max_value=30,
            value=int(st.session_state.min_gap_setting),
            step=1,
            key="min_gap_setting",
            help="Minimum NWS-based chance minus the live Kalshi ask, in percentage points.",
        )
        min_market_price_pct = st.slider(
            "Minimum bid price / market probability",
            min_value=1,
            max_value=100,
            value=int(st.session_state.min_market_price_setting),
            step=1,
            key="min_market_price_setting",
            help=(
                "Exclude Top Bets when the side's live Kalshi ask is at or below "
                "this percentage. 95 means roughly a 95¢ contract / 95% market-implied "
                "probability. Higher values are more conservative about which market "
                "outcomes are even eligible for Top Bets."
            ),
        )


    st.caption(
        "Live ranking data is cached for 5 minutes, with a short-lived persistent "
        "startup cache for faster first render. Chart-only GFS/history loads only "
        "when City Explorer is opened. Changing settings reranks cached markets immediately. "
        "Settlement-source rounding protection remains fixed. The minimum Kalshi "
        "market probability is adjustable from 1% to 100%; increasing it filters "
        "out more low-priced/outlier contracts. Top Bets also require enough historical "
        "calibration, a positive edge after an uncertainty haircut, no strong "
        "near-peak live conflict, and no major unresolved model disagreement."
    )

min_nws_chance = min_nws_pct / 100
min_gap = min_gap_pct / 100
min_market_price = min_market_price_pct / 100

# ------------------------------------------------------------------
# Live-data refresh layer
# ------------------------------------------------------------------
# Ordinary UI reruns reuse this cached universe for five minutes.
# Only the explicit refresh button forces all cities to be scanned again.
refresh_col, status_col = st.columns([0.42, 0.58])

with refresh_col:
    if st.button(
        "↻ Refresh live data",
        use_container_width=True,
        help="Force a fresh Kalshi + current weather scan. Chart-only GFS/history data is not refreshed here.",
    ):
        st.session_state.force_live_refresh = True
        clear_live_data_caches()
        st.rerun()

force_live_refresh = bool(
    st.session_state.pop("force_live_refresh", False)
)
startup_payload = None if force_live_refresh else load_persistent_scan()

if startup_payload is not None:
    all_rows, errors, refreshed_at = startup_payload
    startup_cache_used = True
else:
    with st.spinner("Loading live market essentials…"):
        all_rows, errors, refreshed_at = scan_live_market_universe()
    startup_cache_used = False

with status_col:
    try:
        refreshed_ts = pd.Timestamp(refreshed_at)
        age_seconds = max(
            0,
            (pd.Timestamp.now(tz="UTC") - refreshed_ts).total_seconds(),
        )
        if age_seconds < 60:
            age_text = f"{int(age_seconds)}s ago"
        else:
            age_text = f"{int(age_seconds // 60)}m ago"
        if startup_cache_used:
            st.caption(f"Instant startup cache · refreshed {age_text}")
        else:
            st.caption(f"Using cached live scan · refreshed {age_text}")
    except Exception:
        st.caption("Using cached live scan")

if errors:
    with st.expander(f"Scan notes ({len(errors)})", expanded=False):
        for message in errors:
            st.caption(message)

if not all_rows:
    st.warning("No matching open weather markets were found.")
    st.stop()

df = pd.DataFrame(all_rows)

# Normalize ranking-critical numeric fields. Cached payloads and some API values
# can deserialize as object/string dtype, which must not reach comparisons/abs().
numeric_ranking_columns = [
    "conservative_edge",
    "conservative_prob",
    "ask",
    "nws_sigma_f",
    "nws_sigma_samples",
    "trajectory_current_gap_f",
    "hours_to_expected_peak",
    "nws_high_f",
    "adjusted_nws_center_f",
    "expected_remaining_heating_f",
    "final_high_sigma_down_f",
    "final_high_sigma_up_f",
    "final_high_upside_p95_f",
    "ensemble_median_f",
    "kalshi_implied_temp_f",
    "bet_quality_score",
    "opportunity_score",
    "volume",
]
for col in numeric_ranking_columns:
    if col not in df.columns:
        df[col] = pd.NA
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "settlement_rounding_risk" not in df.columns:
    df["settlement_rounding_risk"] = False

df = df[df["conservative_edge"].notna()].copy()



def conservative_probability_lower_bound(row):
    """
    Approximate a safety lower bound by shrinking probability toward 50%
    according to forecast uncertainty and calibration depth.
    """
    def safe_float(value, default):
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    p = safe_float(row.get("conservative_prob"), 0.5)
    sigma = safe_float(row.get("nws_sigma_f"), 2.0)
    n = int(safe_float(row.get("nws_sigma_samples"), 0))

    uncertainty_penalty = min(0.18, 0.025 * max(0.0, sigma - 1.0))
    sample_penalty = 0.06 if n < 8 else (0.03 if n < 16 else 0.0)
    penalty = uncertainty_penalty + sample_penalty

    if p >= 0.5:
        return max(0.5, p - penalty)
    return min(0.5, p + penalty)


def model_disagreement_f(row):
    vals = []
    for key in ("nws_high_f", "adjusted_nws_center_f", "ensemble_median_f", "kalshi_implied_temp_f"):
        v = row.get(key)
        if v is not None and not pd.isna(v):
            try:
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
            except (TypeError, ValueError):
                pass
    if len(vals) < 2:
        return 0.0
    return max(vals) - min(vals)


df["safety_prob_lower"] = df.apply(conservative_probability_lower_bound, axis=1)
df["safety_edge_lower"] = df["safety_prob_lower"] - df["ask"]
df["model_disagreement_f"] = df.apply(model_disagreement_f, axis=1)

rounding_safe_mask = ~df["settlement_rounding_risk"].fillna(False).astype(bool)

# User-adjustable outlier-safety rule: exclude sides whose live Kalshi ask
# is at or below the selected minimum market probability.
kalshi_market_floor_mask = (
    df["ask"].notna()
    & (df["ask"] >= min_market_price)
)

historical_depth_mask = (
    pd.to_numeric(df["nws_sigma_samples"], errors="coerce")
    .fillna(0)
    .astype(int)
    >= 8
)

trajectory_gap_numeric = pd.to_numeric(
    df["trajectory_current_gap_f"], errors="coerce"
)
hours_to_peak_numeric = pd.to_numeric(
    df["hours_to_expected_peak"], errors="coerce"
).fillna(99.0)

trajectory_conflict_mask = ~(
    trajectory_gap_numeric.notna()
    & (hours_to_peak_numeric <= 3.0)
    & (trajectory_gap_numeric.abs() >= 1.5)
)

model_agreement_mask = df["model_disagreement_f"].fillna(0.0) <= 4.0

qualified_all = df[
    (df["nws_support"] == True)
    & (df["conservative_prob"] >= min_nws_chance)
    & (df["conservative_edge"] >= min_gap)
    & (df["safety_edge_lower"] >= max(min_gap, 0.03))
    & historical_depth_mask
    & trajectory_conflict_mask
    & model_agreement_mask
    & rounding_safe_mask
    & kalshi_market_floor_mask
].copy()

qualified_all = qualified_all.sort_values(
    ["bet_quality_score", "conservative_edge", "opportunity_score", "volume"],
    ascending=[False, False, False, False],
)
qualified = qualified_all.head(top_n).copy()


def contract_status(row):
    """Explain whether a contract is a current Top Bet."""
    ticker = row["market_ticker"]
    side = row["side"]

    best_match = qualified[
        (qualified["market_ticker"] == ticker) & (qualified["side"] == side)
    ]
    if not best_match.empty:
        rank = list(qualified.index).index(best_match.index[0]) + 1
        return f"TOP BET #{rank}", "best"

    reasons = []
    if not bool(row.get("nws_support", False)):
        reasons.append("NWS does not support this side")
    if float(row.get("conservative_prob", 0.0)) < min_nws_chance:
        reasons.append(f"NWS chance below {min_nws_chance*100:.0f}% cutoff")
    if float(row.get("conservative_edge", -1.0)) < min_gap:
        reasons.append(f"Weather Edge below {min_gap*100:.0f} pp")
    if float(row.get("ask", 0.0) or 0.0) < min_market_price:
        reasons.append(
            f"Kalshi market probability is below "
            f"{min_market_price*100:.0f}% minimum"
        )
    if int(row.get("nws_sigma_samples", 0) or 0) < 8:
        reasons.append("not enough historical calibration samples")
    if float(row.get("safety_edge_lower", -1.0) or -1.0) < max(min_gap, 0.03):
        reasons.append("edge is not strong enough after uncertainty haircut")
    if (
        row.get("trajectory_current_gap_f") is not None
        and not pd.isna(row.get("trajectory_current_gap_f"))
        and float(row.get("hours_to_expected_peak", 99) or 99) <= 3
        and abs(float(row.get("trajectory_current_gap_f"))) >= 1.5
    ):
        reasons.append("live observations conflict with the forecast near peak")
    if float(row.get("model_disagreement_f", 0.0) or 0.0) > 4.0:
        reasons.append("major unresolved model disagreement")
    if bool(row.get("settlement_rounding_risk", False)):
        ambiguous = row.get("settlement_ambiguous_degrees") or []
        degree_text = "/".join(f"{int(v)}°" for v in ambiguous)
        reasons.append(
            f"settlement-source / rounding ambiguity around {degree_text}"
        )

    return (
        "NOT CURRENTLY A TOP BET"
        + (f" · {' · '.join(reasons)}" if reasons else ""),
        "other",
    )


def bet_explanation_blurb(r):
    """Return a compact data-driven explanation of the opportunity and its main risk."""
    try:
        side = str(r.get("side", "BET"))
        prob = float(r.get("conservative_prob", 0.0))
        ask = float(r.get("ask", 0.0))
        edge_pp = float(r.get("conservative_edge", 0.0)) * 100.0
        quality = float(r.get("bet_quality_score", 0.0))
        hours_peak = r.get("hours_to_expected_peak")
        traj_gap = r.get("trajectory_current_gap_f")
        obs = r.get("observed_high_f")
        adjusted = r.get("adjusted_nws_center_f")
        sigma_source = r.get("nws_sigma_source")
        sigma_n = int(r.get("nws_sigma_samples", 0) or 0)
        evolution_adj = r.get("forecast_evolution_adjustment_f")
        evolution_n = int(r.get("forecast_evolution_matches", 0) or 0)

        strengths = [
            f"WeatherEdge estimates {prob*100:.0f}% for {side} versus a {ask*100:.0f}¢ ask, a {edge_pp:+.1f} pp edge"
        ]
        if quality >= 75:
            strengths.append(f"quality is {quality:.0f}/100")
        if evolution_n >= 6 and evolution_adj is not None and not pd.isna(evolution_adj):
            strengths.append(
                f"forecast-evolution history contributes a {float(evolution_adj):+.1f}°F adjustment "
                f"from {evolution_n} comparable historical snapshots"
            )
        if hours_peak is not None and not pd.isna(hours_peak):
            hp = float(hours_peak)
            if hp <= 0:
                strengths.append("the expected hottest point has already passed")
            elif hp <= 2:
                strengths.append("the expected hottest point is very close")
        if obs is not None and not pd.isna(obs) and adjusted is not None and not pd.isna(adjusted):
            if abs(float(adjusted) - float(obs)) <= 0.35:
                strengths.append("the observed high is already pinning the model's downside floor")

        risks = []
        if traj_gap is not None and not pd.isna(traj_gap):
            gap = float(traj_gap)
            if gap <= -1.0:
                risks.append(f"observations are running {abs(gap):.1f}°F below the stored NWS trajectory")
            elif gap >= 1.0:
                risks.append(f"observations are running {gap:.1f}°F above the stored NWS trajectory")
        if hours_peak is not None and not pd.isna(hours_peak) and float(hours_peak) > 3:
            risks.append(f"about {float(hours_peak):.1f} hours remain until the expected daily peak")
        if sigma_source != "historical" or sigma_n < 8:
            risks.append("historical forecast-error calibration is still limited")
        if not risks:
            risks.append("forecast error and a late temperature move can still change the outcome")

        why = "; ".join(strengths[:3]) + "."
        risk = "; ".join(risks[:2]) + "."
        return why, risk
    except Exception:
        return (
            "WeatherEdge sees a favorable probability-versus-price gap for this contract.",
            "Forecast error and market movement remain the main risks.",
        )


def render_contract_detail(r, show_weather=True):
    """One consistent contract detail view used by Best Bets, Suggested Bets and City Explorer."""
    status_text, status_kind = contract_status(r)
    if status_kind == "best":
        st.success(status_text)
    elif status_kind == "suggested":
        st.info(status_text)
    else:
        st.warning(status_text)

    if bool(r.get("daily_high_locked", False)):
        locked_obs = r.get("observed_high_f")
        st.success(
            "Daily high locked: the expected hottest part of the day is clearly past "
            "and the recent temperature trajectory is flat/falling. WeatherEdge is "
            f"therefore treating {float(locked_obs):.1f}°F as the final daily high and "
            "assigning no probability to a higher temperature."
        )

    if float(r.get("ask", 0.0) or 0.0) <= 0.05:
        st.warning(
            "Excluded from Top Bets for outlier safety: Kalshi is pricing this side "
            "at 5% implied probability or less. WeatherEdge will not recommend "
            "extreme low-market-probability sides even when its model sees an edge."
        )

    if bool(r.get("settlement_rounding_risk", False)):
        ambiguous = r.get("settlement_ambiguous_degrees") or []
        degree_text = " and ".join(f"{int(v)}°F" for v in ambiguous)
        st.warning(
            "Excluded from Top Bets for settlement-source / rounding safety. "
            f"Based on the observed high, {degree_text} are treated as ambiguous "
            "whole-degree outcomes, so WeatherEdge will not recommend either side "
            "of a contract touching those degrees."
        )

    st.markdown("<div class='bet-shell'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card-title'>{r['city']} · {r['date_label']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='bet-callout'>"
        f"<div class='bet-callout-label'>CONTRACT</div>"
        f"<div class='bet-callout-main'>{r['side']} on “{r['market_subtitle']}”</div>"
        f"<div class='bet-callout-sub'>Current ask: {r['ask']*100:.0f}¢ · ticker {r['market_ticker']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    qscore = float(r.get("bet_quality_score", 0.0))
    qlabel = r.get("bet_quality_label", bet_quality_label(qscore))
    traj = float(r.get("trajectory_agreement_score", 0.5))
    st.markdown(
        f"<div class='quality-card'>"
        f"<div class='quality-top'>"
        f"<div><div class='quality-label'>BET QUALITY SCORE</div>"
        f"<div class='quality-value'>{qscore:.0f}<span style='font-size:1rem;color:#E0DAE7'>/100</span></div></div>"
        f"<div class='quality-grade'>{qlabel}</div>"
        f"</div>"
        f"<div class='quality-sub'>"
        f"Edge {r['conservative_edge']*100:+.1f} pp · NWS chance {r['conservative_prob']*100:.1f}% · "
        f"trajectory agreement {traj*100:.0f}% · cutoff {min_nws_chance*100:.0f}%"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latest NWS high", "—" if pd.isna(r["nws_high_f"]) else f"{int(r['nws_high_f'])}°F")
        obs = r.get("observed_high_f")
        st.metric("High observed so far", "—" if obs is None or pd.isna(obs) else f"{obs:.0f}°F")
        obs_time = r.get("observed_high_time_local")
        if obs is not None and not pd.isna(obs) and obs_time is not None and not pd.isna(obs_time):
            try:
                st.caption(f"Observed at {pd.Timestamp(obs_time).strftime('%-I:%M %p')} local time")
            except Exception:
                pass
        implied = r.get("kalshi_implied_temp_f")
        st.metric("Kalshi implied high (approx.)", "—" if implied is None or pd.isna(implied) else f"{implied:.1f}°F")
    with c2:
        st.metric("Kalshi price", f"{r['ask']*100:.0f}¢")
        st.metric("NWS-based chance", fmt_pct(r["conservative_prob"]))
        st.metric(
            "Weather Edge",
            f"{r['conservative_edge']*100:+.1f} pp",
            help=f"NWS-based chance minus the live Kalshi {r['side']} ask.",
        )

    recent_adj = r.get("recent_days_adjustment_f")
    recent_n = int(r.get("recent_days_n", 0) or 0)
    recent_consistency = float(r.get("recent_days_consistency", 0.0) or 0.0)
    yesterday_error = r.get("yesterday_forecast_error_f")
    if recent_n > 0 and recent_adj is not None and not pd.isna(recent_adj):
        recent_bits = [f"{recent_n} recent completed day" + ("s" if recent_n != 1 else "")]
        if yesterday_error is not None and not pd.isna(yesterday_error):
            recent_bits.append(f"yesterday miss {float(yesterday_error):+.1f}°F")
        recent_bits.append(f"{recent_consistency*100:.0f}% directional consistency")
        st.warning(
            "Recent forecast-error adjustment: "
            f"{float(recent_adj):+.1f}°F · " + " · ".join(recent_bits)
        )

    evolution_adj = r.get("forecast_evolution_adjustment_f")
    evolution_n = int(r.get("forecast_evolution_matches", 0) or 0)
    evolution_conf = float(r.get("forecast_evolution_confidence", 0.0) or 0.0)
    change_6h = r.get("forecast_change_6h_f")
    change_24h = r.get("forecast_change_24h_f")

    if evolution_n >= 6:
        bits = []
        if change_6h is not None and not pd.isna(change_6h):
            bits.append(f"6h revision {float(change_6h):+.1f}°F")
        if change_24h is not None and not pd.isna(change_24h):
            bits.append(f"24h revision {float(change_24h):+.1f}°F")
        bits.append(f"{evolution_n} historical matches")
        bits.append(f"{evolution_conf*100:.0f}% model confidence")
        st.info(
            "Forecast evolution adjustment: "
            f"{float(evolution_adj or 0.0):+.1f}°F · "
            + " · ".join(bits)
        )

    mismatch = r.get("temperature_mismatch_f")
    hours_left = r.get("hours_to_settlement")
    mismatch_text = "—" if mismatch is None or pd.isna(mismatch) else f"{mismatch:+.1f}°F"
    time_text = "—" if hours_left is None or pd.isna(hours_left) else (
        f"{hours_left:.1f}h" if hours_left < 48 else f"{hours_left/24:.1f}d"
    )
    mismatch_sub = "NWS minus Kalshi implied high"
    if mismatch is not None and not pd.isna(mismatch):
        mismatch_sub = "NWS warmer than market" if mismatch > 0 else (
            "NWS cooler than market" if mismatch < 0 else "NWS and market aligned"
        )
    time_sub = "Until market settlement"
    if hours_left is not None and not pd.isna(hours_left):
        if hours_left <= 6:
            time_sub = "Very close to settlement"
        elif hours_left <= 24:
            time_sub = "Settles within 24 hours"
        elif hours_left <= 48:
            time_sub = "Settles within 2 days"

    peak_left = r.get("hours_to_expected_peak")
    expected_peak = r.get("expected_peak_local")
    if peak_left is not None and not pd.isna(peak_left):
        peak_when = "past expected peak" if peak_left <= 0 else f"{peak_left:.1f}h to expected hottest point"
        if expected_peak is not None and not pd.isna(expected_peak):
            try:
                peak_clock = pd.Timestamp(expected_peak).strftime("%-I:%M %p")
                peak_when += f" · expected around {peak_clock}"
            except Exception:
                pass
        st.caption(
            f"NWS probability calibration: {peak_when} · "
            f"{int(r.get('nws_sigma_samples', 0) or 0)} historical forecast-error samples in this lead bucket."
        )

    st.markdown("<div class='section-kicker'>MARKET TIMING & MISMATCH</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='signal-strip'>"
        f"<div class='signal-tile'><div class='signal-label'>KALSHI TEMP MISMATCH</div>"
        f"<div class='signal-value'>{mismatch_text}</div><div class='signal-sub'>{mismatch_sub}</div></div>"
        f"<div class='signal-tile'><div class='signal-label'>TIME TO SETTLEMENT</div>"
        f"<div class='signal-value'>{time_text}</div><div class='signal-sub'>{time_sub}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


    contract_key = f"{r['market_ticker']}|{r['side']}"
    watched_bets = list(st.session_state.get("watched_bets", []))
    is_watching = contract_key in watched_bets

    if st.button(
        ("★ Stop watching" if is_watching else "☆ Watch this bet"),
        key=f"watch_toggle_{r['market_ticker']}_{r['side']}",
        use_container_width=True,
    ):
        if is_watching:
            watched_bets = [k for k in watched_bets if k != contract_key]
        else:
            watched_bets.append(contract_key)
        st.session_state.watched_bets = watched_bets
        st.rerun()

    st.link_button("Open this bet on Kalshi ↗", r["kalshi_event_url"], use_container_width=True)
    st.caption(f"Settlement location: **{r['station_hint']}**")

    if r.get("nws_forecast"):
        st.caption(f"NWS: {r['nws_forecast']}")

    st.markdown("<div class='section-kicker'>SOURCE DATA</div>", unsafe_allow_html=True)
    source_cols = st.columns(2)
    with source_cols[0]:
        if r.get("nws_forecast_url"):
            st.link_button("Latest NWS predictions ↗", r["nws_forecast_url"], use_container_width=True)
    with source_cols[1]:
        observed_url = r.get("observed_data_url")
        if observed_url and isinstance(observed_url, str) and observed_url.startswith(("http://", "https://")):
            st.link_button("Observed temperatures ↗", observed_url, use_container_width=True)

    if show_weather:
        range_chart = forecast_range_summary_chart(r, show_bet_overlay=(status_kind == "best"))
        if range_chart is not None:
            st.markdown("<div class='section-kicker'>DAILY HIGH PROBABILITY</div>", unsafe_allow_html=True)
            st.altair_chart(range_chart, use_container_width=True)
            render_probability_chart_legend(r)
        render_bet_forecast(r["city"], r["date"])

    st.markdown("<div class='section-kicker'>YOUR KALSHI POSITION</div>", unsafe_allow_html=True)
    st.info(
        "Kalshi account data is not connected yet. When read-only account access is added, "
        "this panel will show whether you hold this contract, your side, entry price, size, "
        "current value/P&L, and eventually entry-vs-now WeatherEdge conditions."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# Primary app navigation. Buttons from recommendations can jump directly into City Explorer.
def go_to_city_explorer(city, contract_key):
    """
    Streamlit callback used by navigation buttons.

    Widget callbacks run before the next script rerun, which makes it safe to
    update the radio widget's session-state key here. Directly assigning
    st.session_state.main_view after the radio has already been instantiated
    raises StreamlitAPIException.
    """
    st.session_state.explorer_city = city
    st.session_state.explorer_contract = contract_key
    st.session_state.main_view = "City Explorer"


main_view = st.radio(
    "Main section",
    ["Best Bets", "My Bets", "City Explorer"],
    horizontal=True,
    key="main_view",
    label_visibility="collapsed",
)

if main_view == "Best Bets":
    st.subheader("Best Bets")
    st.caption(
        "Tap any ranked bet to open that exact contract directly in City Explorer."
    )

    if qualified.empty:
        st.info(
            f"No candidates currently pass the safety filters: "
            f"NWS-based chance ≥ {min_nws_chance*100:.0f}% and Weather Edge ≥ {min_gap*100:.0f} pp."
        )
    else:
        best_rows = [r for _, r in qualified.iterrows()]
        for rank, r in enumerate(best_rows, start=1):
            contract_key = f"{r['market_ticker']}|{r['side']}"

            bet_date = r.get("date_label")
            if not bet_date:
                try:
                    bet_date = pd.Timestamp(r.get("date")).strftime("%a %b %-d")
                except Exception:
                    bet_date = str(r.get("date") or "—")

            nws_high = r.get("nws_high_f")
            adjusted_high = r.get("adjusted_nws_center_f")
            nws_high_text = (
                "—"
                if nws_high is None or pd.isna(nws_high)
                else f"{float(nws_high):.1f}°F"
            )
            adjusted_high_text = (
                "—"
                if adjusted_high is None or pd.isna(adjusted_high)
                else f"{float(adjusted_high):.1f}°F"
            )

            st.markdown(
                f"<div class='bet-callout'>"
                f"<div class='bet-callout-label'>TOP BET #{rank} · {bet_date}</div>"
                f"<div class='bet-callout-main'>{r['city']} · {r['market_subtitle']} · {r['side']}</div>"
                f"<div class='bet-callout-sub'>"
                f"Latest NWS high {nws_high_text} · "
                f"WeatherEdge adjusted high {adjusted_high_text} · "
                f"Quality {r['bet_quality_score']:.0f}/100 · "
                f"WeatherEdge chance {r['conservative_prob']*100:.0f}% · "
                f"Edge {r['conservative_edge']*100:+.1f} pp"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            st.button(
                "Open this bet in City Explorer →",
                key=f"open_best_{rank}_{r['market_ticker']}_{r['side']}",
                use_container_width=True,
                on_click=go_to_city_explorer,
                args=(r["city"], contract_key),
            )

elif main_view == "My Bets":
    st.subheader("My Bets")
    st.caption(
        "Your personal bet dashboard. Watched contracts appear here immediately. "
        "Placed/active bets will populate automatically once read-only Kalshi account access is connected."
    )

    watched_keys = list(st.session_state.get("watched_bets", []))
    placed_bets = list(st.session_state.get("placed_bets", []))

    # Current open-market lookup from the live scan.
    live_lookup = {
        f"{row['market_ticker']}|{row['side']}": row
        for _, row in df.iterrows()
    }

    # Normalize future Kalshi-position entries into keyed rows.
    placed_lookup = {}
    for item in placed_bets:
        try:
            key = f"{item.get('market_ticker')}|{item.get('side')}"
            placed_lookup[key] = item
        except Exception:
            continue

    all_keys = []
    for key in watched_keys + list(placed_lookup.keys()):
        if key not in all_keys:
            all_keys.append(key)

    if not all_keys:
        st.info(
            "No bets here yet. Open any contract in City Explorer and tap "
            "“Watch this bet.” Once Kalshi is connected, active placed bets will also appear automatically."
        )
    else:
        st.markdown("<div class='section-kicker'>YOUR BETS</div>", unsafe_allow_html=True)

        for idx, key in enumerate(all_keys):
            live_row = live_lookup.get(key)
            placed = placed_lookup.get(key)
            is_watching = key in watched_keys
            is_placed = placed is not None

            if live_row is not None:
                city = live_row["city"]
                subtitle = live_row["market_subtitle"]
                side = live_row["side"]
                price = live_row["ask"] * 100
                nws_prob = live_row["conservative_prob"] * 100
                edge = live_row["conservative_edge"] * 100
                quality = live_row["bet_quality_score"]
                ticker = live_row["market_ticker"]
            else:
                city = placed.get("city", "Unknown city") if placed else "Unavailable"
                subtitle = placed.get("market_subtitle", key) if placed else key
                side = placed.get("side", "") if placed else ""
                price = None
                nws_prob = None
                edge = None
                quality = None
                ticker = placed.get("market_ticker", key.split("|")[0]) if placed else key.split("|")[0]

            badges = []
            if is_placed:
                badges.append("PLACED BET")
            if is_watching:
                badges.append("WATCHING")
            badge_text = " · ".join(badges)

            st.markdown(
                f"<div class='bet-callout'>"
                f"<div class='bet-callout-label'>{badge_text}</div>"
                f"<div class='bet-callout-main'>{city} · {subtitle} · {side}</div>"
                + (
                    f"<div class='bet-callout-sub'>"
                    f"Current ask {price:.0f}¢ · NWS {nws_prob:.0f}% · "
                    f"Edge {edge:+.1f} pp · Quality {quality:.0f}/100"
                    f"</div>"
                    if live_row is not None else
                    "<div class='bet-callout-sub'>This contract is not in the current open-market scan.</div>"
                )
                + "</div>",
                unsafe_allow_html=True,
            )

            if is_placed:
                cols = st.columns(3)
                entry_price = placed.get("entry_price")
                contracts = placed.get("contracts")
                pnl = placed.get("unrealized_pnl")
                cols[0].metric(
                    "Entry price",
                    "—" if entry_price is None else f"{float(entry_price)*100:.0f}¢"
                )
                cols[1].metric(
                    "Contracts",
                    "—" if contracts is None else f"{int(contracts)}"
                )
                cols[2].metric(
                    "Unrealized P/L",
                    "—" if pnl is None else f"${float(pnl):+.2f}"
                )

            if live_row is not None:
                st.button(
                    "Open in City Explorer →",
                    key=f"my_bets_open_{idx}_{ticker}_{side}",
                    use_container_width=True,
                    on_click=go_to_city_explorer,
                    args=(city, key),
                )

            if is_watching:
                if st.button(
                    "Remove from Watching",
                    key=f"my_bets_remove_watch_{idx}_{ticker}_{side}",
                    use_container_width=True,
                ):
                    st.session_state.watched_bets = [
                        k for k in st.session_state.get("watched_bets", [])
                        if k != key
                    ]
                    st.rerun()

        if not placed_bets:
            st.markdown("<div class='section-kicker'>KALSHI CONNECTION</div>", unsafe_allow_html=True)
            st.info(
                "Placed bets are not connected yet. This page is ready for read-only Kalshi "
                "position data so active trades can appear automatically with entry price, size, "
                "current value/P&L, and WeatherEdge conditions."
            )

elif main_view == "City Explorer":
    st.subheader("City Explorer")
    st.caption(
        "Pick a city and date, then move through its contracts at the top. "
        "Collapse the bet panel whenever you want a weather-only view."
    )

    available_cities = [c for c in PRESETS if c in set(df["city"])]
    if not available_cities:
        st.info("No configured cities currently have open temperature contracts.")
    else:
        default_city = st.session_state.explorer_city
        if default_city not in available_cities:
            default_city = available_cities[0]

        city = st.selectbox(
            "Explore city",
            available_cities,
            index=available_cities.index(default_city),
            key="city_explorer_select",
        )
        st.session_state.explorer_city = city
        city_df = df[df["city"] == city].copy().sort_values(
            ["date", "market_ticker", "side"]
        )

        available_dates = list(city_df["date"].drop_duplicates())
        date_labels = [
            pd.Timestamp(d).strftime("%a %b %-d") for d in available_dates
        ]

        desired_contract = st.session_state.get("explorer_contract")
        desired_date_idx = 0
        if desired_contract:
            for idx, d in enumerate(available_dates):
                probe = city_df[city_df["date"] == d]
                keys = {
                    f"{r['market_ticker']}|{r['side']}"
                    for _, r in probe.iterrows()
                }
                if desired_contract in keys:
                    desired_date_idx = idx
                    break

        selected_date_label = st.selectbox(
            "Market date",
            date_labels,
            index=desired_date_idx,
            key=f"explorer_date_{city}",
        )
        selected_date = available_dates[date_labels.index(selected_date_label)]
        date_df = city_df[city_df["date"] == selected_date].copy()

        contract_rows = [
            r for _, r in date_df.sort_values(
                ["market_subtitle", "side", "ask"],
                ascending=[True, True, True],
            ).iterrows()
        ]
        contract_keys = [
            f"{r['market_ticker']}|{r['side']}" for r in contract_rows
        ]

        desired = st.session_state.get("explorer_contract")
        default_idx = contract_keys.index(desired) if desired in contract_keys else 0

        # Persist an index per city/date so Previous/Next acts like a swipeable deck.
        idx_key = f"explorer_contract_idx_{city}_{selected_date}"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = default_idx
        if desired in contract_keys:
            st.session_state[idx_key] = contract_keys.index(desired)
        st.session_state[idx_key] = max(
            0, min(len(contract_rows) - 1, int(st.session_state[idx_key]))
        )

        selected_row = contract_rows[st.session_state[idx_key]]
        selected_key = contract_keys[st.session_state[idx_key]]
        st.session_state.explorer_contract = selected_key

        selected_status, selected_status_kind = contract_status(selected_row)
        watch_key = selected_key
        is_watching = watch_key in st.session_state.get("watched_bets", [])
        placed_lookup = {
            f"{x.get('market_ticker')}|{x.get('side')}": x
            for x in st.session_state.get("placed_bets", [])
            if isinstance(x, dict)
        }
        is_placed = watch_key in placed_lookup

        bet_panel = st.expander(
            f"{selected_row['side']} · {selected_row['market_subtitle']} · "
            f"{selected_row['ask']*100:.0f}¢",
            expanded=True,
        )
        with bet_panel:
            badges = []
            if selected_status_kind == "best":
                badges.append(selected_status)
            if is_watching:
                badges.append("WATCHING")
            if is_placed:
                badges.append("PLACED BET")
            if badges:
                st.markdown(
                    "<div class='bet-callout-label'>"
                    + " · ".join(badges)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div class='bet-callout-main'>{selected_row['side']} on "
                f"{selected_row['market_subtitle']}</div>"
                f"<div class='bet-callout-sub'>Ask {selected_row['ask']*100:.0f}¢ · "
                f"WeatherEdge chance {selected_row['conservative_prob']*100:.0f}% · "
                f"Edge {selected_row['conservative_edge']*100:+.1f} pp · "
                f"Quality {selected_row['bet_quality_score']:.0f}/100</div>",
                unsafe_allow_html=True,
            )

            yes_ask = selected_row.get("yes_ask")
            no_ask = selected_row.get("no_ask")
            market_bits = []
            if yes_ask is not None and not pd.isna(yes_ask):
                market_bits.append(f"YES {float(yes_ask)*100:.0f}¢")
            if no_ask is not None and not pd.isna(no_ask):
                market_bits.append(f"NO {float(no_ask)*100:.0f}¢")
            volume = float(selected_row.get("volume", 0) or 0)
            open_interest = float(selected_row.get("open_interest", 0) or 0)
            if volume > 0:
                market_bits.append(f"volume {volume:,.0f}")
            if open_interest > 0:
                market_bits.append(f"open interest {open_interest:,.0f}")
            if market_bits:
                st.caption("Market: " + " · ".join(market_bits))

            if selected_status_kind == "best":
                why_text, risk_text = bet_explanation_blurb(selected_row)
                st.markdown(
                    f"**Why this bet?** {why_text} **Watch:** {risk_text}"
                )

            kalshi_url = selected_row.get("kalshi_event_url") or selected_row.get("contract_url")
            if kalshi_url:
                st.link_button(
                    "Open this bet on Kalshi ↗",
                    kalshi_url,
                    use_container_width=True,
                )

            prev_col, count_col, next_col = st.columns([1, 1.2, 1])
            with prev_col:
                if st.button(
                    "← Previous",
                    disabled=st.session_state[idx_key] <= 0,
                    use_container_width=True,
                    key=f"prev_contract_{city}_{selected_date}",
                ):
                    st.session_state[idx_key] -= 1
                    st.session_state.explorer_contract = contract_keys[
                        st.session_state[idx_key]
                    ]
                    st.rerun()
            with count_col:
                st.caption(
                    f"Bet {st.session_state[idx_key] + 1} of {len(contract_rows)}"
                )
            with next_col:
                if st.button(
                    "Next →",
                    disabled=st.session_state[idx_key] >= len(contract_rows) - 1,
                    use_container_width=True,
                    key=f"next_contract_{city}_{selected_date}",
                ):
                    st.session_state[idx_key] += 1
                    st.session_state.explorer_contract = contract_keys[
                        st.session_state[idx_key]
                    ]
                    st.rerun()

            if is_watching:
                if st.button(
                    "★ Remove from Watching",
                    use_container_width=True,
                    key=f"unwatch_top_{selected_key}",
                ):
                    st.session_state.watched_bets = [
                        k for k in st.session_state.get("watched_bets", [])
                        if k != watch_key
                    ]
                    st.rerun()
            else:
                if st.button(
                    "☆ Watch this bet",
                    use_container_width=True,
                    key=f"watch_top_{selected_key}",
                ):
                    watched = list(st.session_state.get("watched_bets", []))
                    if watch_key not in watched:
                        watched.append(watch_key)
                    st.session_state.watched_bets = watched
                    st.rerun()

        representative = hydrate_city_explorer_row(selected_row)

        # Weather summary stays immediately below the collapsible bet deck.
        st.markdown(
            f"<div class='card-title'>{city} · {selected_date_label}</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        raw_nws_high = float(representative["nws_high_f"])
        adjusted_center = representative.get("adjusted_nws_center_f")
        if adjusted_center is None or pd.isna(adjusted_center):
            adjusted_center = (
                raw_nws_high
                + float(representative.get("nws_bias_f") or 0.0)
                + float(representative.get("trajectory_adjustment_f") or 0.0)
            )
            observed_floor = representative.get("observed_high_f")
            if observed_floor is not None and not pd.isna(observed_floor):
                adjusted_center = max(float(adjusted_center), float(observed_floor))
        adjusted_center = float(adjusted_center)
        m1.metric(
            "Latest NWS high",
            f"{raw_nws_high:.1f}°F",
            help="The latest published NWS daily-high forecast for this city and date.",
        )
        m2.metric(
            "WeatherEdge adjusted high",
            f"{adjusted_center:.1f}°F",
            delta=(
                None
                if abs(adjusted_center - raw_nws_high) < 0.05
                else f"{adjusted_center - raw_nws_high:+.1f}° vs NWS"
            ),
            help="WeatherEdge model center after historical calibration and live trajectory adjustment. This is not an NWS-published value.",
        )
        obs = representative.get("observed_high_f")
        obs_time = representative.get("observed_high_time_local")
        obs_label = "High observed so far"
        if obs_time is not None and not pd.isna(obs_time):
            try:
                obs_label += f" ({pd.Timestamp(obs_time).strftime('%-I:%M %p')})"
            except Exception:
                pass
        m3.metric(
            obs_label,
            "—" if obs is None or pd.isna(obs) else f"{obs:.1f}°F",
        )
        implied = representative.get("kalshi_implied_temp_f")
        m4.metric(
            "Kalshi implied high",
            "—" if implied is None or pd.isna(implied) else f"{implied:.1f}°F",
        )

        if bool(representative.get("daily_high_locked", False)):
            st.success(
                f"Daily high treated as locked at {float(representative.get('observed_high_f')):.1f}°F. "
                "The expected hottest part of the day is clearly past, so WeatherEdge "
                "is not assigning probability to a higher temperature."
            )

        gap_now = representative.get("trajectory_current_gap_f")
        if gap_now is not None and not pd.isna(gap_now):
            trajectory_adj = float(representative.get("trajectory_adjustment_f") or 0.0)
            evolution_adj = float(representative.get("forecast_evolution_adjustment_f") or 0.0)
            recent_adj = float(representative.get("recent_days_adjustment_f") or 0.0)
            unconstrained = representative.get("unconstrained_adjusted_center_f")
            parts = []
            if abs(trajectory_adj) >= 0.1:
                parts.append(f"live trajectory {trajectory_adj:+.1f}°")
            if abs(evolution_adj) >= 0.1:
                parts.append(f"forecast evolution {evolution_adj:+.1f}°")
            if abs(recent_adj) >= 0.1:
                parts.append(f"recent-day error {recent_adj:+.1f}°")
            if parts:
                explanation = " · ".join(parts)
                if (
                    unconstrained is not None
                    and not pd.isna(unconstrained)
                    and abs(float(unconstrained) - adjusted_center) >= 0.1
                ):
                    st.caption(
                        f"WeatherEdge adjustments: {explanation}. "
                        f"Unconstrained center {float(unconstrained):.1f}°F was capped "
                        f"to a physically plausible {adjusted_center:.1f}°F using today's "
                        f"NWS forecast, observed high, and time remaining to the expected peak."
                    )
                else:
                    st.caption(
                        f"WeatherEdge adjustments: {explanation}. "
                        f"Current model center: {adjusted_center:.1f}°F."
                    )

        st.markdown("<div class='section-kicker'>SOURCE DATA</div>", unsafe_allow_html=True)
        source_cols = st.columns(2)
        with source_cols[0]:
            if representative.get("nws_forecast_url"):
                st.link_button(
                    "Latest NWS predictions ↗",
                    representative["nws_forecast_url"],
                    use_container_width=True,
                )
        with source_cols[1]:
            observed_url = representative.get("observed_data_url")
            if observed_url and isinstance(observed_url, str):
                st.link_button(
                    "Observed temperatures ↗",
                    observed_url,
                    use_container_width=True,
                )

        if (
            representative.get("previous_day_high_f") is not None
            and representative.get("previous_day_predicted_high_f") is None
        ):
            st.warning(
                "Previous-day observed high is available, but WeatherEdge could not "
                "find a stored NWS forecast snapshot for that previous day's high. "
                "This is a missing-history condition, not a hidden chart line."
            )

        range_chart = forecast_range_summary_chart(
            representative,
            show_bet_overlay=(selected_status_kind == "best"),
        )
        if range_chart is not None:
            st.markdown(
                "<div class='section-kicker'>DAILY HIGH PROBABILITY</div>",
                unsafe_allow_html=True,
            )
            st.altair_chart(range_chart, use_container_width=True)
            render_probability_chart_legend(representative)

        render_bet_forecast(city, selected_date)

        # Full contract detail is still available, but no longer buried below
        # weather as the only way to see what bet is selected.
        with st.expander("Full contract details", expanded=False):
            render_contract_detail(selected_row, show_weather=False)

with st.expander("How to read this"):
    st.markdown(
        """
**Latest NWS high** is the newest official point forecast for the settlement-station area.

**NWS-based chance** converts the latest NWS high into a probability distribution using historical NWS-versus-observed errors. Large recent misses now primarily widen uncertainty and only modestly shift the center, but guardrails prevent historical errors from implying implausible same-day heating. Calibration is keyed to how far the forecast is from the city’s expected hottest point of the day. On same-day markets, the distribution is now also conditioned on the live observed-vs-NWS trajectory: persistent misses matter increasingly as the expected peak approaches, and a near-peak shortfall can pull the center lower and tighten remaining-upside uncertainty. GFS is not used in this probability.

**Observed high so far** is a hard floor on same-day markets. WeatherEdge assigns zero final-high probability below the exact highest temperature already observed, then renormalizes all remaining probability above that floor.

**Kalshi implied high (approx.)** is reconstructed from the live prices of the event's temperature brackets. It is useful for spotting NWS-vs-market temperature disagreement, but it may differ slightly from the forecast number displayed in Kalshi's app.

**Weather Edge** is NWS-based contract probability minus the live Kalshi ask for the displayed side. A positive value favors that side; for example, +20 pp toward YES means the NWS-based chance is 20 percentage points above the YES ask. GFS is not used in Weather Edge.

**Time to settlement** matters because the NWS uncertainty used by WeatherEdge gets tighter as the outcome gets closer. The opportunity ranking prioritizes contract edge, then larger NWS-vs-Kalshi temperature mismatches, with an extra boost as settlement gets close.

The GFS ensemble remains on the forecast-range chart only as optional context. It does **not** affect probabilities, qualification, gaps, or rankings.
"""
    )

st.caption(
    "Research tool only. Forecasts can be wrong, prices can move, and settlement rules matter."
)
