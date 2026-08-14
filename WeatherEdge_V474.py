
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
                "dy": -18,
                "dx": -24,
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
            "dy": -18,
            "dx": 24,
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
                "dy": -18,
                "dx": -24,
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
                    "dy": 20,
                    "dx": 24,
                })
    except Exception:
        pass

    if label_rows:
        label_df = pd.DataFrame(label_rows)

        # High-visibility peak labels. Each series gets its own color and x/y
        # offset so current/previous observed and forecast maxima do not sit on
        # top of one another on a phone.
        label_colors = {
            "Observed": "#FF78C9",
            "NWS prediction": "#B99CFF",
            "Previous observed high": "#FF775F",
            "Previous NWS predicted high": "#45D9FF",
        }

        for kind_name, kind_color in label_colors.items():
            kind_df = label_df[label_df["kind"] == kind_name]
            if kind_df.empty:
                continue

            # Larger peak marker.
            layers.append(
                alt.Chart(kind_df)
                .mark_point(
                    filled=True,
                    size=170,
                    color=kind_color,
                    stroke="#050814",
                    strokeWidth=2.4,
                )
                .encode(
                    x=alt.X("time:T"),
                    y=alt.Y("temp_f:Q"),
                    tooltip=[
                        alt.Tooltip("kind:N", title="High"),
                        alt.Tooltip("temp_f:Q", title="Temperature", format=".1f"),
                        alt.Tooltip("time:T", title="Time", format="%b %-d, %-I:%M %p"),
                    ],
                )
            )

            dx = int(kind_df.iloc[0].get("dx", 0))
            dy = int(kind_df.iloc[0].get("dy", -18))
            baseline = "bottom" if dy < 0 else "top"

            # First text layer is a thick dark halo.
            layers.append(
                alt.Chart(kind_df)
                .mark_text(
                    align="center",
                    baseline=baseline,
                    dx=dx,
                    dy=dy,
                    fontSize=18,
                    fontWeight="bold",
                    color=kind_color,
                    stroke="#050814",
                    strokeWidth=6,
                )
                .encode(
                    x=alt.X("time:T"),
                    y=alt.Y("temp_f:Q"),
                    text="label:N",
                )
            )
            # Crisp colored text drawn over the halo.
            layers.append(
                alt.Chart(kind_df)
                .mark_text(
                    align="center",
                    baseline=baseline,
                    dx=dx,
                    dy=dy,
                    fontSize=18,
                    fontWeight="bold",
                    color=kind_color,
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
            height=390,
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

    # Adapt the horizontal scale to the actual snapshot window. More ticks are
    # useful over short windows; longer windows emphasize day boundaries.
    span_hours = max(
        1.0,
        (
            history["snapshot_time"].max()
            - history["snapshot_time"].min()
        ).total_seconds() / 3600.0,
    )
    if span_hours <= 18:
        x_tick_count = 10
        x_format = "%-I %p"
    elif span_hours <= 42:
        x_tick_count = 12
        x_format = "%b %-d · %-I %p"
    elif span_hours <= 84:
        x_tick_count = 10
        x_format = "%b %-d · %-I %p"
    else:
        x_tick_count = 9
        x_format = "%b %-d"

    # Explicit y domain / integer-degree ticks make 1°F forecast revisions
    # immediately visible instead of compressing them into a vague scale.
    y_low = float(history["predicted_high_f"].min())
    y_high = float(history["predicted_high_f"].max())
    if abs(y_high - y_low) < 1.0:
        y_low -= 1.0
        y_high += 1.0
    else:
        y_low = math.floor(y_low - 0.75)
        y_high = math.ceil(y_high + 0.75)
    y_tick_count = max(4, min(10, int(round(y_high - y_low)) + 1))

    base = (
        alt.Chart(history)
        .mark_line(
            point=alt.OverlayMarkDef(
                filled=True,
                size=76,
                stroke="#08101F",
                strokeWidth=1.4,
            ),
            strokeWidth=3.5,
            color="#B79CFF",
            interpolate="step-after",
        )
        .encode(
            x=alt.X(
                "snapshot_time:T",
                title="Forecast snapshot time",
                axis=alt.Axis(
                    format=x_format,
                    labelAngle=-22,
                    labelPadding=9,
                    labelOverlap="greedy",
                    tickCount=x_tick_count,
                    grid=True,
                    gridOpacity=0.52,
                    gridWidth=1.15,
                    tickSize=8,
                    tickWidth=1.4,
                    domainWidth=1.4,
                ),
            ),
            y=alt.Y(
                "predicted_high_f:Q",
                title="Predicted daily high (°F)",
                scale=alt.Scale(
                    zero=False,
                    domain=[y_low, y_high],
                    nice=False,
                ),
                axis=alt.Axis(
                    tickCount=y_tick_count,
                    format=".0f",
                    grid=True,
                    gridOpacity=0.54,
                    gridWidth=1.15,
                    tickSize=8,
                    tickWidth=1.4,
                    domainWidth=1.4,
                    labelPadding=8,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "snapshot_time:T",
                    title="Forecast changed / snapshot",
                    format="%b %-d, %-I:%M %p",
                ),
                alt.Tooltip(
                    "predicted_high_f:Q",
                    title="Predicted high",
                    format=".1f",
                ),
            ],
        )
    )

    # Highlight the exact moments where the predicted high changed.
    change_history = history.copy()
    change_history["_prev"] = change_history["predicted_high_f"].shift(1)
    change_history = change_history[
        change_history["_prev"].isna()
        | (
            (change_history["predicted_high_f"] - change_history["_prev"])
            .abs() >= 0.05
        )
    ].copy()

    change_marks = (
        alt.Chart(change_history)
        .mark_point(
            filled=True,
            size=120,
            color="#42DCFF",
            stroke="#08101F",
            strokeWidth=2,
        )
        .encode(
            x=alt.X("snapshot_time:T"),
            y=alt.Y("predicted_high_f:Q"),
            tooltip=[
                alt.Tooltip(
                    "snapshot_time:T",
                    title="Change time",
                    format="%b %-d, %-I:%M %p",
                ),
                alt.Tooltip(
                    "predicted_high_f:Q",
                    title="New predicted high",
                    format=".1f",
                ),
            ],
        )
    )

    chart = (
        alt.layer(base, change_marks)
        .properties(
            height=315,
            background="#0A0D18",
            title=alt.TitleParams(
                text="How the predicted high has changed",
                subtitle=(
                    "Cyan markers show each forecast revision · tap/hover for exact time"
                ),
                anchor="start",
            ),
        )
        .configure_axis(
            labelFontSize=15,
            titleFontSize=16,
            labelFontWeight=650,
            titleFontWeight=750,
            labelColor="#F0ECF7",
            titleColor="#FFFFFF",
            gridColor="#77728B",
            gridOpacity=0.50,
            domainColor="#B2ACC3",
            tickColor="#B2ACC3",
        )
        .configure_title(
            fontSize=19,
            subtitleFontSize=13,
            color="#FFFFFF",
            subtitleColor="#C8C3D4",
        )
        .configure_view(
            stroke="#26304C",
            strokeWidth=1,
        )
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
  min-height:132px;
  overflow:hidden;
  border:1px solid rgba(83,132,255,.36);
  border-radius:22px;
  padding:1.05rem 1.25rem .95rem;
  margin:.15rem 0 1rem;
  background:
    linear-gradient(90deg, rgba(2,7,24,.96) 0%, rgba(2,7,24,.73) 46%, rgba(2,7,24,.26) 100%),
    url("data:image/webp;base64,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") 72% 48% / cover no-repeat;
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
  font-size:1.78rem;
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
  margin-top:.18rem;
  color:#D955D4;
  font-size:.72rem;
  letter-spacing:.095em;
  text-transform:uppercase;
  font-weight:790;
}
.we-sub {
  max-width:410px;
  color:#B8C6E8;
  font-size:.82rem;
  margin-top:.32rem;
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


/* compact orbital identity: fewer objects, more breathing room */
.we-hero {
  --we-hero-image: url("data:image/webp;base64,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") 72% 48% / cover no-repeat;
  background:
    linear-gradient(90deg, rgba(2,7,24,.985) 0%, rgba(2,7,24,.90) 35%, rgba(2,7,24,.30) 70%, rgba(2,7,24,.10) 100%),
    var(--we-hero-image, none);
}
.we-hero .we-sub {max-width:390px;}
.we-hero .we-tag {max-width:410px;}
.we-hero .we-status,
.we-hero .we-weather-strip,
.we-hero .we-orbit,
.we-hero .we-sat {display:none!important;}

.we-satellite-badge {
  display:inline-flex;
  align-items:center;
  gap:.48rem;
  margin-top:.58rem;
  padding:.32rem .58rem;
  border:1px solid rgba(72,211,255,.28);
  border-radius:999px;
  background:rgba(3,12,31,.66);
  color:#C8EFFF!important;
  font-size:.67rem;
  font-weight:800;
  letter-spacing:.075em;
  text-transform:uppercase;
  backdrop-filter:blur(8px);
}
.we-satellite-badge .live {
  width:6px;height:6px;border-radius:50%;
  background:#29E6A6;box-shadow:0 0 8px rgba(41,230,166,.8);
}
.we-satellite-badge .sat {font-size:.92rem;}

.we-mission-strip {
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:.55rem;
  margin:.55rem 0 .9rem;
}
.we-mission-item {
  min-height:54px;
  display:flex;
  align-items:center;
  gap:.65rem;
  padding:.62rem .78rem;
  border:1px solid rgba(92,123,213,.26);
  border-radius:13px;
  background:linear-gradient(145deg,rgba(7,16,40,.78),rgba(7,12,29,.72));
  backdrop-filter:blur(12px);
}
.we-mission-icon {
  width:29px;height:29px;
  display:grid;place-items:center;
  flex:0 0 29px;
  border-radius:9px;
  background:rgba(111,74,255,.11);
  border:1px solid rgba(127,87,255,.24);
  font-size:1rem;
  filter:drop-shadow(0 0 7px rgba(82,205,255,.18));
}
.we-mission-title {
  color:#EAF0FF!important;
  font-size:.68rem;
  font-weight:850;
  letter-spacing:.055em;
  text-transform:uppercase;
}
.we-mission-sub {
  color:#8799C2!important;
  font-size:.61rem;
  line-height:1.2;
  margin-top:.08rem;
}

.we-weather-dock {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:.8rem;
  margin:1rem 0 .85rem;
  padding:.65rem .85rem;
  border-top:1px solid rgba(85,112,198,.20);
  border-bottom:1px solid rgba(85,112,198,.20);
}
.we-weather-dock-copy {
  color:#8FA2CD!important;
  font-size:.67rem;
  font-weight:800;
  letter-spacing:.105em;
  text-transform:uppercase;
}
.we-weather-dock-icons {
  display:flex;gap:.72rem;align-items:center;
  font-size:1.05rem;
}
.we-weather-dock-icons span {
  opacity:.88;
  filter:drop-shadow(0 0 6px rgba(103,190,255,.22));
}

/* satellite motif repeated quietly across the app rather than crowding the hero */
.section-kicker:before {
  content:"🛰";
  color:#63DFFF;
  margin-right:.45rem;
  filter:drop-shadow(0 0 6px rgba(34,214,255,.35));
}
.we-orbit-rail {
  background:
    linear-gradient(90deg,rgba(8,16,39,.76),rgba(10,17,39,.58))!important;
  border-color:rgba(85,116,205,.22)!important;
}
.we-orbit-icons span {font-size:1rem;opacity:.82;}

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
  .we-tag {font-size:.62rem;line-height:1.35;max-width:67%;}
  .we-satellite-badge {font-size:.58rem;padding:.24rem .42rem;margin-top:.42rem;}
  .we-mission-strip {grid-template-columns:1fr;gap:.4rem;margin:.45rem 0 .72rem;}
  .we-mission-item {min-height:43px;padding:.48rem .62rem;}
  .we-mission-icon {width:25px;height:25px;flex-basis:25px;font-size:.88rem;}
  .we-mission-sub {display:none;}
  .we-weather-dock {padding:.52rem .55rem;margin:.75rem 0 .7rem;}
  .we-weather-dock-copy {font-size:.57rem;}
  .we-weather-dock-icons {gap:.48rem;font-size:.95rem;}

  .block-container {padding-left:.82rem;padding-right:.82rem;padding-top:.45rem;}
  .we-hero {min-height:126px;padding:.88rem .9rem;background-position:70% 48%;}
  .we-brand {font-size:1.46rem;}
  .we-sat {right:1.55rem;top:1.25rem;font-size:1.75rem;}
  .we-orbit {right:.65rem;top:.5rem;width:120px;height:60px;}
  .we-sub {max-width:68%;font-size:.73rem;line-height:1.38;}
  .we-weather-strip {font-size:1.1rem;gap:.38rem;}
  .we-status {display:none!important;}
  .we-chip {font-size:.62rem;padding:.25rem .42rem;}
  .signal-strip {grid-template-columns:1fr 1fr;gap:.45rem;}
  .signal-tile {padding:.64rem .68rem;}
  [data-testid="stMetricValue"] {font-size:1.42rem!important;}
  h1 {font-size:2rem!important;}
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* =========================
   V474: NEON COAST / ORBIT
   Clean mobile-first visual pass.
   ========================= */

:root {
  --coast-pink:#ff4fc8;
  --coast-coral:#ff6b87;
  --coast-violet:#8d62ff;
  --coast-cyan:#3ce6ff;
  --coast-aqua:#55f4d0;
  --coast-sun:#ffc857;
  --night:#030817;
  --night2:#07112b;
}

/* Less decoration in the background. One restrained coastal glow. */
.stApp {
  background:
    radial-gradient(ellipse at 88% 2%, rgba(255,79,200,.11), transparent 23rem),
    radial-gradient(ellipse at 8% 36%, rgba(60,230,255,.07), transparent 26rem),
    linear-gradient(170deg,#030713 0%,#07102a 47%,#040817 100%) !important;
}
.stApp:before {
  opacity:.16 !important;
  background-image:
    radial-gradient(circle at 20% 22%, rgba(255,255,255,.8) 0 1px, transparent 1.5px),
    radial-gradient(circle at 78% 35%, rgba(93,220,255,.85) 0 1px, transparent 1.5px) !important;
  background-size:300px 300px, 410px 410px !important;
}

/* The masthead is now genuinely small. */
.we-hero-clean {
  min-height:104px !important;
  height:104px;
  margin:.05rem 0 .72rem !important;
  padding:.88rem 1rem !important;
  border-radius:16px !important;
  border:1px solid rgba(98,147,255,.28) !important;
  background:
    linear-gradient(90deg,rgba(3,7,22,.98) 0%,rgba(3,7,22,.86) 43%,rgba(3,7,22,.20) 78%,rgba(3,7,22,.08) 100%),
    var(--we-hero-image) !important;
  box-shadow:
    0 10px 30px rgba(0,0,0,.24),
    0 0 28px rgba(255,79,200,.045) !important;
}
.we-hero-clean:after {
  height:2px !important;
  background:linear-gradient(90deg,var(--coast-pink),var(--coast-violet),var(--coast-cyan)) !important;
}
.we-hero-clean .we-brand {
  font-size:1.55rem !important;
  line-height:1.05 !important;
  max-width:62%;
  white-space:nowrap;
}
.we-hero-clean .we-brand span {
  font-size:.62rem !important;
  color:var(--coast-pink) !important;
}
.we-hero-clean .we-tag {
  max-width:58% !important;
  margin-top:.34rem !important;
  font-size:.64rem !important;
  line-height:1.25 !important;
  letter-spacing:.08em !important;
  color:#e8b6e7 !important;
}
.we-mini-status {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  margin-top:.48rem;
  color:#a9b9dc;
  font-size:.58rem;
  font-weight:850;
  letter-spacing:.12em;
}
.we-mini-status span {
  width:6px;height:6px;border-radius:50%;
  background:#43efb4;
  box-shadow:0 0 8px rgba(67,239,180,.72);
}

/* Kill V474's extra top-of-page modules even if stale markup survives. */
.we-mission-strip,
.we-weather-dock,
.we-orbit-rail:first-of-type {
  display:none !important;
}

/* Clean glass, not glass everywhere. */
[data-testid="stMetric"],
[data-testid="stExpander"],
.bet-callout,
.quality-card,
.signal-tile {
  background:linear-gradient(145deg,rgba(11,19,46,.91),rgba(5,12,29,.93)) !important;
  border-color:rgba(103,135,219,.22) !important;
  box-shadow:0 7px 22px rgba(0,0,0,.20) !important;
  backdrop-filter:blur(9px) !important;
}
[data-testid="stMetric"] {
  border-radius:13px !important;
  padding:11px 12px !important;
}
[data-testid="stMetric"]:hover {
  transform:none !important;
  border-color:rgba(60,230,255,.38) !important;
}

/* Coastal-vaporwave accent: sunset pink + ocean cyan, used sparingly. */
.section-kicker {
  color:#d7c6ff !important;
  letter-spacing:.15em !important;
}
.section-kicker:before {
  content:"✦" !important;
  color:var(--coast-pink) !important;
  filter:drop-shadow(0 0 6px rgba(255,79,200,.38));
}
.bet-callout {
  border-color:rgba(255,79,200,.34) !important;
  background:
    radial-gradient(circle at 92% 14%,rgba(255,94,163,.11),transparent 8rem),
    linear-gradient(145deg,rgba(23,18,57,.94),rgba(6,15,34,.94)) !important;
}
.quality-card {
  border-color:rgba(60,230,255,.30) !important;
}
.quality-value {
  color:var(--coast-pink) !important;
}
.quality-grade {
  color:var(--coast-cyan) !important;
}

/* Buttons: slimmer, cleaner, sunset-to-ocean edge. */
.stButton button, [data-testid="stLinkButton"] a {
  min-height:2.45rem !important;
  border-radius:11px !important;
  background:linear-gradient(135deg,rgba(31,24,70,.90),rgba(8,25,48,.94)) !important;
  border:1px solid rgba(115,132,228,.34) !important;
  box-shadow:none !important;
}
.stButton button:hover, [data-testid="stLinkButton"] a:hover {
  border-color:rgba(255,79,200,.65) !important;
  box-shadow:0 0 14px rgba(255,79,200,.10) !important;
}

/* Charts stay visually dominant. */
[data-testid="stVegaLiteChart"] {
  border-radius:13px !important;
  border-color:rgba(80,118,205,.20) !important;
  box-shadow:0 7px 22px rgba(0,0,0,.17) !important;
}

/* Sidebar: orbital control panel with a subtle tropical-night glow. */
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 0% 92%,rgba(255,79,200,.10),transparent 18rem),
    radial-gradient(circle at 100% 14%,rgba(60,230,255,.06),transparent 16rem),
    linear-gradient(180deg,#050a19,#081027 60%,#040817) !important;
}
.sidebar-brand {
  padding:.72rem .72rem !important;
  border-radius:12px !important;
  background:linear-gradient(135deg,rgba(91,49,174,.13),rgba(7,17,39,.65)) !important;
}

/* A tiny coastal/orbital footer motif, instead of a large icon parade. */
.we-coast-signoff {
  margin:1.15rem 0 .45rem;
  padding:.6rem .75rem;
  border-top:1px solid rgba(93,122,205,.17);
  color:#8496bf;
  font-size:.63rem;
  letter-spacing:.09em;
  text-transform:uppercase;
  display:flex;
  justify-content:space-between;
  align-items:center;
}
.we-coast-signoff .icons {
  letter-spacing:.35rem;
  font-size:.9rem;
  opacity:.72;
}

/* Mobile is the priority. */
@media (max-width:760px) {
  .block-container {
    padding-left:.72rem !important;
    padding-right:.72rem !important;
    padding-top:.30rem !important;
  }
  .we-hero-clean {
    min-height:88px !important;
    height:88px !important;
    padding:.70rem .78rem !important;
    margin-bottom:.58rem !important;
    background-position:73% 49% !important;
  }
  .we-hero-clean .we-brand {
    font-size:1.25rem !important;
    max-width:64%;
  }
  .we-hero-clean .we-brand span {
    font-size:.52rem !important;
    margin-left:.35rem !important;
  }
  .we-hero-clean .we-tag {
    font-size:.55rem !important;
    max-width:61% !important;
    margin-top:.25rem !important;
  }
  .we-mini-status {
    margin-top:.32rem !important;
    font-size:.50rem !important;
  }
  h1 {font-size:1.78rem !important;}
  h2 {font-size:1.48rem !important;}
  h3 {font-size:1.15rem !important;}
  .signal-strip {gap:.38rem !important;}
  .signal-tile {padding:.56rem .60rem !important;}
  [data-testid="stMetricValue"] {font-size:1.34rem !important;}
  [data-testid="stMetricLabel"] p {font-size:.66rem !important;}
}

/* No fake-loading ornament anywhere. */
.we-orbit,
.we-sat,
.we-weather-strip,
.we-satellite-badge {
  display:none !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <div class="we-hero we-hero-clean we-v474-hero">
      <div class="we-v474-copy">
        <div class="we-brand">WEATHEREDGE <span>ORBITAL</span></div>
        <div class="we-v474-edge">PROBABILITY FROM THE EDGE OF SPACE</div>
      </div>
      <div class="we-v474-satellite">🛰️</div>
      <div class="we-v474-sunset"></div>
      <div class="we-v474-palm">🌴</div>
      <div class="we-v474-weather">
        <span>☀️</span><span>🌤️</span><span>🌧️</span><span>⚡</span>
        <span>❄️</span><span>🌬️</span><span>🌴</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown("""
<style>
/* ============================================================
   V474 · NEON COAST ORBITAL
   Cleaner satellite + beach futurism throughout the app.
   ============================================================ */

:root {
  --v474-pink:#ff52bd;
  --v474-coral:#ff6b7f;
  --v474-violet:#9d6cff;
  --v474-cyan:#42dcff;
  --v474-aqua:#56efcf;
  --v474-gold:#ffc857;
  --v474-night:#030716;
  --v474-panel:rgba(7,15,37,.91);
}

/* More restrained star field, richer neon-coast glow. */
.stApp {
  background:
    radial-gradient(ellipse at 5% 20%, rgba(31,172,255,.09), transparent 26rem),
    radial-gradient(ellipse at 94% 12%, rgba(255,71,180,.12), transparent 25rem),
    radial-gradient(ellipse at 82% 76%, rgba(118,74,255,.055), transparent 30rem),
    linear-gradient(168deg,#020611 0%,#07112b 50%,#030817 100%) !important;
}
.stApp:before {
  opacity:.16 !important;
  background-image:
    radial-gradient(circle at 24% 17%, rgba(255,255,255,.8) 0 1px, transparent 1.5px),
    radial-gradient(circle at 74% 28%, rgba(77,218,255,.9) 0 1px, transparent 1.5px),
    radial-gradient(circle at 58% 74%, rgba(255,82,189,.85) 0 1px, transparent 1.5px) !important;
  background-size:340px 340px, 430px 430px, 510px 510px !important;
}

/* ---------- Header ---------- */
.we-v474-hero {
  min-height:158px !important;
  height:158px !important;
  margin:.08rem 0 .85rem !important;
  padding:1.05rem 1.12rem .78rem !important;
  overflow:hidden !important;
  isolation:isolate !important;
  border-radius:19px !important;
  border:1px solid rgba(91,140,236,.30) !important;
  background:
    radial-gradient(circle at 19% 71%, rgba(255,75,176,.19), transparent 7.5rem),
    radial-gradient(circle at 73% 18%, rgba(49,203,255,.10), transparent 8.5rem),
    linear-gradient(132deg,#05091b 0%,#10133a 45%,#08142d 72%,#040a1b 100%) !important;
  box-shadow:
    0 12px 34px rgba(0,0,0,.28),
    0 0 24px rgba(255,82,189,.06),
    inset 0 0 0 1px rgba(66,220,255,.035) !important;
}

/* Wipe every inherited pseudo/background layer from old versions. */
.we-v474-hero:before {
  content:"" !important;
  position:absolute !important;
  inset:0 !important;
  pointer-events:none !important;
  z-index:0 !important;
  opacity:1 !important;
  background:
    linear-gradient(to top, rgba(255,66,167,.065), transparent 34%),
    repeating-linear-gradient(
      to bottom,
      transparent 0 14px,
      rgba(113,95,224,.028) 15px,
      transparent 16px
    ) !important;
}
.we-v474-hero:after {
  content:"" !important;
  position:absolute !important;
  left:0 !important; right:0 !important; bottom:0 !important;
  height:3px !important;
  z-index:5 !important;
  background:linear-gradient(90deg,var(--v474-pink),var(--v474-violet),var(--v474-cyan)) !important;
  box-shadow:0 0 12px rgba(255,82,189,.34),0 0 10px rgba(66,220,255,.22) !important;
}
.we-v474-hero > * {position:relative;z-index:3 !important;}
.we-v474-copy {max-width:68%;}
.we-v474-hero .we-brand {
  max-width:none !important;
  white-space:nowrap !important;
  font-size:1.72rem !important;
  line-height:1 !important;
  letter-spacing:-.025em !important;
  color:white !important;
  text-shadow:0 2px 16px rgba(0,0,0,.6),0 0 14px rgba(147,99,255,.14) !important;
}
.we-v474-hero .we-brand span {
  margin-left:.38rem !important;
  font-size:.64rem !important;
  letter-spacing:.14em !important;
  color:var(--v474-pink) !important;
}
.we-v474-edge {
  margin-top:.55rem;
  color:#ff75d1 !important;
  font-size:.75rem;
  font-weight:900;
  letter-spacing:.115em;
  line-height:1.25;
  text-transform:uppercase;
  text-shadow:0 0 13px rgba(255,82,189,.20);
}
.we-v474-sub {
  margin-top:.30rem;
  color:#b8c4e5 !important;
  font-size:.69rem;
  font-weight:680;
  letter-spacing:.035em;
}
.we-v474-satellite {
  position:absolute !important;
  right:1.05rem !important;
  top:.78rem !important;
  z-index:4 !important;
  font-size:2.18rem;
  transform:rotate(-7deg);
  filter:drop-shadow(0 0 12px rgba(66,220,255,.52));
}
.we-v474-sunset {
  position:absolute !important;
  right:4.1rem !important;
  bottom:-1.25rem !important;
  width:84px;height:84px;border-radius:50%;
  z-index:1 !important;
  background:
    repeating-linear-gradient(
      to bottom,
      rgba(255,102,151,.92) 0 5px,
      rgba(255,198,87,.90) 6px 9px,
      rgba(255,99,176,.86) 10px 13px
    );
  box-shadow:0 0 34px rgba(255,86,171,.20);
  opacity:.56;
}
.we-v474-palm {
  position:absolute !important;
  right:1.0rem !important;
  bottom:1.28rem !important;
  z-index:2 !important;
  font-size:1.55rem;
  opacity:.74;
  filter:drop-shadow(0 0 7px rgba(255,80,180,.22));
}
.we-v474-weather {
  position:absolute !important;
  left:1.08rem !important;
  bottom:.68rem !important;
  display:flex;
  align-items:center;
  gap:.70rem;
  z-index:4 !important;
  font-size:1.02rem;
}
.we-v474-weather span {
  filter:drop-shadow(0 0 5px rgba(77,198,255,.19));
}

/* Permanently suppress legacy header widgets / ghost layers. */
.we-mini-status,
.we-status,
.we-chip,
.we-mission-strip,
.we-weather-dock,
.we-orbit-rail:first-of-type,
.we-satellite-badge,
.we-orbit,
.we-sat,
.we-weather-strip {
  display:none !important;
}

/* ---------- Neon coast language throughout ---------- */
[data-testid="stMetric"],
[data-testid="stExpander"],
.bet-callout,
.quality-card,
.signal-tile {
  background:
    linear-gradient(145deg,rgba(10,18,43,.93),rgba(5,12,30,.94)) !important;
  border:1px solid rgba(91,126,213,.22) !important;
  box-shadow:0 7px 23px rgba(0,0,0,.19) !important;
  backdrop-filter:blur(10px) !important;
}
[data-testid="stMetric"]:nth-of-type(odd),
.signal-tile:nth-child(odd) {
  border-top-color:rgba(255,82,189,.25) !important;
}
[data-testid="stMetric"]:nth-of-type(even),
.signal-tile:nth-child(even) {
  border-top-color:rgba(66,220,255,.24) !important;
}
.bet-callout {
  border-left:2px solid rgba(255,82,189,.58) !important;
  background:
    radial-gradient(circle at 96% 8%,rgba(255,82,189,.105),transparent 8rem),
    linear-gradient(145deg,rgba(21,17,54,.94),rgba(5,14,34,.94)) !important;
}
.quality-card {
  border-left:2px solid rgba(66,220,255,.50) !important;
  background:
    radial-gradient(circle at 92% 12%,rgba(66,220,255,.075),transparent 8rem),
    linear-gradient(145deg,rgba(7,23,49,.94),rgba(13,12,43,.94)) !important;
}
.quality-value {color:var(--v474-pink) !important;}
.quality-grade {color:var(--v474-cyan) !important;}

.section-kicker {
  color:#d7c8ff !important;
  letter-spacing:.16em !important;
}
.section-kicker:before {
  content:"🛰" !important;
  color:var(--v474-cyan) !important;
  margin-right:.45rem;
  filter:drop-shadow(0 0 5px rgba(66,220,255,.30));
}

/* headings get a tiny neon horizon, not a giant decorative block */
h2, h3 {
  text-shadow:0 0 14px rgba(137,103,255,.08);
}
h2:after {
  content:"";
  display:block;
  width:56px;
  height:2px;
  margin-top:.28rem;
  border-radius:99px;
  background:linear-gradient(90deg,var(--v474-pink),var(--v474-cyan));
  opacity:.65;
}

/* Controls */
.stButton button, [data-testid="stLinkButton"] a {
  background:
    linear-gradient(135deg,rgba(32,23,72,.91),rgba(7,25,49,.95)) !important;
  border:1px solid rgba(101,128,221,.34) !important;
  border-radius:11px !important;
  box-shadow:none !important;
}
.stButton button:hover, [data-testid="stLinkButton"] a:hover {
  border-color:rgba(255,82,189,.60) !important;
  box-shadow:0 0 14px rgba(255,82,189,.10) !important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background:rgba(6,15,36,.96) !important;
  border-color:rgba(86,124,215,.26) !important;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
  border-color:rgba(66,220,255,.52) !important;
  box-shadow:0 0 0 1px rgba(255,82,189,.12) !important;
}
[data-testid="stRadio"] label {
  background:linear-gradient(145deg,rgba(8,17,41,.94),rgba(8,14,34,.94)) !important;
  border-color:rgba(91,124,211,.27) !important;
}
[data-testid="stRadio"] label:hover {
  border-color:rgba(255,82,189,.55) !important;
}

/* Charts stay clean and dark. Decorative effects stop at chart border. */
[data-testid="stVegaLiteChart"] {
  background:#080b16 !important;
  border:1px solid rgba(82,113,199,.22) !important;
  border-radius:14px !important;
  box-shadow:0 7px 22px rgba(0,0,0,.17) !important;
}

/* Sidebar: subtle palms-at-night + orbit control panel feel. */
[data-testid="stSidebar"] {
  background:
    radial-gradient(circle at 4% 90%,rgba(255,82,189,.095),transparent 16rem),
    radial-gradient(circle at 98% 10%,rgba(66,220,255,.055),transparent 15rem),
    linear-gradient(180deg,#040918,#081028 57%,#040817) !important;
}
.sidebar-brand {
  border-left:2px solid rgba(255,82,189,.36) !important;
  border-right:1px solid rgba(66,220,255,.16) !important;
}
.sidebar-brand .orb:after {
  content:" 🌴";
  font-size:.72rem;
}

/* Footers get a tiny beach-orbit wink, not a giant banner. */
.we-coast-signoff {
  border-top-color:rgba(94,126,211,.16) !important;
  color:#8397c0 !important;
}
.we-coast-signoff .icons {
  color:#d49bea !important;
}

/* ---------- Mobile ---------- */
@media (max-width:760px) {
  .we-v474-hero {
    min-height:146px !important;
    height:146px !important;
    padding:.92rem .88rem .68rem !important;
    margin-bottom:.70rem !important;
  }
  .we-v474-copy {max-width:73%;}
  .we-v474-hero .we-brand {
    font-size:1.43rem !important;
  }
  .we-v474-hero .we-brand span {
    font-size:.53rem !important;
  }
  .we-v474-edge {
    margin-top:.46rem;
    font-size:.65rem;
    line-height:1.28;
    letter-spacing:.10em;
  }
  .we-v474-sub {
    margin-top:.23rem;
    font-size:.60rem;
  }
  .we-v474-satellite {
    right:.68rem !important;
    top:.68rem !important;
    font-size:1.75rem !important;
  }
  .we-v474-sunset {
    right:2.75rem !important;
    width:72px;height:72px;
  }
  .we-v474-palm {
    right:.58rem !important;
    bottom:1.05rem !important;
    font-size:1.28rem !important;
  }
  .we-v474-weather {
    left:.85rem !important;
    bottom:.54rem !important;
    gap:.53rem !important;
    font-size:.90rem !important;
  }

  /* Make controls slightly denser while keeping tap targets healthy. */
  .stButton button, [data-testid="stLinkButton"] a {
    min-height:2.48rem !important;
  }
  [data-testid="stMetric"] {
    padding:10px 11px !important;
  }
}

/* Nothing decorative is allowed to sit over app content. */
.we-v474-satellite,
.we-v474-sunset,
.we-v474-palm {
  pointer-events:none !important;
}
</style>
""", unsafe_allow_html=True)



# V474 visual refinement: slightly larger premium masthead, with all accidental
# ghost/background copy suppressed. No forecast, market, or ranking logic changes.
st.markdown("""
<style>
/* Give the orbital masthead a little more presence without returning to V468 bulk. */
.we-hero-clean {
  min-height:126px !important;
  height:126px !important;
  padding:1.02rem 1.12rem !important;
  border-radius:18px !important;
  overflow:hidden !important;
  isolation:isolate !important;
  background:
    linear-gradient(90deg,
      rgba(2,6,20,.98) 0%,
      rgba(3,7,23,.90) 39%,
      rgba(5,8,27,.34) 67%,
      rgba(5,8,27,.10) 100%),
    var(--we-hero-image) !important;
  box-shadow:
    0 12px 34px rgba(0,0,0,.28),
    0 0 24px rgba(255,79,200,.08),
    inset 0 0 0 1px rgba(83,221,255,.07) !important;
}

/* Decorative neon horizon only. No text or pseudo-copy. */
.we-hero-clean:before {
  content:"" !important;
  position:absolute !important;
  left:0 !important;
  right:0 !important;
  bottom:0 !important;
  top:auto !important;
  height:34% !important;
  pointer-events:none !important;
  opacity:1 !important;
  background:
    linear-gradient(to top, rgba(255,61,190,.075), transparent 78%) !important;
  z-index:0 !important;
}
.we-hero-clean:after {
  content:"" !important;
  position:absolute !important;
  left:0 !important;
  right:0 !important;
  bottom:0 !important;
  height:3px !important;
  background:linear-gradient(90deg,#ff4fc8 0%,#9c63ff 50%,#3ce6ff 100%) !important;
  box-shadow:0 0 12px rgba(255,79,200,.38),0 0 12px rgba(60,230,255,.24) !important;
  z-index:3 !important;
}

/* Keep every real hero label above imagery and eliminate inherited ghost layers. */
.we-hero-clean > * {
  position:relative !important;
  z-index:4 !important;
}
.we-hero-clean .we-brand {
  font-size:1.72rem !important;
  letter-spacing:-.025em !important;
  text-shadow:0 2px 15px rgba(0,0,0,.58) !important;
}
.we-hero-clean .we-brand span {
  font-size:.68rem !important;
  letter-spacing:.12em !important;
}
.we-hero-clean .we-tag {
  max-width:61% !important;
  margin-top:.42rem !important;
  font-size:.70rem !important;
  line-height:1.34 !important;
  color:#f2c4ec !important;
  text-shadow:0 1px 10px rgba(0,0,0,.72) !important;
}
.we-mini-status {
  margin-top:.55rem !important;
  padding:.18rem .42rem !important;
  width:max-content !important;
  border:1px solid rgba(70,239,184,.18) !important;
  border-radius:999px !important;
  background:rgba(2,10,25,.46) !important;
  backdrop-filter:blur(6px) !important;
}

/* Explicitly remove old V468/V474 decorative text-bearing layers that can
   remain visually underneath the masthead due to broad legacy selectors. */
.we-hero-clean .we-sub,
.we-hero-clean .we-status,
.we-hero-clean .we-chip,
.we-hero-clean .we-satellite-badge,
.we-hero-clean .we-weather-strip,
.we-hero-clean .we-orbit,
.we-hero-clean .we-sat,
.we-hero-clean .we-mission-strip,
.we-hero-clean .we-mission-item,
.we-hero-clean .we-orbit-rail {
  display:none !important;
}

/* Do not allow generic app pseudo-elements to paint through the hero. */
.we-hero-clean p:before,
.we-hero-clean p:after,
.we-hero-clean div:not(.we-mini-status):not(.we-brand):not(.we-tag):before,
.we-hero-clean div:not(.we-mini-status):not(.we-brand):not(.we-tag):after {
  content:none !important;
}

/* Mobile: ~110px, enough to feel designed but still compact. */
@media (max-width:760px) {
  .we-hero-clean {
    min-height:110px !important;
    height:110px !important;
    padding:.82rem .88rem !important;
    background-position:72% 48% !important;
  }
  .we-hero-clean .we-brand {
    font-size:1.43rem !important;
    max-width:68% !important;
  }
  .we-hero-clean .we-brand span {
    font-size:.56rem !important;
  }
  .we-hero-clean .we-tag {
    max-width:62% !important;
    font-size:.60rem !important;
    margin-top:.31rem !important;
  }
  .we-mini-status {
    margin-top:.38rem !important;
    font-size:.50rem !important;
  }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ============================================================
   V474 · ORBITAL NEON COAST
   Real photography is used as low-opacity atmosphere, never
   behind dense data text or inside the plotting area.
   ============================================================ */

/* Requested header cleanup */
.we-v474-sub { display:none !important; }

/* Real Earth / orbital photography in the hero. */
.we-v474-hero {
  background:
    linear-gradient(90deg,rgba(2,6,20,.94) 0%,rgba(5,7,28,.78) 46%,rgba(4,8,24,.48) 100%),
    url("https://images-assets.nasa.gov/image/PIA00123/PIA00123~orig.jpg") center 46% / cover no-repeat !important;
}
.we-v474-hero:before {
  background:
    radial-gradient(circle at 18% 76%,rgba(255,70,182,.17),transparent 7rem),
    linear-gradient(to top,rgba(255,55,177,.08),transparent 38%) !important;
}

/* A photographic neon-coast wash across the page margins.
   Dense content retains opaque/glass panels for readability. */
.stApp {
  background:
    linear-gradient(rgba(3,7,21,.93),rgba(3,8,24,.96)),
    url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1800&q=72")
      center top / cover fixed no-repeat !important;
}

/* Section atmosphere: a tiny photographic strip, kept outside text. */
h2 {
  position:relative;
  overflow:visible;
}
h2:before {
  content:"";
  display:inline-block;
  width:22px;
  height:22px;
  margin-right:8px;
  vertical-align:-4px;
  border-radius:7px;
  border:1px solid rgba(91,219,255,.30);
  background:
    linear-gradient(145deg,rgba(255,65,183,.18),rgba(24,45,98,.16)),
    url("https://images-assets.nasa.gov/image/PIA00123/PIA00123~orig.jpg")
      center / cover no-repeat;
  box-shadow:0 0 12px rgba(66,220,255,.10);
}

/* Glass cards with tiny real-photo edge treatments. */
.bet-callout,
.quality-card,
[data-testid="stMetric"],
[data-testid="stExpander"] {
  position:relative;
  overflow:hidden;
}
.bet-callout:after,
.quality-card:after {
  content:"";
  position:absolute;
  right:0;
  top:0;
  bottom:0;
  width:7px;
  pointer-events:none;
  background:
    linear-gradient(rgba(255,74,186,.20),rgba(53,218,255,.18)),
    url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=400&q=60")
      center / cover no-repeat;
  opacity:.75;
}

/* Neon horizon dividers. */
hr {
  border:0 !important;
  height:1px !important;
  background:linear-gradient(90deg,transparent,rgba(255,72,186,.48),rgba(64,218,255,.42),transparent) !important;
  box-shadow:0 0 10px rgba(255,72,186,.08);
}

/* Sidebar gets a real Earth limb at the very top, then returns to clean glass. */
[data-testid="stSidebar"] > div:first-child {
  background:
    linear-gradient(rgba(4,9,24,.90),rgba(4,9,24,.98) 220px),
    url("https://images-assets.nasa.gov/image/PIA00123/PIA00123~orig.jpg")
      center top / 100% auto no-repeat !important;
}

/* Empty-state diagnostics */
.filter-readout {
  margin:.7rem 0 1rem;
  padding:.85rem .9rem;
  border-radius:14px;
  border:1px solid rgba(97,134,221,.28);
  background:
    radial-gradient(circle at 96% 0%,rgba(255,73,183,.10),transparent 8rem),
    rgba(6,15,38,.93);
}
.filter-readout-title {
  color:#ff78cf;
  font-size:.72rem;
  font-weight:900;
  letter-spacing:.14em;
  margin-bottom:.55rem;
}
.filter-readout-grid {
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:.45rem .8rem;
  color:#b9c7e7;
  font-size:.82rem;
}
.filter-readout-grid b { color:#fff; }
.scan-total {
  margin:.35rem 0 .55rem;
  color:#c9d5ef;
  font-size:.88rem;
}
.filter-fail-row {
  display:grid;
  grid-template-columns:minmax(0,1fr) auto 64px;
  gap:.65rem;
  align-items:center;
  padding:.48rem .62rem;
  margin:.28rem 0;
  border-radius:9px;
  border:1px solid rgba(83,116,195,.18);
  background:rgba(5,13,33,.76);
  color:#bdc9e4;
  font-size:.80rem;
}
.filter-fail-row strong { color:#ff77c8; font-size:.95rem; }
.filter-fail-row small { color:#7186ae; text-align:right; }

.near-miss-card {
  position:relative;
  overflow:hidden;
  margin:.55rem 0;
  padding:.82rem .9rem .86rem;
  border-radius:14px;
  border:1px solid rgba(255,92,185,.24);
  border-left:2px solid #ff62bd;
  background:
    radial-gradient(circle at 98% 0%,rgba(66,220,255,.09),transparent 8rem),
    linear-gradient(145deg,rgba(17,15,49,.96),rgba(5,15,34,.96));
}
.near-miss-card:after {
  content:"🛰️  🌴";
  position:absolute;
  right:.72rem;
  top:.64rem;
  opacity:.38;
  font-size:.88rem;
}
.near-miss-tag {
  color:#ff72c9;
  font-size:.65rem;
  font-weight:900;
  letter-spacing:.10em;
}
.near-miss-title {
  margin-top:.34rem;
  color:#fff;
  font-weight:850;
  font-size:1rem;
}
.near-miss-contract {
  color:#c6d1ea;
  font-size:.83rem;
  margin-top:.12rem;
}
.near-miss-numbers {
  color:#63dfff;
  font-size:.78rem;
  font-weight:750;
  margin-top:.42rem;
}
.near-miss-reason {
  color:#98a8ca;
  font-size:.75rem;
  line-height:1.38;
  margin-top:.32rem;
}

/* Keep data itself pristine: no photos behind charts/tables. */
[data-testid="stVegaLiteChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  background:#070b17 !important;
}

/* Mobile density */
@media(max-width:760px) {
  .filter-readout-grid { grid-template-columns:1fr; }
  .filter-fail-row {
    grid-template-columns:minmax(0,1fr) auto 58px;
    font-size:.76rem;
  }
  .we-v474-hero {
    min-height:152px !important;
    height:152px !important;
  }
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* V474 utility polish */
.city-local-clock {
  display:inline-flex;
  align-items:center;
  gap:.55rem;
  margin:.42rem 0 .78rem;
  padding:.46rem .66rem;
  border-radius:12px;
  border:1px solid rgba(66,220,255,.24);
  background:
    radial-gradient(circle at 0% 50%,rgba(255,82,189,.09),transparent 5rem),
    rgba(6,15,36,.82);
  box-shadow:0 0 14px rgba(66,220,255,.035);
}
.city-local-clock > span {
  color:#42dcff !important;
  font-size:1rem;
}
.city-local-clock div {
  display:flex;
  align-items:baseline;
  gap:.48rem;
}
.city-local-clock small {
  color:#8ea4ce !important;
  font-size:.61rem !important;
  font-weight:850;
  letter-spacing:.10em;
}
.city-local-clock strong {
  color:#fff !important;
  font-size:.88rem;
  white-space:nowrap;
}
@media(max-width:760px) {
  .city-local-clock {
    display:flex;
    width:max-content;
    max-width:100%;
  }
  .city-local-clock div {
    gap:.38rem;
  }
  .city-local-clock small {
    font-size:.55rem !important;
  }
  .city-local-clock strong {
    font-size:.80rem;
  }
}
</style>
""", unsafe_allow_html=True)

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
if "bet_date_filter_setting" not in st.session_state:
    st.session_state.bet_date_filter_setting = "Both dates"

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


        bet_date_filter = st.radio(
            "Bet date",
            ["Both dates", "Today only", "Tomorrow only"],
            index=["Both dates", "Today only", "Tomorrow only"].index(
                st.session_state.bet_date_filter_setting
                if st.session_state.bet_date_filter_setting
                in ("Both dates", "Today only", "Tomorrow only")
                else "Both dates"
            ),
            key="bet_date_filter_setting",
            help=(
                "Restrict the Best Bets candidate pool before ranking. "
                "Both dates preserves the existing behavior."
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

# Best Bet date filter applies to the actual candidate universe BEFORE ranking,
# diagnostics, and Top-N selection. City Explorer still shows every available date.
ranking_df = df.copy()
ranking_date_note = "Both available market dates"

if "date" in ranking_df.columns and not ranking_df.empty:
    ranking_df["date"] = pd.to_datetime(
        ranking_df["date"], errors="coerce"
    ).dt.date

    # Resolve Today/Tomorrow by each row's city-local date. This avoids UTC
    # midnight incorrectly moving western US markets into the wrong bucket.
    local_today_by_city = {}
    for _city in ranking_df["city"].dropna().unique():
        _cfg = PRESETS.get(_city, {})
        _tz_name = _cfg.get("tz", "UTC")
        try:
            local_today_by_city[_city] = datetime.now(
                ZoneInfo(_tz_name)
            ).date()
        except Exception:
            local_today_by_city[_city] = datetime.now().date()

    if bet_date_filter == "Today only":
        _mask = ranking_df.apply(
            lambda r: r.get("date") == local_today_by_city.get(
                r.get("city"), datetime.now().date()
            ),
            axis=1,
        )
        ranking_df = ranking_df[_mask].copy()
        ranking_date_note = "Today only"
    elif bet_date_filter == "Tomorrow only":
        _mask = ranking_df.apply(
            lambda r: r.get("date") == (
                local_today_by_city.get(
                    r.get("city"), datetime.now().date()
                ) + timedelta(days=1)
            ),
            axis=1,
        )
        ranking_df = ranking_df[_mask].copy()
        ranking_date_note = "Tomorrow only"



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


ranking_ranking_df["safety_prob_lower"] = ranking_df.apply(conservative_probability_lower_bound, axis=1)
ranking_ranking_df["safety_edge_lower"] = ranking_ranking_df["safety_prob_lower"] - ranking_df["ask"]
ranking_ranking_df["model_disagreement_f"] = ranking_df.apply(model_disagreement_f, axis=1)

# Keep ranking schema stable even when a date filter yields zero rows.
for _ranking_col, _ranking_default in (
    ("safety_prob_lower", float("nan")),
    ("safety_edge_lower", float("nan")),
    ("model_disagreement_f", float("nan")),
):
    if _ranking_col not in ranking_df.columns:
        ranking_df[_ranking_col] = _ranking_default

rounding_safe_mask = ~ranking_df["settlement_rounding_risk"].fillna(False).astype(bool)

# User-adjustable outlier-safety rule: exclude sides whose live Kalshi ask
# is at or below the selected minimum market probability.
kalshi_market_floor_mask = (
    ranking_df["ask"].notna()
    & (ranking_df["ask"] >= min_market_price)
)

historical_depth_mask = (
    pd.to_numeric(ranking_df["nws_sigma_samples"], errors="coerce")
    .fillna(0)
    .astype(int)
    >= 8
)

trajectory_gap_numeric = pd.to_numeric(
    ranking_df["trajectory_current_gap_f"], errors="coerce"
)
hours_to_peak_numeric = pd.to_numeric(
    ranking_df["hours_to_expected_peak"], errors="coerce"
).fillna(99.0)

trajectory_conflict_mask = ~(
    trajectory_gap_numeric.notna()
    & (hours_to_peak_numeric <= 3.0)
    & (trajectory_gap_numeric.abs() >= 1.5)
)

model_agreement_mask = ranking_ranking_df["model_disagreement_f"].fillna(0.0) <= 4.0

qualified_all = ranking_df[
    (ranking_df["nws_support"] == True)
    & (ranking_df["conservative_prob"] >= min_nws_chance)
    & (ranking_df["conservative_edge"] >= min_gap)
    & (ranking_df["safety_edge_lower"] >= max(min_gap, 0.03))
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


def top_bet_failure_reasons(row):
    """Return every active Top Bet safety rule this candidate currently fails."""
    reasons = []

    if not bool(row.get("nws_support", False)):
        reasons.append("NWS does not support this side")

    try:
        if float(row.get("conservative_prob", 0.0)) < min_nws_chance:
            reasons.append(
                f"NWS-based chance below {min_nws_chance*100:.0f}% minimum"
            )
    except (TypeError, ValueError):
        reasons.append("NWS-based chance unavailable")

    try:
        if float(row.get("conservative_edge", -1.0)) < min_gap:
            reasons.append(
                f"Weather Edge below {min_gap*100:.0f} pp minimum"
            )
    except (TypeError, ValueError):
        reasons.append("Weather Edge unavailable")

    try:
        ask = float(row.get("ask", 0.0) or 0.0)
        if ask < min_market_price:
            reasons.append(
                f"market probability below {min_market_price*100:.0f}% minimum"
            )
    except (TypeError, ValueError):
        reasons.append("market probability unavailable")

    try:
        if int(row.get("nws_sigma_samples", 0) or 0) < 8:
            reasons.append("fewer than 8 historical calibration samples")
    except (TypeError, ValueError):
        reasons.append("historical calibration depth unavailable")

    try:
        if float(row.get("safety_edge_lower", -1.0) or -1.0) < max(min_gap, 0.03):
            reasons.append("uncertainty-adjusted edge below safety minimum")
    except (TypeError, ValueError):
        reasons.append("uncertainty-adjusted edge unavailable")

    try:
        traj_gap = row.get("trajectory_current_gap_f")
        hours_peak = row.get("hours_to_expected_peak", 99)
        if (
            traj_gap is not None
            and not pd.isna(traj_gap)
            and float(hours_peak if hours_peak is not None else 99) <= 3.0
            and abs(float(traj_gap)) >= 1.5
        ):
            reasons.append("live trajectory conflicts with forecast near peak")
    except (TypeError, ValueError):
        pass

    try:
        if float(row.get("model_disagreement_f", 0.0) or 0.0) > 4.0:
            reasons.append("model disagreement exceeds 4°F")
    except (TypeError, ValueError):
        pass

    if bool(row.get("settlement_rounding_risk", False)):
        reasons.append("settlement / rounding protection")

    return reasons


def build_top_bet_diagnostics(frame):
    """
    Build an independent failure audit plus the nearest misses.

    Failure counts are intentionally independent: one candidate can fail more
    than one safety check. That is more useful diagnostically than forcing each
    candidate into a single arbitrary bucket.
    """
    if frame is None or frame.empty:
        return {}, pd.DataFrame()

    audit_labels = [
        ("NWS support", lambda r: bool(r.get("nws_support", False))),
        (
            f"NWS chance ≥ {min_nws_chance*100:.0f}%",
            lambda r: float(r.get("conservative_prob", 0.0) or 0.0) >= min_nws_chance,
        ),
        (
            f"Weather Edge ≥ {min_gap*100:.0f} pp",
            lambda r: float(r.get("conservative_edge", -1.0) or -1.0) >= min_gap,
        ),
        (
            f"Market probability ≥ {min_market_price*100:.0f}%",
            lambda r: float(r.get("ask", 0.0) or 0.0) >= min_market_price,
        ),
        (
            "Historical calibration ≥ 8 samples",
            lambda r: int(r.get("nws_sigma_samples", 0) or 0) >= 8,
        ),
        (
            f"Uncertainty-adjusted edge ≥ {max(min_gap, 0.03)*100:.0f} pp",
            lambda r: float(r.get("safety_edge_lower", -1.0) or -1.0)
            >= max(min_gap, 0.03),
        ),
        (
            "Near-peak trajectory agreement",
            lambda r: not (
                r.get("trajectory_current_gap_f") is not None
                and not pd.isna(r.get("trajectory_current_gap_f"))
                and float(r.get("hours_to_expected_peak", 99) or 99) <= 3.0
                and abs(float(r.get("trajectory_current_gap_f"))) >= 1.5
            ),
        ),
        (
            "Model disagreement ≤ 4°F",
            lambda r: float(r.get("model_disagreement_f", 0.0) or 0.0) <= 4.0,
        ),
        (
            "Settlement / rounding safety",
            lambda r: not bool(r.get("settlement_rounding_risk", False)),
        ),
    ]

    counts = {}
    for label, check in audit_labels:
        failures = 0
        for _, row in frame.iterrows():
            try:
                passed = bool(check(row))
            except Exception:
                passed = False
            if not passed:
                failures += 1
        counts[label] = failures

    near = frame.copy()
    near["_failure_reasons"] = near.apply(top_bet_failure_reasons, axis=1)
    near["_failure_count"] = near["_failure_reasons"].apply(len)
    near = near[near["_failure_count"] > 0].copy()
    if not near.empty:
        near["_quality_sort"] = pd.to_numeric(
            near.get("bet_quality_score"), errors="coerce"
        ).fillna(-999)
        near["_edge_sort"] = pd.to_numeric(
            near.get("conservative_edge"), errors="coerce"
        ).fillna(-999)
        near = near.sort_values(
            ["_failure_count", "_quality_sort", "_edge_sort"],
            ascending=[True, False, False],
        ).head(3)

    return counts, near


top_bet_failure_counts, closest_top_bet_misses = build_top_bet_diagnostics(ranking_df)


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
            f"No candidates currently pass every Top Bet safety filter "
            f"for {ranking_date_note.lower()}. WeatherEdge is keeping the safety "
            "rules intact rather than silently loosening them."
        )

        st.markdown(
            f"""
            <div class="filter-readout">
              <div class="filter-readout-title">ACTIVE SAFETY SETTINGS</div>
              <div class="filter-readout-grid">
                <span>NWS chance <b>≥ {min_nws_chance*100:.0f}%</b></span>
                <span>Weather Edge <b>≥ {min_gap*100:.0f} pp</b></span>
                <span>Market probability <b>≥ {min_market_price*100:.0f}%</b></span>
                <span>Uncertainty edge <b>≥ {max(min_gap, 0.03)*100:.0f} pp</b></span>
                <span>Calibration <b>≥ 8 samples</b></span>
                <span>Model spread <b>≤ 4°F</b></span>
                <span>Market date <b>{ranking_date_note}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Why candidates were filtered out")
        st.caption(
            "Counts below are independent. A candidate can fail more than one rule."
        )
        total_candidates = len(ranking_df)
        st.markdown(
            f"<div class='scan-total'>Scanned <b>{total_candidates}</b> candidate sides</div>",
            unsafe_allow_html=True,
        )
        for label, count in top_bet_failure_counts.items():
            pct = (count / total_candidates * 100.0) if total_candidates else 0.0
            st.markdown(
                f"<div class='filter-fail-row'>"
                f"<span>{label}</span>"
                f"<strong>{count}</strong>"
                f"<small>{pct:.0f}% failed</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if not closest_top_bet_misses.empty:
            st.markdown("#### Closest candidates")
            st.caption(
                "Diagnostic only. These are NOT RECOMMENDED bets; they are shown "
                "so you can see exactly what almost qualified."
            )
            for _, miss in closest_top_bet_misses.iterrows():
                miss_date = miss.get("date_label") or str(miss.get("date") or "—")
                reasons = miss.get("_failure_reasons") or []
                reason_text = " · ".join(reasons)
                ask = float(miss.get("ask", 0.0) or 0.0) * 100.0
                prob = float(miss.get("conservative_prob", 0.0) or 0.0) * 100.0
                edge = float(miss.get("conservative_edge", 0.0) or 0.0) * 100.0
                st.markdown(
                    f"""
                    <div class="near-miss-card">
                      <div class="near-miss-tag">NOT RECOMMENDED · NEAREST MISS</div>
                      <div class="near-miss-title">{miss.get('city','—')} · {miss_date}</div>
                      <div class="near-miss-contract">{miss.get('market_subtitle','—')} · {miss.get('side','—')}</div>
                      <div class="near-miss-numbers">
                        Model {prob:.0f}% · Market {ask:.0f}% · Edge {edge:+.0f} pp
                      </div>
                      <div class="near-miss-reason">{reason_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
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

        # Local clock for the selected city. Useful for judging how much of the
        # heating window is actually left without mentally converting time zones.
        try:
            _city_now = datetime.now(ZoneInfo(PRESETS[city]["tz"]))
            _city_time_text = _city_now.strftime("%-I:%M %p %Z")
        except Exception:
            _city_time_text = "—"
        st.markdown(
            f"<div class='city-local-clock'>"
            f"<span>◷</span><div><small>CURRENT TIME IN LOCATION</small>"
            f"<strong>{_city_time_text}</strong></div></div>",
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
