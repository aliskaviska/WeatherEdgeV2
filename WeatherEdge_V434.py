
import math
import os
import re
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

def get_json(url, params=None, timeout=25):
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

@st.cache_data(ttl=180, show_spinner=False)
def get_kalshi_markets(series_ticker):
    out, cursor = [], None
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
    return out

@st.cache_data(ttl=3600)
def get_series_info(series_ticker):
    data = get_json(f"{KALSHI_BASE}/series/{series_ticker}")
    return data.get("series", {})

@st.cache_data(ttl=300, show_spinner=False)
def get_event(event_ticker):
    if not event_ticker:
        return {}
    data = get_json(f"{KALSHI_BASE}/events/{event_ticker}")
    return data.get("event", {})

@st.cache_data(ttl=900)
def get_nws_daily(lat, lon):
    point = get_json(f"https://api.weather.gov/points/{lat},{lon}")
    forecast_url = point["properties"]["forecast"]
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



def rounding_risk_distance_f(observed_high, kind, lo, hi):
    """
    Distance from the observed daily high to the nearest contract strike/threshold.

    Kalshi temperature contracts are expressed in whole-degree buckets/thresholds,
    while live observations can arrive with decimals. If the observed high is very
    close to one of those whole-degree strike values, tiny reporting/rounding
    differences can flip how the market ultimately settles.

    Returns the absolute distance in °F to the nearest relevant strike, or None.
    """
    if observed_high is None or pd.isna(observed_high):
        return None

    t = float(observed_high)
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
        return None

    return min(abs(t - strike) for strike in strikes)


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
    trajectory_sigma_multiplier=1.0
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
        float(nws_high)
        + nws_bias_f(calibration_lead, calibration)
        + float(trajectory_adjustment_f or 0.0)
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
def build_nws_error_calibration(city, station_id, tz_name):
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
    rows, err = get_snapshot_rows(city, None)
    snap, norm_err = normalize_snapshot_rows(rows)
    if err or norm_err or snap.empty or not station_id:
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
    return calibration


def _clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0





def recent_completed_day_adjustment(contract_date, hours_to_peak, calibration):
    """Strong recency-weighted correction from recent completed forecast misses."""
    neutral = {
        "adjustment_f": 0.0,
        "sigma_multiplier": 1.0,
        "weighted_error_f": 0.0,
        "n_days": 0,
        "consistency": 0.0,
        "yesterday_error_f": None,
    }
    try:
        history = (calibration or {}).get("__meta__", {}).get("recent_daily_errors", []) or []
        parsed = []
        for item in history:
            try:
                d = datetime.fromisoformat(str(item["date"])).date()
                days_back = (contract_date - d).days
                if 1 <= days_back <= 5:
                    parsed.append((days_back, float(item["error_f"])))
            except Exception:
                continue
        if not parsed:
            return neutral

        parsed.sort(key=lambda x: x[0])
        day_weights = {1: 1.00, 2: 0.60, 3: 0.35, 4: 0.20, 5: 0.12}
        weights = [day_weights[d] for d, _ in parsed]
        errors = [e for _, e in parsed]
        wsum = sum(weights)
        weighted_error = sum(w * e for w, e in zip(weights, errors)) / max(wsum, 1e-9)

        meaningful = [e for e in errors if abs(e) >= 0.75]
        if meaningful:
            pos = sum(1 for e in meaningful if e > 0)
            neg = sum(1 for e in meaningful if e < 0)
            consistency = max(pos, neg) / len(meaningful)
        else:
            consistency = 0.0

        yesterday = next((e for d, e in parsed if d == 1), None)
        n = len(parsed)

        strength = 0.55 if n == 1 else (0.72 if n == 2 else 0.88)
        if consistency >= 0.80:
            strength += 0.10
        elif consistency < 0.60:
            strength *= 0.55

        htp = float(hours_to_peak) if hours_to_peak is not None else 6.0
        peak_relevance = 1.0 if htp <= 2 else (0.90 if htp <= 6 else 0.75)
        strength = min(0.98, strength * peak_relevance)
        adjustment = weighted_error * strength

        # A large miss yesterday should have a large minimum effect today.
        if yesterday is not None and abs(yesterday) >= 3.0:
            minimum = min(3.5, abs(yesterday) * 0.65)
            if consistency >= 0.60 or n == 1:
                adjustment = math.copysign(max(abs(adjustment), minimum), yesterday)

        adjustment = max(-6.0, min(6.0, adjustment))

        recent_mae = sum(w * abs(e) for w, e in zip(weights, errors)) / max(wsum, 1e-9)
        if consistency < 0.60 and recent_mae >= 1.5:
            sigma_multiplier = min(1.45, 1.10 + 0.08 * recent_mae)
        elif recent_mae >= 3.0:
            sigma_multiplier = min(1.30, 1.04 + 0.05 * recent_mae)
        elif recent_mae <= 1.0 and n >= 2:
            sigma_multiplier = 0.92
        else:
            sigma_multiplier = 1.0

        return {
            "adjustment_f": float(adjustment),
            "sigma_multiplier": float(sigma_multiplier),
            "weighted_error_f": float(weighted_error),
            "n_days": int(n),
            "consistency": float(consistency),
            "yesterday_error_f": None if yesterday is None else float(yesterday),
        }
    except Exception:
        return neutral


def forecast_evolution_adjustment(city, cfg, contract_date, hours_to_peak, calibration):
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

        rows, err = get_snapshot_rows(city, contract_date)
        snap, norm_err = normalize_snapshot_rows(rows)
        if err or norm_err or snap.empty:
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


def live_trajectory_adjustment(city, cfg, contract_date, hours_to_peak):
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
    }
    try:
        tz = ZoneInfo(cfg["tz"])
        now_local = pd.Timestamp.now(tz=tz)

        # Only condition same-day markets on live trajectory.
        if now_local.date() != contract_date:
            return neutral

        rows, err = get_snapshot_rows(city, contract_date)
        if err or not rows:
            return neutral
        snap, norm_err = normalize_snapshot_rows(rows)
        if norm_err or snap.empty:
            return neutral

        obs = get_station_observations(
            cfg.get("station_id"), cfg["tz"], contract_date
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
        }
    except Exception:
        return neutral


def trajectory_agreement_score(city, cfg, contract_date, nws_high):
    """
    Score recent observed-vs-NWS trajectory agreement from 0 to 1.

    Uses the latest stored NWS projection for hours that have already occurred
    today and compares it with observed station temperatures at nearby times.
    Returns a neutral 0.5 when there is not enough overlapping data.
    """
    try:
        rows, err = get_snapshot_rows(city, contract_date)
        if err or not rows:
            return 0.5
        snap, norm_err = normalize_snapshot_rows(rows)
        if norm_err or snap.empty:
            return 0.5

        obs = get_station_observations(cfg.get("station_id"), cfg["tz"], contract_date)
        if obs is None or obs.empty:
            return 0.5

        tz = ZoneInfo(cfg["tz"])
        now_local = pd.Timestamp.now(tz=tz)
        latest_key = snap["snapshot_key"].max()
        fc = snap[snap["snapshot_key"] == latest_key].copy()
        fc["time_local"] = fc["valid_at"].dt.tz_convert(tz)
        fc = fc[
            (fc["time_local"].dt.date == contract_date)
            & (fc["time_local"] <= now_local)
        ][["time_local", "temp_f"]].dropna().sort_values("time_local")
        if fc.empty:
            return 0.5

        ob = obs.copy()
        ob["time_local"] = pd.to_datetime(ob["time"], utc=True, errors="coerce").dt.tz_convert(tz)
        ob = ob[
            (ob["time_local"].dt.date == contract_date)
            & (ob["time_local"] <= now_local)
        ][["time_local", "temp_f"]].dropna().sort_values("time_local")
        if ob.empty:
            return 0.5

        # Match each forecast point to nearest observation within 75 minutes.
        merged = pd.merge_asof(
            fc.sort_values("time_local"),
            ob.sort_values("time_local"),
            on="time_local",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=75),
            suffixes=("_fc", "_obs"),
        ).dropna(subset=["temp_f_fc", "temp_f_obs"])

        if len(merged) < 2:
            return 0.5

        mae = float((merged["temp_f_obs"] - merged["temp_f_fc"]).abs().mean())

        # <=0.5°F is excellent; >=4°F is poor.
        mae_component = _clamp01(1.0 - (mae - 0.5) / 3.5)

        # Compare recent directional trend where possible.
        if len(merged) >= 3:
            fc_delta = float(merged["temp_f_fc"].iloc[-1] - merged["temp_f_fc"].iloc[-3])
            obs_delta = float(merged["temp_f_obs"].iloc[-1] - merged["temp_f_obs"].iloc[-3])
            if abs(fc_delta) < 0.4 and abs(obs_delta) < 0.4:
                trend_component = 1.0
            elif fc_delta == 0 or obs_delta == 0:
                trend_component = 0.7
            else:
                trend_component = 1.0 if (fc_delta > 0) == (obs_delta > 0) else 0.25
        else:
            trend_component = 0.6

        return _clamp01(0.75 * mae_component + 0.25 * trend_component)
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


def build_city_rows(city, cfg):
    calibration = build_nws_error_calibration(
        city, cfg.get("station_id"), cfg["tz"]
    )
    markets = get_kalshi_markets(cfg["series"])
    if not markets:
        return []

    series_info = get_series_info(cfg["series"])
    nws = {r["date"]: r for r in get_nws_daily(cfg["lat"], cfg["lon"])}

    # GFS is retained strictly for the visual context chart. It does not enter
    # NWS probabilities, edge calculations, qualification, or ranking.
    try:
        ens = get_gfs_ensemble_daily_highs(cfg["lat"], cfg["lon"], cfg["tz"])
    except Exception:
        ens = None

    markets_by_event = {}
    for market in markets:
        markets_by_event.setdefault(market.get("event_ticker"), []).append(market)
    implied_by_event = {
        event_ticker: market_implied_temperature(group)
        for event_ticker, group in markets_by_event.items()
    }

    event_cache = {}
    rows = []
    live_context_cache = {}

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

        if d not in live_context_cache:
            live_context_cache[d] = {
                "trajectory_score": trajectory_agreement_score(city, cfg, d, nws_high),
                "trajectory_live": live_trajectory_adjustment(
                    city, cfg, d, hours_to_peak
                ),
                "forecast_evolution": forecast_evolution_adjustment(
                    city, cfg, d, hours_to_peak, calibration
                ),
                "recent_days": recent_completed_day_adjustment(
                    d, hours_to_peak, calibration
                ),
            }
        trajectory_score = live_context_cache[d]["trajectory_score"]
        trajectory_live = live_context_cache[d]["trajectory_live"]
        forecast_evolution = live_context_cache[d]["forecast_evolution"]
        recent_days = live_context_cache[d]["recent_days"]

        observed_high = None
        observed_high_time = None
        previous_day_high = None
        previous_day_high_time = None
        previous_3day_avg_high = None
        previous_day_prediction = None
        try:
            observed_high, observed_high_time = observed_daily_high_details(
                cfg.get("station_id"), cfg["tz"], d
            )
            previous_day_high, previous_day_high_time, previous_3day_avg_high = recent_observed_high_summary(
                cfg.get("station_id"), cfg["tz"], d
            )
            previous_day_prediction = previous_day_forecast_summary(
                city, cfg["tz"], d
            )
        except Exception:
            observed_high = None
            observed_high_time = None
            previous_day_high = None
            previous_day_high_time = None
            previous_3day_avg_high = None
            previous_day_prediction = None

        hours_left = hours_to_deadline(m, cfg["tz"])
        p_yes = nws_yes_probability(
            nws_high, kind, lo, hi,
            hours_left=hours_left,
            observed_high=observed_high,
            calibration=calibration,
            hours_to_peak=hours_to_peak,
            trajectory_adjustment_f=(
                trajectory_live["adjustment_f"]
                + forecast_evolution["adjustment_f"]
                + recent_days["adjustment_f"]
            ),
            trajectory_sigma_multiplier=(
                trajectory_live["sigma_multiplier"]
                * forecast_evolution["sigma_multiplier"]
                * recent_days["sigma_multiplier"]
            ),
        )
        if p_yes is None:
            continue

        # Optional GFS reference values for the chart only.
        ensemble_median = ensemble_low = ensemble_high = None
        ensemble_daily_highs = []
        if ens is not None and d in ens.index:
            daily_members = pd.Series(ens.loc[d].values).dropna().astype(float)
            if observed_high is not None and not daily_members.empty:
                daily_members = daily_members.clip(lower=float(observed_high))
            if not daily_members.empty:
                ensemble_daily_highs = [float(v) for v in daily_members.tolist()]
                ensemble_median = float(daily_members.median())
                ensemble_low = float(daily_members.quantile(0.10))
                ensemble_high = float(daily_members.quantile(0.90))

        event_ticker = m.get("event_ticker")
        if event_ticker not in event_cache:
            try:
                event_cache[event_ticker] = get_event(event_ticker)
            except Exception:
                event_cache[event_ticker] = {}
        event_info = event_cache[event_ticker]

        title = event_info.get("title") or m.get("title") or series_info.get("title") or f"{city} high temperature"
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or bracket
        settlement = source_names(series_info, event_info)
        contract_url = series_info.get("contract_url")
        implied_temp = implied_by_event.get(event_ticker)
        temp_gap = float(nws_high) - implied_temp if implied_temp is not None else None
        adjusted_nws_center = (
            float(nws_high)
            + nws_bias_f(hours_to_peak, calibration)
            + float(trajectory_live["adjustment_f"] or 0.0)
            + float(forecast_evolution["adjustment_f"] or 0.0)
            + float(recent_days["adjustment_f"] or 0.0)
        )
        if observed_high is not None and not pd.isna(observed_high):
            adjusted_nws_center = max(adjusted_nws_center, float(observed_high))
        nws_support_yes = point_forecast_supports_yes(
            adjusted_nws_center, kind, lo, hi
        )
        rounding_distance = rounding_risk_distance_f(
            observed_high, kind, lo, hi
        )

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
                "observed_data_url": nws_climate_url(cfg),
                "nws_forecast": nrow.get("nws_detail"),
                "nws_forecast_url": nrow.get("nws_forecast_url"),
                "nws_sigma_f": (
                    nws_sigma_f(hours_to_peak, calibration)
                    * trajectory_live["sigma_multiplier"]
                    * forecast_evolution["sigma_multiplier"]
                    * recent_days["sigma_multiplier"]
                ),
                "nws_bias_f": nws_bias_f(hours_to_peak, calibration),
                # Canonical WeatherEdge final-high center. The daily high can never
                # finish below a temperature that has already been observed.
                "adjusted_nws_center_f": adjusted_nws_center,
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



@st.cache_data(ttl=300, show_spinner=False)
def scan_live_market_universe():
    """
    Refresh all supported cities concurrently, then cache the combined result
    for five minutes. Network-bound city scans benefit substantially from a
    modest thread pool.
    """
    rows = []
    errors = []
    max_workers = min(8, max(1, len(PRESETS)))

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
    return rows, errors, refreshed_at




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
        "get_event",
        "get_nws_daily",
        "get_station_observations",
        "get_observed_high_so_far",
        "get_gfs_ensemble_daily_highs",
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
    # Pull the city's stored history, then let latest_projection_chart select the exact
    # valid-time display window. This is what makes the purple NWS line extend backward.
    params = {
        "select": "*",
        "city": f"eq.{city}",
        "order": "id.asc",
    }

    rows = []
    page_size = 1000
    for start in range(0, 30000, page_size):
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
    return out, None


@st.cache_data(ttl=180, show_spinner=False)
def get_station_observations(station_id, tz_name, target_date):
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





def previous_day_forecast_summary(city, tz_name, target_date):
    """
    Summarize stored NWS predictions for the previous day's final high.

    Preferred source:
      forecasts issued on the calendar day before the previous observed day
      (the original historical comparison requested for this chart).

    Fallback:
      if that exact source-day slice is unavailable, use all stored snapshots
      issued before the end of the previous observed day that predicted that
      previous day's temperatures. This prevents the historical prediction
      reference from disappearing simply because the collector missed one
      calendar day's snapshots.

    Returns average/min/max/latest and snapshot count.
    """
    rows, err = get_snapshot_rows(city, None)
    snap, norm_err = normalize_snapshot_rows(rows)
    if err or norm_err or snap.empty:
        return None

    tz = ZoneInfo(tz_name)
    previous_day = target_date - timedelta(days=1)
    preferred_source_day = target_date - timedelta(days=2)

    work = snap.copy()
    work["valid_local"] = work["valid_at"].dt.tz_convert(tz)
    work["snapshot_local"] = work["snapshot_at"].dt.tz_convert(tz)
    work["valid_date"] = work["valid_local"].dt.date
    work["snapshot_date"] = work["snapshot_local"].dt.date

    # All snapshots that actually predicted hours belonging to the previous day.
    target_work = work[work["valid_date"] == previous_day].copy()
    if target_work.empty:
        return None

    preferred = target_work[
        target_work["snapshot_date"] == preferred_source_day
    ].copy()

    if not preferred.empty:
        chosen_work = preferred
        source_label = "day-before forecasts"
    else:
        # Fallback to any forecast snapshots captured no later than the end of
        # the previous day. This preserves useful historical information.
        previous_day_end = pd.Timestamp(
            datetime.combine(
                previous_day,
                datetime.max.time(),
                tzinfo=tz,
            )
        )
        fallback = target_work[
            target_work["snapshot_local"] <= previous_day_end
        ].copy()

        if fallback.empty:
            return None

        # Keep a practical recent window so ancient/stale snapshots do not
        # dominate the average if the database contains very early forecasts.
        latest_snapshot = fallback["snapshot_local"].max()
        window_start = latest_snapshot - pd.Timedelta(hours=36)
        chosen_work = fallback[
            fallback["snapshot_local"] >= window_start
        ].copy()
        source_label = "available prior forecasts"

    projected = (
        chosen_work.groupby("snapshot_key", as_index=False)
        .agg(
            projected_high_f=("temp_f", "max"),
            snapshot_at=("snapshot_at", "min"),
        )
        .dropna(subset=["projected_high_f"])
        .sort_values("snapshot_at")
    )
    if projected.empty:
        return None

    values = projected["projected_high_f"].astype(float)
    return {
        "average_f": float(values.mean()),
        "low_f": float(values.min()),
        "high_f": float(values.max()),
        "latest_f": float(projected.iloc[-1]["projected_high_f"]),
        "n_snapshots": int(len(values)),
        "source_label": source_label,
    }



def observed_daily_high_details(station_id, tz_name, target_date):
    """Return the observed daily high and first local timestamp tied for that high."""
    obs = get_station_observations(station_id, tz_name, target_date)
    if obs is None or obs.empty:
        return None, None
    tz = ZoneInfo(tz_name)
    work = obs.copy()
    work["time_local"] = pd.to_datetime(work["time"], utc=True, errors="coerce").dt.tz_convert(tz)
    work["temp_f"] = pd.to_numeric(work["temp_f"], errors="coerce")
    work = work.dropna(subset=["time_local", "temp_f"]).sort_values("time_local")
    if work.empty:
        return None, None
    high = float(work["temp_f"].max())
    tied = work[work["temp_f"] >= high - 1e-9]
    peak_time = tied.iloc[0]["time_local"] if not tied.empty else None
    return high, peak_time


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

    anchors = [v for v in [nws_raw, nws_center, observed, implied] if v is not None] + gfs_values
    if not anchors:
        return None

    spread = max(4.0, 4.0 * (nws_sigma or 1.5))
    x_min = math.floor(min(anchors) - spread)
    x_max = math.ceil(max(anchors) + spread)
    if x_max - x_min < 10:
        mid = (x_min + x_max) / 2
        x_min, x_max = math.floor(mid - 5), math.ceil(mid + 5)

    step = 0.10
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
                        tickCount=min(22, max(10, (x_max - x_min) * 2 + 1)),
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

    # Historical reference: forecasts made the day before for the previous day's high.
    prev_pred_avg = clean_number(row.get("previous_day_prediction_avg_f"))
    prev_pred_low = clean_number(row.get("previous_day_prediction_low_f"))
    prev_pred_high = clean_number(row.get("previous_day_prediction_high_f"))
    prev_pred_latest = clean_number(row.get("previous_day_prediction_latest_f"))
    prev_pred_n = int(row.get("previous_day_prediction_n") or 0)

    if prev_pred_avg is not None:
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
                    opacity=0.10,
                    stroke=history_color,
                    strokeWidth=1.5,
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

        # Always show the average of those prior-day forecasts as a vertical line.
        history_avg = pd.DataFrame([{
            "temperature": prev_pred_avg,
            "label": f"Previous-day forecast average · {prev_pred_n} snapshots",
        }])
        layers.append(
            alt.Chart(history_avg)
            .mark_rule(
                color=history_color,
                strokeWidth=2.5,
                strokeDash=[2, 3],
            )
            .encode(
                x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                tooltip=[
                    alt.Tooltip("label:N", title="Historical forecast"),
                    alt.Tooltip("temperature:Q", title="Average prediction", format=".1f"),
                ],
            )
        )

    # Most recent NWS predicted high for the previous day. Keep this as a
    # simple dotted vertical reference so yesterday's forecast is immediately
    # comparable with yesterday's observed high and today's distribution.
    if prev_pred_latest is not None:
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
        observed_label = "Observed so far"
        if observed_detail:
            observed_label += f" · {observed_detail}"
        marker_specs.append((observed_label, observed, observed_detail))
    if implied is not None:
        marker_specs.append(("Kalshi implied temp", implied, ""))
    if previous_day_high is not None:
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
    if previous_3day_avg is not None:
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
            if label.startswith("Observed so far"):
                return "#FF2D8D"
            if label.startswith("Kalshi implied temp"):
                return "#FFD400"
            if label.startswith("Previous day high"):
                return "#00E676"
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
                    opacity=0.16,
                    stroke=top_color,
                    strokeWidth=2,
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
        obs_label = "Observed so far"
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
            f"Kalshi implied temp · {float(row.get('kalshi_implied_temp_f')):.1f}°F",
            "#FFD400",
            "dash",
        ))

    prev_pred_avg = row.get("previous_day_prediction_avg_f")
    prev_pred_low = row.get("previous_day_prediction_low_f")
    prev_pred_high = row.get("previous_day_prediction_high_f")
    prev_pred_latest = row.get("previous_day_prediction_latest_f")
    prev_pred_n = int(row.get("previous_day_prediction_n") or 0)

    if present(prev_pred_latest):
        items.append((
            f"Previous day NWS predicted high · {float(prev_pred_latest):.1f}°F",
            "#63D8FF",
            "dot",
        ))

    if present(prev_pred_avg):
        source_label = row.get("previous_day_prediction_source") or "prior forecasts"
        if present(prev_pred_low) and present(prev_pred_high) and abs(float(prev_pred_high) - float(prev_pred_low)) >= 0.05:
            items.append((
                f"Previous-day forecast range · {float(prev_pred_low):.1f}–{float(prev_pred_high):.1f}°F · "
                f"avg {float(prev_pred_avg):.1f}°F · {prev_pred_n} snapshots",
                "#36C2FF",
                "history_band",
            ))
        else:
            items.append((
                f"Previous-day prediction · {float(prev_pred_avg):.1f}°F · {prev_pred_n} snapshot"
                + ("s" if prev_pred_n != 1 else ""),
                "#36C2FF",
                "dash",
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
        items.append((prev_label, "#00E676", "dash"))

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
    """Observed vs stitched NWS forecast history with readable numeric values."""
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
        ) + timedelta(hours=18)
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
    all_times = list(forecast["time"])
    if not observations.empty:
        all_times += list(observations["time"])
    x_min = min(all_times) if all_times else now_local - pd.Timedelta(hours=12)
    x_max = max(all_times) if all_times else now_local + pd.Timedelta(hours=12)

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
            tickCount=14,
            grid=True,
            gridOpacity=0.45,
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
            gridOpacity=0.48,
            tickSize=7,
            format=".0f",
        ),
    )

    layers = []

    if not observations.empty:
        layers.append(
            alt.Chart(observations)
            .mark_line(
                point=alt.OverlayMarkDef(filled=True, size=38),
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
                    alt.Tooltip(
                        "time:T",
                        title="Observed time",
                        format="%b %-d, %-I:%M %p",
                    ),
                    alt.Tooltip(
                        "temp_f:Q",
                        title="Observed",
                        format=".1f",
                    ),
                ],
            )
        )

    layers.append(
        alt.Chart(forecast)
        .mark_line(
            point=alt.OverlayMarkDef(filled=True, size=38),
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
                alt.Tooltip(
                    "time:T",
                    title="Forecast valid time",
                    format="%b %-d, %-I:%M %p",
                ),
                alt.Tooltip(
                    "temp_f:Q",
                    title="NWS prediction",
                    format=".1f",
                ),
                alt.Tooltip(
                    "snapshot_at:T",
                    title="Stored from snapshot",
                    format="%b %-d, %-I:%M %p",
                ),
            ],
        )
    )

    # Label only the newest point in each series, keeping the plot clean.
    label_rows = []
    if not observations.empty:
        p = observations.sort_values("time").iloc[-1]
        label_rows.append({
            "time": p["time"],
            "temp_f": float(p["temp_f"]),
            "Series": "Observed",
            "label": f"{float(p['temp_f']):.1f}°",
        })
    if not forecast.empty:
        p = forecast.sort_values("time").iloc[-1]
        label_rows.append({
            "time": p["time"],
            "temp_f": float(p["temp_f"]),
            "Series": "NWS prediction",
            "label": f"{float(p['temp_f']):.1f}°",
        })

    if label_rows:
        label_df = pd.DataFrame(label_rows)
        layers.append(
            alt.Chart(label_df)
            .mark_text(
                align="left",
                baseline="middle",
                dx=7,
                fontSize=11,
                fontWeight="bold",
            )
            .encode(
                x=alt.X("time:T"),
                y=alt.Y("temp_f:Q"),
                text="label:N",
                color=alt.Color("Series:N", scale=color_scale, legend=None),
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
        .resolve_scale(color="shared")
        .properties(
            height=340,
            background="#11121B",
            title=alt.TitleParams(
                text="Observed vs NWS prediction",
                subtitle=(
                    "Solid = observed · dashed = stored NWS forecast · "
                    "cyan = current time"
                ),
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
            symbolSize=170,
        )
        .configure_title(
            fontSize=19,
            subtitleFontSize=13,
            color="#FAF7FF",
            subtitleColor="#D4CEDD",
        )
        .configure_view(strokeWidth=0)
    )
    return chart, latest_key


def max_projection_history_chart(snapshot_df, tz_name):
    """One point per collector run: that run's predicted maximum for the selected day."""
    if snapshot_df.empty:
        return None, pd.DataFrame()

    tz = ZoneInfo(tz_name)
    history = (
        snapshot_df.groupby("snapshot_key", as_index=False)["temp_f"]
        .max()
        .rename(columns={"snapshot_key": "snapshot_time", "temp_f": "predicted_high_f"})
        .sort_values("snapshot_time")
    )
    history["snapshot_time"] = history["snapshot_time"].dt.tz_convert(tz)
    if history.empty:
        return None, history

    chart = (
        alt.Chart(history)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=58), strokeWidth=3, color="#B79CFF")
        .encode(
            x=alt.X(
                "snapshot_time:T",
                title="Forecast snapshot time",
                axis=alt.Axis(format="%b %-d, %-I:%M %p", labelAngle=-30, tickCount=8, grid=True, gridOpacity=0.30),
            ),
            y=alt.Y(
                "predicted_high_f:Q",
                title="Predicted daily high (°F)",
                scale=alt.Scale(zero=False, padding=12),
            ),
            tooltip=[
                alt.Tooltip("snapshot_time:T", title="Snapshot", format="%b %-d, %-I:%M %p"),
                alt.Tooltip("predicted_high_f:Q", title="Predicted high", format=".1f"),
            ],
        )
        .properties(
            height=260,
            title=alt.TitleParams(
                text="How the predicted high has changed",
                subtitle="Each point is the maximum temperature projected by one stored forecast snapshot",
                anchor="start",
            ),
        )
        .configure_axis(
            labelFontSize=14,
            titleFontSize=15,
            labelColor="#DED9EA",
            titleColor="#F5F1FA",
            gridColor="#5D5870",
            gridOpacity=0.20,
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
    history_chart, history = max_projection_history_chart(snapshot_df, cfg["tz"])

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




st.set_page_config(page_title="WeatherEdge", page_icon="🌦️", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --ink:#FFFFFF;
  --body:#F4F1F8;
  --muted:#D9D3E2;
  --muted-strong:#E7E2ED;
  --violet:#D0C3FF;
  --pink:#FFB7DA;
  --cyan:#9DF3F8;
  --panel:#171722;
  --panel-2:#1D1C2A;
  --border:rgba(224,218,235,.30);
}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
.stApp {
  background:
    radial-gradient(circle at 12% 5%, rgba(188,168,255,.16), transparent 31rem),
    radial-gradient(circle at 90% 18%, rgba(140,234,242,.10), transparent 29rem),
    linear-gradient(160deg, #09090F 0%, #12111B 48%, #0A1016 100%);
  color:var(--ink);
  font-size:17px;
  line-height:1.6;
}
.block-container {max-width:900px; padding-top:1.25rem; padding-bottom:3rem;}

/* Global typography and contrast */
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], .stCaption, p, li, span {
  color:var(--body);
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  font-size:1.03rem;
  line-height:1.62;
}
h1,h2,h3,h4,h5,h6 {color:#FFFFFF!important; font-weight:780!important;}
h1 {font-size:3rem!important; letter-spacing:-.035em!important; line-height:1.05!important;}
h2 {font-size:2rem!important; letter-spacing:-.02em!important;}
h3 {font-size:1.45rem!important;}
a {color:#B7F6FA!important;}
small, .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color:#DDD7E6!important;
  font-size:.96rem!important;
  line-height:1.5!important;
}

/* Sidebar: deliberately higher contrast than the main canvas */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg, #171723 0%, #11111A 100%)!important;
  border-right:1px solid rgba(255,255,255,.13)!important;
}
[data-testid="stSidebar"] > div {background:transparent!important;}
[data-testid="stSidebar"] * {color:#F8F6FB!important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color:#FFFFFF!important;
  font-size:1rem!important;
  font-weight:720!important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {
  color:#E1DCE8!important;
}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.18)!important;}

/* Inputs and controls */
label, [data-testid="stWidgetLabel"] p {
  color:#F8F6FB!important;
  font-size:1rem!important;
  font-weight:700!important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background:#20202C!important;
  color:#FFFFFF!important;
  border-color:rgba(255,255,255,.28)!important;
}
[data-baseweb="select"] span, [data-baseweb="select"] input {color:#FFFFFF!important;}
[data-baseweb="popover"] {color:#FFFFFF!important;}
[data-baseweb="menu"] {background:#20202C!important;}
[data-baseweb="menu"] li {color:#FFFFFF!important;}
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBar"] {color:#FFFFFF!important;}
[data-testid="stSlider"] div[role="slider"] {background:#A9F1F5!important;}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {background-color:#535164!important;}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {background-color:#9DF3F8!important;}
[data-testid="stCheckbox"] label p,
[data-testid="stToggle"] label p {color:#FFFFFF!important;}

/* Radio pills */
[data-testid="stRadio"] > div {gap:.45rem;}
[data-testid="stRadio"] label {
  background:#242332;
  border:1px solid rgba(255,255,255,.24);
  border-radius:999px;
  padding:.42rem .76rem;
  transition:.15s ease;
  color:#FFFFFF!important;
}
[data-testid="stRadio"] label p {font-size:.99rem!important; color:#FFFFFF!important; font-weight:680!important;}
[data-testid="stRadio"] label:hover {
  border-color:rgba(157,243,248,.82);
  background:#2B2A3A;
}

/* Buttons and links */
.stButton button, [data-testid="stLinkButton"] a {
  font-size:1rem!important;
  font-weight:760!important;
  color:#FFFFFF!important;
  background:#2A2939!important;
  border:1px solid rgba(255,255,255,.30)!important;
}
.stButton button:hover, [data-testid="stLinkButton"] a:hover {
  border-color:#9DF3F8!important;
  background:#343247!important;
}

/* Metrics */
[data-testid="stMetric"] {
  background:linear-gradient(145deg, #1B1A26, #171621);
  border:1px solid rgba(255,255,255,.20);
  padding:15px;
  border-radius:16px;
  box-shadow:0 10px 34px rgba(0,0,0,.20);
}
[data-testid="stMetricLabel"] p {
  color:#E4DEE9!important;
  font-size:.96rem!important;
  font-weight:680!important;
}
[data-testid="stMetricValue"] {
  color:#FFFFFF!important;
  font-size:1.7rem!important;
  font-weight:800!important;
}
[data-testid="stMetricDelta"] {color:#DFF9E8!important;}

/* Alerts, expanders, tables */
[data-testid="stAlert"] {
  background:#1D1C28!important;
  border:1px solid rgba(255,255,255,.18)!important;
}
[data-testid="stAlert"] p {
  font-size:1rem!important;
  line-height:1.55!important;
  color:#F7F4FA!important;
}
[data-testid="stExpander"] {
  background:#171721!important;
  border:1px solid rgba(255,255,255,.18)!important;
  border-radius:14px!important;
}
[data-testid="stExpander"] summary {
  background:#171721!important;
  color:#FFFFFF!important;
  border-radius:13px!important;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details[open] > summary,
[data-testid="stExpander"] details[open] > summary:hover {
  background:#20202C!important;
  color:#FFFFFF!important;
}
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] details[open] > summary * {
  color:#FFFFFF!important;
  fill:#FFFFFF!important;
}
[data-testid="stExpander"] summary p {color:#FFFFFF!important; font-weight:720!important;}
[data-testid="stDataFrame"] {border:1px solid rgba(255,255,255,.18)!important; border-radius:12px!important; overflow:hidden;}
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) {gap:.7rem;}

/* Custom cards */
.small-note {font-size:.98rem; color:#E0DAE7; line-height:1.55;}
.card-title {font-size:1.65rem; font-weight:800; margin-bottom:.15rem; letter-spacing:-.02em; color:#FFFFFF;}
.card-sub {font-size:1.05rem; color:#E4DEE9; margin-bottom:.8rem; line-height:1.55;}
.section-kicker {font-size:.79rem; letter-spacing:.14em; color:#D4C8FF; font-weight:850; margin:1.25rem 0 .55rem;}
.bet-shell {padding:.25rem 0 .5rem;}
.bet-callout {
  margin:.35rem 0 .75rem;
  padding:1rem 1.05rem;
  border-radius:18px;
  background:linear-gradient(135deg, #242239, #1B2B30);
  border:1px solid rgba(208,195,255,.42);
  box-shadow:0 12px 34px rgba(0,0,0,.22);
}
.bet-callout-label {font-size:.77rem; letter-spacing:.13em; color:#D9CFFF; font-weight:850; margin-bottom:.28rem;}
.bet-callout-main {font-size:1.3rem; line-height:1.38; color:#FFFFFF; font-weight:800;}
.bet-callout-sub {font-size:.98rem; color:#E8E3ED; margin-top:.3rem;}

.quality-card {
  margin:.55rem 0 .85rem;
  padding:1rem 1.05rem;
  border-radius:18px;
  background:linear-gradient(135deg, #173238, #2B2441);
  border:1px solid rgba(157,243,248,.48);
  box-shadow:0 12px 34px rgba(0,0,0,.22);
}
.quality-top {display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;}
.quality-label {font-size:.75rem; letter-spacing:.13em; color:#B5FAFD; font-weight:900;}
.quality-value {font-size:2.3rem; line-height:1; color:#FFFFFF; font-weight:900;}
.quality-grade {font-size:1rem; color:#E0D6FF; font-weight:850; text-align:right;}
.quality-sub {font-size:.92rem; color:#EBE7F0; margin-top:.46rem; line-height:1.5;}

.signal-strip {display:grid; grid-template-columns:1fr 1fr; gap:.65rem; margin:.7rem 0 .85rem;}
.signal-tile {
  padding:.82rem .92rem;
  border-radius:16px;
  background:#1B1A26;
  border:1px solid rgba(255,255,255,.20);
  min-width:0;
}
.signal-label {font-size:.71rem; letter-spacing:.105em; text-transform:uppercase; color:#D7CDFF; font-weight:850; margin-bottom:.24rem;}
.signal-value {font-size:1.3rem; line-height:1.12; color:#FFFFFF; font-weight:820;}
.signal-sub {font-size:.82rem; line-height:1.4; color:#DDD7E6; margin-top:.24rem;}

code {color:#C8F7DA!important; background:#151A17!important; font-size:.92em!important;}
hr {border-color:rgba(255,255,255,.16)!important;}

@media (max-width:520px) {
  .stApp {font-size:16px;}
  h1 {font-size:2.45rem!important;}
  .signal-strip {gap:.5rem;}
  .signal-tile {padding:.72rem .74rem;}
  .signal-value {font-size:1.18rem;}
  .signal-label {font-size:.65rem;}
  .quality-value {font-size:2rem;}
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ WeatherEdge")
st.caption("Best Bets, watched bets tracking, and a city-by-city weather market explorer.")

st.divider()

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
if "rounding_buffer_setting" not in st.session_state:
    st.session_state.rounding_buffer_setting = 0.5

with st.sidebar:
    st.markdown("### Top Bet Settings")
    st.caption(
        "Adjust the safety cutoffs and how many ranked Top Bets WeatherEdge shows."
    )

    with st.expander("Adjust Top Bet filters", expanded=False):
        top_n = st.slider(
            "Number of Top Bets",
            min_value=3,
            max_value=8,
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

        rounding_buffer_f = st.slider(
            "Rounding safety buffer (°F)",
            min_value=0.25,
            max_value=1.00,
            value=float(st.session_state.rounding_buffer_setting),
            step=0.05,
            key="rounding_buffer_setting",
            help=(
                "Exclude Top Bets when the observed daily high is this close to a "
                "whole-degree contract strike/threshold. Default 0.50°F protects "
                "against nearest-degree rounding ambiguity."
            ),
        )

    st.caption(
        "Live market scan is cached for 5 minutes. Changing these settings reranks "
        "the cached markets immediately and does not trigger a new live scan. "
        "Rounding safety excludes both YES and NO for a market when its observed "
        "high is too close to a whole-degree strike."
    )

min_nws_chance = min_nws_pct / 100
min_gap = min_gap_pct / 100
rounding_buffer_f = float(st.session_state.get("rounding_buffer_setting", 0.5))

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
        help="Force a fresh Kalshi + current weather scan. Otherwise the app reuses the latest scan for up to 5 minutes.",
    ):
        clear_live_data_caches()
        st.rerun()

with st.spinner("Refreshing live markets in parallel…"):
    all_rows, errors, refreshed_at = scan_live_market_universe()

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
df = df[df["conservative_edge"].notna()].copy()


rounding_safe_mask = (
    df["rounding_risk_distance_f"].isna()
    | (df["rounding_risk_distance_f"] > rounding_buffer_f)
)

qualified_all = df[
    (df["nws_support"] == True)
    & (df["conservative_prob"] >= min_nws_chance)
    & (df["conservative_edge"] >= min_gap)
    & rounding_safe_mask
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
    rounding_distance = row.get("rounding_risk_distance_f")
    if (
        rounding_distance is not None
        and not pd.isna(rounding_distance)
        and float(rounding_distance) <= rounding_buffer_f
    ):
        reasons.append(
            f"rounding risk: observed high is only {float(rounding_distance):.2f}°F "
            f"from a contract strike"
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

    rounding_distance = r.get("rounding_risk_distance_f")
    if (
        rounding_distance is not None
        and not pd.isna(rounding_distance)
        and float(rounding_distance) <= rounding_buffer_f
    ):
        st.warning(
            f"Excluded from Top Bets for rounding safety: the observed daily high "
            f"is only {float(rounding_distance):.2f}°F from a whole-degree contract "
            f"strike. Current safety buffer: ±{rounding_buffer_f:.2f}°F."
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
        st.metric("Observed so far", "—" if obs is None or pd.isna(obs) else f"{obs:.0f}°F")
        obs_time = r.get("observed_high_time_local")
        if obs is not None and not pd.isna(obs) and obs_time is not None and not pd.isna(obs_time):
            try:
                st.caption(f"Observed at {pd.Timestamp(obs_time).strftime('%-I:%M %p')} local time")
            except Exception:
                pass
        implied = r.get("kalshi_implied_temp_f")
        st.metric("Kalshi implied temp (approx.)", "—" if implied is None or pd.isna(implied) else f"{implied:.1f}°F")
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
    mismatch_sub = "NWS minus Kalshi implied temp"
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
            label = (
                f"#{rank}  {r['city']} · {r['market_subtitle']} · {r['side']}  "
                f"|  Quality {r['bet_quality_score']:.0f}/100  "
                f"|  NWS {r['conservative_prob']*100:.0f}%  "
                f"|  Edge {r['conservative_edge']*100:+.1f} pp"
            )
            st.button(
                label,
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

        representative = selected_row

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
        obs_label = "Observed so far"
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
            "Kalshi implied temp",
            "—" if implied is None or pd.isna(implied) else f"{implied:.1f}°F",
        )

        gap_now = representative.get("trajectory_current_gap_f")
        if gap_now is not None and not pd.isna(gap_now):
            adj = float(representative.get("trajectory_adjustment_f") or 0.0)
            obs_floor = representative.get("observed_high_f")
            raw_center = (
                raw_nws_high
                + float(representative.get("nws_bias_f") or 0.0)
                + adj
            )
            floor_bound = (
                obs_floor is not None
                and not pd.isna(obs_floor)
                and raw_center < float(obs_floor)
            )
            if abs(adj) >= 0.5 or floor_bound:
                if floor_bound:
                    st.caption(
                        f"Live trajectory adjustment: observations are currently "
                        f"{float(gap_now):+.1f}°F versus the stored NWS trajectory. "
                        f"The raw adjusted center would be {raw_center:.1f}°F, but the "
                        f"observed-high floor keeps the final-high center at "
                        f"{adjusted_center:.1f}°F."
                    )
                else:
                    st.caption(
                        f"Live trajectory adjustment: observations are currently "
                        f"{float(gap_now):+.1f}°F versus the stored NWS trajectory; "
                        f"the final-high distribution center is adjusted {adj:+.1f}°F."
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

**NWS-based chance** converts the latest NWS high into a probability distribution using historical NWS-versus-observed errors. Calibration is keyed to how far the forecast is from the city’s expected hottest point of the day. On same-day markets, the distribution is now also conditioned on the live observed-vs-NWS trajectory: persistent misses matter increasingly as the expected peak approaches, and a near-peak shortfall can pull the center lower and tighten remaining-upside uncertainty. GFS is not used in this probability.

**Observed high so far** is a hard floor on same-day markets. WeatherEdge assigns zero final-high probability below the exact highest temperature already observed, then renormalizes all remaining probability above that floor.

**Kalshi implied temp (approx.)** is reconstructed from the live prices of the event's temperature brackets. It is useful for spotting NWS-vs-market temperature disagreement, but it may differ slightly from the forecast number displayed in Kalshi's app.

**Weather Edge** is NWS-based contract probability minus the live Kalshi ask for the displayed side. A positive value favors that side; for example, +20 pp toward YES means the NWS-based chance is 20 percentage points above the YES ask. GFS is not used in Weather Edge.

**Time to settlement** matters because the NWS uncertainty used by WeatherEdge gets tighter as the outcome gets closer. The opportunity ranking prioritizes contract edge, then larger NWS-vs-Kalshi temperature mismatches, with an extra boost as settlement gets close.

The GFS ensemble remains on the forecast-range chart only as optional context. It does **not** affect probabilities, qualification, gaps, or rankings.
"""
    )

st.caption(
    "Research tool only. Forecasts can be wrong, prices can move, and settlement rules matter."
)
