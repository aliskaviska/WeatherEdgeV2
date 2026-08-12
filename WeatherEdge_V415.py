
import math
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

@st.cache_data(ttl=30)
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

@st.cache_data(ttl=300)
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
    calibration=None, hours_to_peak=None
):
    """Convert the latest NWS point high into an empirically calibrated probability.

    The distribution is centered on the latest NWS daily-high forecast plus any
    historically measured forecast bias for forecasts made this far from the
    expected hottest point of the day. Its spread is learned from historical
    NWS-vs-observed errors in the same peak-relative lead bucket.

    ``hours_left`` is retained for backward compatibility, but probability
    calibration now prefers ``hours_to_peak``.
    """
    if nws_high is None or pd.isna(nws_high):
        return None

    calibration_lead = hours_to_peak if hours_to_peak is not None else hours_left
    mean = float(nws_high) + nws_bias_f(calibration_lead, calibration)
    sigma = nws_sigma_f(calibration_lead, calibration)

    floor = None
    if observed_high is not None and not pd.isna(observed_high):
        floor = float(observed_high)
        mean = max(mean, floor)

    def cdf(x):
        return normal_cdf(x, mean, sigma)

    floor_cdf = cdf(floor - 0.5) if floor is not None else 0.0
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
            lower = max(float(lo) - 0.5, floor - 0.5)
            upper = float(hi) + 0.5
            raw = max(0.0, cdf(upper) - cdf(lower))
        elif kind in ("below", "below_equal"):
            cutoff = (float(hi) - 0.5) if kind == "below" else (float(hi) + 0.5)
            raw = max(0.0, cdf(cutoff) - floor_cdf)
        else:
            raw = max(0.0, 1.0 - cdf(max(float(lo) - 0.5, floor - 0.5)))
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
    Learn NWS daily-high forecast error relative to the day's hottest point.

    For every completed historical day:
      1. Find the actual observed daily high and the time of the high.
      2. Compare each stored NWS daily-high projection with the final observed high.
      3. Bucket that error by how many hours the forecast snapshot was before the
         day's actual hottest point.

    The returned metadata also contains a typical local peak time and recent
    daily peak times. Live forecasts use that typical time, modestly adjusted
    toward the previous day's peak.
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

    recent_dates = sorted(work["target_date"].unique())[-45:]
    work = work[work["target_date"].isin(recent_dates)].copy()

    projected = (
        work.groupby(["snapshot_key", "target_date"], as_index=False)
        .agg(
            projected_high_f=("temp_f", "max"),
            snapshot_at=("snapshot_at", "min"),
        )
    )

    observed_stats = {}
    peak_minutes_by_date = {}
    for d in recent_dates:
        obs = get_station_observations(station_id, tz_name, d)
        if obs is None or obs.empty:
            continue

        x = obs.copy()
        x["time_local"] = pd.to_datetime(x["time"], utc=True, errors="coerce").dt.tz_convert(tz)
        x["temp_f"] = pd.to_numeric(x["temp_f"], errors="coerce")
        x = x.dropna(subset=["time_local", "temp_f"]).sort_values("time_local")
        if x.empty:
            continue

        high = float(x["temp_f"].max())
        # Temperature often plateaus around the high. Use the midpoint of all
        # readings within 0.15°F of the daily maximum as the hottest-time estimate.
        peak_rows = x[x["temp_f"] >= high - 0.15].copy()
        minutes = (
            peak_rows["time_local"].dt.hour * 60
            + peak_rows["time_local"].dt.minute
            + peak_rows["time_local"].dt.second / 60.0
        )
        peak_minutes = float(minutes.median())
        peak_dt = pd.Timestamp(datetime.combine(d, datetime.min.time(), tzinfo=tz)) + pd.Timedelta(minutes=peak_minutes)

        observed_stats[d] = {"high_f": high, "peak_dt": peak_dt}
        peak_minutes_by_date[d.isoformat()] = peak_minutes

    samples = []
    for _, row in projected.iterrows():
        d = row["target_date"]
        stat = observed_stats.get(d)
        if stat is None:
            continue

        snapshot_local = pd.Timestamp(row["snapshot_at"]).tz_convert(tz)
        hours_to_peak = (stat["peak_dt"] - snapshot_local).total_seconds() / 3600.0
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

    peak_values = list(peak_minutes_by_date.values())
    typical_peak = float(pd.Series(peak_values).median()) if peak_values else 15 * 60.0
    calibration["__meta__"] = {
        "typical_peak_minutes": typical_peak,
        "peak_minutes_by_date": peak_minutes_by_date,
        "n_peak_days": len(peak_values),
    }
    return calibration

def _clamp01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


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

        trajectory_score = trajectory_agreement_score(city, cfg, d, nws_high)
        peak_context = expected_peak_context(calibration, d, cfg["tz"])
        hours_to_peak = peak_context["hours_to_peak"]

        observed_high = None
        observed_high_time = None
        previous_day_high = None
        previous_day_high_time = None
        previous_3day_avg_high = None
        try:
            observed_high, observed_high_time = observed_daily_high_details(
                cfg.get("station_id"), cfg["tz"], d
            )
            previous_day_high, previous_day_high_time, previous_3day_avg_high = recent_observed_high_summary(
                cfg.get("station_id"), cfg["tz"], d
            )
        except Exception:
            observed_high = None
            observed_high_time = None
            previous_day_high = None
            previous_day_high_time = None
            previous_3day_avg_high = None

        hours_left = hours_to_deadline(m, cfg["tz"])
        p_yes = nws_yes_probability(
            nws_high, kind, lo, hi,
            hours_left=hours_left,
            observed_high=observed_high,
            calibration=calibration,
            hours_to_peak=hours_to_peak,
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
        nws_support_yes = point_forecast_supports_yes(nws_high, kind, lo, hi)

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
                "previous_day_high_f": previous_day_high,
                "previous_day_high_time_local": previous_day_high_time,
                "previous_3day_avg_high_f": previous_3day_avg_high,
                "observed_data_url": nws_climate_url(cfg),
                "nws_forecast": nrow.get("nws_detail"),
                "nws_forecast_url": nrow.get("nws_forecast_url"),
                "nws_sigma_f": nws_sigma_f(hours_to_peak, calibration),
                "nws_bias_f": nws_bias_f(hours_to_peak, calibration),
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
    nws_sigma = clean_number(row.get("nws_sigma_f"))
    observed = clean_number(row.get("observed_high_f"))
    implied = clean_number(row.get("kalshi_implied_temp_f"))
    gfs_values = [
        float(v) for v in (row.get("ensemble_daily_highs_f") or [])
        if v is not None and math.isfinite(float(v))
    ]

    if nws_raw is None and not gfs_values:
        return None

    # This mirrors the center actually used by nws_yes_probability.
    nws_center = None
    if nws_raw is not None:
        nws_center = nws_raw + nws_bias
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
            if observed is not None and x < observed - 0.5:
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

    # Highlight the recommended contract only when this is actually a Top Bet.
    bet_kind = row.get("condition_kind")
    bet_low = clean_number(row.get("condition_low_f"))
    bet_high = clean_number(row.get("condition_high_f"))
    bet_side = str(row.get("side") or "").upper()

    if show_bet_overlay and bet_side in ("YES", "NO"):
        band_lo = band_hi = None
        if bet_kind == "range" and bet_low is not None and bet_high is not None:
            band_lo, band_hi = bet_low - 0.5, bet_high + 0.5
        elif bet_kind == "above" and bet_low is not None:
            band_lo, band_hi = bet_low - 0.5, x_max
        elif bet_kind in ("below", "below_equal") and bet_high is not None:
            cutoff = bet_high - 0.5 if bet_kind == "below" else bet_high + 0.5
            band_lo, band_hi = x_min, cutoff

        if band_lo is not None and band_hi is not None:
            top_bet_height = max(
                0.45,
                float(dist_df["density"].max()) * 1.12 if not dist_df.empty else 0.45,
            )
            bet_df = pd.DataFrame([{
                "x1": band_lo,
                "x2": band_hi,
                "y1": 0.0,
                "y2": top_bet_height,
                "label": f"TOP BET · {bet_side}",
            }])
            layers.append(
                alt.Chart(bet_df).mark_rect(
                    color="#FF3B30",
                    opacity=0.12,
                    stroke="#FF3B30",
                    strokeWidth=2,
                ).encode(
                    x=alt.X("x1:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    x2="x2:Q",
                    y=alt.Y("y1:Q"),
                    y2="y2:Q",
                    tooltip=[alt.Tooltip("label:N", title="Top Bet")],
                )
            )

            if bet_kind == "range":
                bet_point_x = (band_lo + band_hi) / 2.0
            elif bet_kind == "above":
                bet_point_x = band_lo
            else:
                bet_point_x = band_hi

            top_bet_point = pd.DataFrame([{
                "temperature": bet_point_x,
                "density": top_bet_height * 0.96,
                "label": f"TOP BET · {bet_side}",
            }])
            layers.append(
                alt.Chart(top_bet_point)
                .mark_point(
                    filled=True,
                    shape="diamond",
                    size=260,
                    color="#FF3B30",
                    stroke="#FFFFFF",
                    strokeWidth=2,
                )
                .encode(
                    x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    y=alt.Y("density:Q"),
                    tooltip=[
                        alt.Tooltip("label:N", title="Top Bet"),
                        alt.Tooltip("temperature:Q", title="Contract reference", format=".1f"),
                    ],
                )
            )

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
                        tickCount=min(14, x_max - x_min + 1),
                        tickSize=7,
                        grid=True,
                        labelExpr="datum.value + '°'",
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

    marker_rows = []
    marker_specs = []

    if nws_raw is not None:
        marker_specs.append(("Raw NWS high", nws_raw, ""))
    if nws_center is not None:
        marker_specs.append(("Calibrated NWS center", nws_center, ""))
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
        palette = [
            "#FFFFFF",  # Raw NWS high
            "#9B7BFF",  # Calibrated NWS center
            "#FF2D8D",  # Observed so far
            "#FFD400",  # Kalshi implied temp
            "#00E676",  # Previous day high
            "#FF8A00",  # Average high, previous 3 days
        ][:len(marker_rows)]
        marker_scale = alt.Scale(
            domain=[r["Marker"] for r in marker_rows],
            range=palette,
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

        if observed is not None:
            observed_label_for_point = next(
                (m["Marker"] for m in marker_rows if m["Marker"].startswith("Observed so far")),
                "Observed so far",
            )
            observed_point = pd.DataFrame([{
                "temperature": observed,
                "density": max(
                    0.04,
                    float(dist_df["density"].max()) * 0.92 if not dist_df.empty else 0.10,
                ),
                "label": observed_label_for_point,
            }])
            layers.append(
                alt.Chart(observed_point)
                .mark_point(
                    filled=True,
                    size=210,
                    color="#FF2D8D",
                    stroke="#FFFFFF",
                    strokeWidth=2,
                )
                .encode(
                    x=alt.X("temperature:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    y=alt.Y("density:Q"),
                    tooltip=[
                        alt.Tooltip("label:N", title="Observed so far"),
                        alt.Tooltip("temperature:Q", title="Temperature", format=".1f"),
                    ],
                )
            )

    if not layers:
        return None

    subtitle = (
        "NWS = probability model used for NWS chance. GFS = reference-only ensemble distribution."
    )

    main_chart = (
        alt.layer(*layers)
        .resolve_scale(color="independent")
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
            gridOpacity=.25,
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
        items.append(("Raw NWS high", "#FFFFFF", "dash"))
        items.append(("Calibrated NWS center", "#9B7BFF", "dash"))

    if present(row.get("observed_high_f")):
        obs_label = "Observed so far"
        obs_time = row.get("observed_high_time_local")
        if present(obs_time):
            try:
                obs_label += f" · {pd.Timestamp(obs_time).strftime('%-I:%M %p')}"
            except Exception:
                pass
        items.append((obs_label, "#FF2D8D", "circle"))

    if present(row.get("kalshi_implied_temp_f")):
        items.append(("Kalshi implied temp", "#FFD400", "dash"))

    if present(row.get("previous_day_high_f")):
        prev_label = "Previous day high"
        prev_time = row.get("previous_day_high_time_local")
        if present(prev_time):
            try:
                prev_label += f" · {pd.Timestamp(prev_time).strftime('%-I:%M %p')}"
            except Exception:
                pass
        items.append((prev_label, "#00E676", "dash"))

    if present(row.get("previous_3day_avg_high_f")):
        items.append(("Average high, previous 3 days", "#FF8A00", "dash"))

    status_text, status_kind = contract_status(row)
    if status_kind == "best":
        items.append((status_text, "#FF3B30", "diamond"))

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
    """Observed vs stitched NWS forecast history, with an explicit current-time marker.

    For each forecast-valid hour we keep the newest stored NWS snapshot that still
    contained that hour. This is why the cron snapshots matter: once an hour passes,
    newer NWS pulls may stop returning it, but the last forecast we stored for that
    hour remains available for comparison with observations.
    """
    if snapshot_df.empty:
        return None, None

    tz = ZoneInfo(tz_name)
    latest_key = snapshot_df["snapshot_key"].max()
    work = snapshot_df.copy()
    work["time"] = work["valid_at"].dt.tz_convert(tz)

    # A forecast is only meaningful as a historical prediction if it was captured
    # at or before the hour it predicts. Keep future forecasts from each snapshot,
    # including forecasts saved on the previous calendar day.
    work = work[work["snapshot_at"] <= work["valid_at"]].copy()

    # Build ONE NWS prediction line across the SAME display window as observations.
    # This intentionally includes the previous evening. The cron snapshots preserve
    # forecasts for hours that later disappear from the live NWS feed.
    display_start = pd.Timestamp(datetime.combine(target_date - timedelta(days=1), datetime.min.time(), tzinfo=tz) + timedelta(hours=18))
    display_end = pd.Timestamp(datetime.combine(target_date + timedelta(days=1), datetime.min.time(), tzinfo=tz))
    window_rows = work[(work["time"] >= display_start) & (work["time"] < display_end)].copy()
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
        x["time"] = pd.to_datetime(x["time"], utc=True, errors="coerce").dt.tz_convert(tz)
        x = x.dropna(subset=["time", "temp_f"])
        obs_parts.append(x[["time", "temp_f"]])
    observations = (pd.concat(obs_parts, ignore_index=True).sort_values("time")
                    if obs_parts else pd.DataFrame(columns=["time", "temp_f"]))
    if not observations.empty:
        observations = observations[(observations["time"] >= display_start) & (observations["time"] < display_end)].copy()
        observations["Series"] = "Observed"

    now_local = pd.Timestamp.now(tz=tz_name)
    all_times = list(forecast["time"])
    if not observations.empty:
        all_times += list(observations["time"])
    x_min = min(all_times) if all_times else now_local - pd.Timedelta(hours=12)
    x_max = max(all_times) if all_times else now_local + pd.Timedelta(hours=12)

    domain = ["Observed", "NWS prediction"]
    rng = ["#FF8FCB", "#B79CFF"]
    layers = []
    common_x = alt.X(
        "time:T", title="Local time",
        axis=alt.Axis(format="%b %-d, %-I %p", labelAngle=-30, tickCount=8, grid=True, tickSize=7),
        scale=alt.Scale(domain=[x_min, x_max]),
    )
    common_y = alt.Y(
        "temp_f:Q", title="Temperature (°F)", scale=alt.Scale(zero=False),
        axis=alt.Axis(tickCount=7, grid=True, tickSize=7),
    )

    if not observations.empty:
        layers.append(alt.Chart(observations).mark_line(
            point=alt.OverlayMarkDef(filled=True, size=42), strokeWidth=3
        ).encode(
            x=common_x, y=common_y,
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("time:T", title="Observed time", format="%b %-d, %-I:%M %p"),
                     alt.Tooltip("temp_f:Q", title="Observed", format=".1f")],
        ))

    layers.append(alt.Chart(forecast).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=42), strokeWidth=3, strokeDash=[7, 4]
    ).encode(
        x=common_x, y=common_y,
        color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("time:T", title="Forecast valid time", format="%b %-d, %-I:%M %p"),
                 alt.Tooltip("temp_f:Q", title="NWS prediction", format=".1f"),
                 alt.Tooltip("snapshot_at:T", title="Stored from snapshot", format="%b %-d, %-I:%M %p")],
    ))

    # Clearly mark NOW when it falls inside the displayed time window.
    if x_min <= now_local <= x_max:
        now_df = pd.DataFrame({"time": [now_local]})
        layers.append(alt.Chart(now_df).mark_rule(color="#8CEAF2", strokeWidth=2, strokeDash=[3, 3]).encode(x="time:T"))
        layers.append(alt.Chart(now_df).mark_text(
            text="NOW", color="#8CEAF2", fontSize=13, fontWeight="bold", angle=0,
            align="left", baseline="top", dx=5, dy=5
        ).encode(x="time:T", y=alt.value(5)))

    chart = alt.layer(*layers).resolve_scale(color="shared").properties(
        height=330, background="#11121B",
        title=alt.TitleParams(
            text="Observed vs NWS prediction",
            subtitle="Solid = observed · dashed = last stored NWS forecast for each hour · cyan line = current time",
            anchor="start",
        ),
    ).configure_axis(
        labelFontSize=13, titleFontSize=15, labelColor="#F1EDF7", titleColor="#FAF7FF",
        gridColor="#777185", gridOpacity=.25, domainColor="#AAA3B5", tickColor="#AAA3B5",
    ).configure_legend(
        orient="top", direction="horizontal", title=None, labelFontSize=14,
        labelColor="#F1EDF7", symbolSize=170,
    ).configure_title(
        fontSize=19, subtitleFontSize=13, color="#FAF7FF", subtitleColor="#D4CEDD",
    ).configure_view(strokeWidth=0)
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
                axis=alt.Axis(format="%b %-d, %-I:%M %p", labelAngle=-30, tickCount=6, grid=False),
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

    m1, m2, m3 = st.columns(3)
    m1.metric("Latest projected high", "—" if pd.isna(latest_high) else f"{latest_high:.0f}°F")
    m2.metric("Observed high so far", "—" if observed_high is None or pd.isna(observed_high) else f"{observed_high:.0f}°F")
    m3.metric("Stored snapshots", f"{history.shape[0]:,}")




st.set_page_config(page_title="WeatherEdge", page_icon="🌦️", layout="centered")

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
[data-testid="stExpander"] summary,
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

with st.sidebar:
    # Keep controls out of the way unless the user is actively working in Best Bets.
    if st.session_state.get("main_view", "Best Bets") == "Best Bets":
        with st.expander("Bet Settings", expanded=False):
            top_n = st.slider(
                "Top candidates", 3, 8, int(st.session_state.top_n_setting), 1,
                key="top_n_setting",
            )
            min_nws_pct = st.slider(
                "Minimum NWS-based chance",
                min_value=55,
                max_value=95,
                value=int(st.session_state.min_nws_setting),
                step=1,
                help="Hard safety cutoff. Bets below this NWS-based win probability are excluded before ranking.",
                key="min_nws_setting",
            )
            min_gap_pct = st.slider(
                "Minimum Weather Edge", 0, 30, int(st.session_state.min_gap_setting), 1,
                key="min_gap_setting",
            )
    else:
        top_n = int(st.session_state.top_n_setting)
        min_nws_pct = int(st.session_state.min_nws_setting)
        min_gap_pct = int(st.session_state.min_gap_setting)

min_nws_chance = min_nws_pct / 100
min_gap = min_gap_pct / 100

# City Explorer needs the full market universe, so always scan every supported city.
cities = list(PRESETS.keys())
all_rows, errors = [], []

with st.spinner("Scanning live markets…"):
    for city in cities:
        try:
            all_rows.extend(build_city_rows(city, PRESETS[city]))
        except Exception as e:
            errors.append(f"{city}: {e}")

if not all_rows:
    st.warning("No matching open weather markets were found.")
    st.stop()

df = pd.DataFrame(all_rows)
df = df[df["conservative_edge"].notna()].copy()


qualified_all = df[
    (df["nws_support"] == True)
    & (df["conservative_prob"] >= min_nws_chance)
    & (df["conservative_edge"] >= min_gap)
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

    return (
        "NOT CURRENTLY A TOP BET"
        + (f" · {' · '.join(reasons)}" if reasons else ""),
        "other",
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
        "Browse every configured Kalshi temperature city whether or not it currently has a Best Bet."
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
        city_df = df[df["city"] == city].copy()

        # Use the nearest active contract date as the city-level weather context.
        city_df = city_df.sort_values(["date", "market_ticker", "side"])
        available_dates = list(city_df["date"].drop_duplicates())
        date_labels = [pd.Timestamp(d).strftime("%a %b %-d") for d in available_dates]
        selected_date_label = st.selectbox(
            "Market date",
            date_labels,
            index=0,
            key=f"explorer_date_{city}",
        )
        selected_date = available_dates[date_labels.index(selected_date_label)]
        date_df = city_df[city_df["date"] == selected_date].copy()

        representative = date_df.sort_values(
            ["bet_quality_score", "conservative_prob"], ascending=[False, False]
        ).iloc[0]

        st.markdown(f"<div class='card-title'>{city} weather overview</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Latest NWS high",
            "—" if pd.isna(representative["nws_high_f"]) else f"{representative['nws_high_f']:.0f}°F",
        )
        obs = representative.get("observed_high_f")
        m2.metric("Observed high so far", "—" if obs is None or pd.isna(obs) else f"{obs:.0f}°F")
        implied = representative.get("kalshi_implied_temp_f")
        m3.metric(
            "Kalshi implied temp",
            "—" if implied is None or pd.isna(implied) else f"{implied:.1f}°F",
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
                st.link_button("Observed temperatures ↗", observed_url, use_container_width=True)

        # All three city weather figures: forecast range + observed/NWS trajectory + snapshot history.
        range_chart = forecast_range_summary_chart(representative, show_bet_overlay=False)
        if range_chart is not None:
            st.markdown("<div class='section-kicker'>DAILY HIGH PROBABILITY</div>", unsafe_allow_html=True)
            st.altair_chart(range_chart, use_container_width=True)
            render_probability_chart_legend(representative)
        render_bet_forecast(city, selected_date)

        st.markdown("<div class='section-kicker'>AVAILABLE BETS</div>", unsafe_allow_html=True)
        st.caption(
            "Select any open side below to inspect its full WeatherEdge analysis. "
            "Top Bets are labeled automatically."
        )

        contract_rows = [r for _, r in date_df.sort_values(
            ["market_subtitle", "side", "ask"], ascending=[True, True, True]
        ).iterrows()]
        contract_labels = []
        contract_keys = []
        for row in contract_rows:
            status, kind = contract_status(row)
            badge = "★ " if kind == "best" else ""
            contract_labels.append(
                f"{badge}{row['market_subtitle']} · {row['side']} {row['ask']*100:.0f}¢ · "
                f"NWS {row['conservative_prob']*100:.0f}% · {status}"
            )
            contract_keys.append(f"{row['market_ticker']}|{row['side']}")

        desired = st.session_state.explorer_contract
        default_idx = contract_keys.index(desired) if desired in contract_keys else 0
        selected_contract_label = st.selectbox(
            "Open contract",
            contract_labels,
            index=default_idx,
            key=f"explorer_contract_select_{city}_{selected_date}",
        )
        contract_idx = contract_labels.index(selected_contract_label)
        selected_row = contract_rows[contract_idx]
        st.session_state.explorer_contract = contract_keys[contract_idx]

        st.divider()
        render_contract_detail(selected_row, show_weather=False)

with st.expander("How to read this"):
    st.markdown(
        """
**Latest NWS high** is the newest official point forecast for the settlement-station area.

**NWS-based chance** converts the latest NWS high into a probability distribution using historical NWS-versus-observed errors. Calibration is now keyed to how far the forecast is from the city’s expected hottest point of the day, not merely hours until formal settlement. The expected peak is anchored to historical peak times and modestly adjusted toward the previous day’s actual peak. GFS is not used in this probability.

**Observed high so far** is treated as a hard floor on same-day markets because the final daily high cannot finish below a temperature already observed.

**Kalshi implied temp (approx.)** is reconstructed from the live prices of the event's temperature brackets. It is useful for spotting NWS-vs-market temperature disagreement, but it may differ slightly from the forecast number displayed in Kalshi's app.

**Weather Edge** is NWS-based contract probability minus the live Kalshi ask for the displayed side. A positive value favors that side; for example, +20 pp toward YES means the NWS-based chance is 20 percentage points above the YES ask. GFS is not used in Weather Edge.

**Time to settlement** matters because the NWS uncertainty used by WeatherEdge gets tighter as the outcome gets closer. The opportunity ranking prioritizes contract edge, then larger NWS-vs-Kalshi temperature mismatches, with an extra boost as settlement gets close.

The GFS ensemble remains on the forecast-range chart only as optional context. It does **not** affect probabilities, qualification, gaps, or rankings.
"""
    )

st.caption(
    "Research tool only. Forecasts can be wrong, prices can move, and settlement rules matter."
)
