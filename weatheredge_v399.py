
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
    "New York": {
        "series": "KXHIGHNY",
        "lat": 40.7812, "lon": -73.9665,
        "tz": "America/New_York",
        "station": "Central Park / NYC settlement area",
        "station_id": "KNYC",
    },
    "Chicago": {
        "series": "KXHIGHCHI",
        "lat": 41.9742, "lon": -87.9073,
        "tz": "America/Chicago",
        "station": "Chicago O'Hare area",
        "station_id": "KORD",
    },
    "Miami": {
        "series": "KXHIGHMIA",
        "lat": 25.7959, "lon": -80.2870,
        "tz": "America/New_York",
        "station": "Miami International Airport area",
        "station_id": "KMIA",
    },
    "Los Angeles": {
        "series": "KXHIGHLAX",
        "lat": 33.9416, "lon": -118.4085,
        "tz": "America/Los_Angeles",
        "station": "Los Angeles International Airport area",
        "station_id": "KLAX",
    },
    "Denver": {
        "series": "KXHIGHDEN",
        "lat": 39.8561, "lon": -104.6737,
        "tz": "America/Denver",
        "station": "Denver International Airport area",
        "station_id": "KDEN",
    },
}


# Known Kalshi public-page slugs for the temperature series.
KALSHI_SERIES_SLUGS = {
    "KXHIGHNY": "highest-temperature-in-nyc",
    "KXHIGHCHI": "highest-temperature-in-chicago",
    "KXHIGHMIA": "highest-temperature-in-miami",
    "KXHIGHLAX": "highest-temperature-in-los-angeles",
    "KXHIGHDEN": "highest-temperature-in-denver",
}

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
    for key in ("occurrence_datetime", "expected_expiration_time", "expiration_time", "close_time"):
        raw = m.get(key)
        if raw:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return dt.astimezone(ZoneInfo(tz_name)).date()
            except Exception:
                pass
    blob = " ".join(str(m.get(k, "")) for k in ("title", "subtitle", "ticker", "event_ticker"))
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", blob)
    if match:
        return datetime.strptime(match.group(0), "%Y-%m-%d").date()
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


def _lead_bucket(hours_left):
    """Stable lead-time buckets used for empirical NWS error calibration."""
    if hours_left is None:
        return "unknown"
    if hours_left <= 3:
        return "0-3h"
    if hours_left <= 6:
        return "3-6h"
    if hours_left <= 12:
        return "6-12h"
    if hours_left <= 24:
        return "12-24h"
    if hours_left <= 48:
        return "24-48h"
    if hours_left <= 72:
        return "48-72h"
    return "72h+"


def heuristic_nws_sigma_f(hours_left):
    """Fallback uncertainty used until enough historical NWS errors exist."""
    if hours_left is None:
        return 2.8
    if hours_left <= 3:
        return 0.9
    if hours_left <= 6:
        return 1.1
    if hours_left <= 12:
        return 1.4
    if hours_left <= 24:
        return 1.8
    if hours_left <= 48:
        return 2.3
    if hours_left <= 72:
        return 2.8
    return 3.4


def nws_sigma_f(hours_left, calibration=None):
    """Use empirically measured NWS forecast error when enough samples exist."""
    bucket = _lead_bucket(hours_left)
    if calibration and bucket in calibration:
        row = calibration[bucket]
        if row.get("n", 0) >= 8 and row.get("sigma_f") is not None:
            # Keep a small floor so a short lucky streak cannot create fake certainty.
            return max(0.55, float(row["sigma_f"]))
    return heuristic_nws_sigma_f(hours_left)


def normal_cdf(x, mean, sigma):
    if sigma <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))


def nws_yes_probability(nws_high, kind, lo, hi, hours_left=None, observed_high=None, calibration=None):
    """Convert the latest NWS point high into an NWS-only contract probability.

    We model the final daily high as a normal distribution centered on the latest
    NWS high, with uncertainty that shrinks as the market approaches settlement.
    For same-day markets, the distribution is conditioned on the final high being
    at least the highest temperature already observed at the settlement station.
    Integer contracts use half-degree continuity boundaries.
    """
    if nws_high is None or pd.isna(nws_high):
        return None
    mean = float(nws_high)
    sigma = nws_sigma_f(hours_left, calibration)
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
        # Remove mass below an already-observed high and renormalize.
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
    Learn NWS daily-high error by lead time from cron snapshots.

    Each historical cron snapshot is reduced to its projected daily maximum for
    each local date. That projected high is compared with the final observed
    station high for the same date. The resulting forecast errors determine
    sigma for each lead-time bucket. GFS is never used here.
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

    # Only completed days can be calibrated against a final observed high.
    work = work[work["target_date"] < now_local.date()].copy()
    if work.empty:
        return {}

    # Keep recent history bounded so app startup remains reasonable.
    recent_dates = sorted(work["target_date"].unique())[-35:]
    work = work[work["target_date"].isin(recent_dates)].copy()

    projected = (
        work.groupby(["snapshot_key", "target_date"], as_index=False)
        .agg(projected_high_f=("temp_f", "max"),
             snapshot_at=("snapshot_at", "min"))
    )

    observed_by_date = {}
    for d in recent_dates:
        obs = get_station_observations(station_id, tz_name, d)
        if not obs.empty:
            observed_by_date[d] = float(obs["temp_f"].max())

    samples = []
    for _, row in projected.iterrows():
        d = row["target_date"]
        observed = observed_by_date.get(d)
        if observed is None:
            continue
        snapshot_local = pd.Timestamp(row["snapshot_at"]).tz_convert(tz_name)
        # Approximate the daily-temperature contract horizon by local end-of-day.
        deadline = pd.Timestamp(datetime.combine(d, datetime.max.time(), tzinfo=tz))
        hours_left = max(0.0, (deadline - snapshot_local).total_seconds() / 3600.0)
        error = observed - float(row["projected_high_f"])
        samples.append({
            "bucket": _lead_bucket(hours_left),
            "error_f": error,
        })

    if not samples:
        return {}

    sdf = pd.DataFrame(samples)
    calibration = {}
    for bucket, grp in sdf.groupby("bucket"):
        errors = grp["error_f"].dropna()
        n = int(len(errors))
        if n == 0:
            continue
        # RMSE measures actual forecast spread around zero and naturally includes bias.
        sigma = float((errors.pow(2).mean()) ** 0.5)
        calibration[bucket] = {
            "n": n,
            "sigma_f": sigma,
            "bias_f": float(errors.mean()),
            "mae_f": float(errors.abs().mean()),
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

        observed_high = None
        try:
            observed_high = get_observed_high_so_far(cfg.get("station_id"), cfg["tz"], d)
        except Exception:
            observed_high = None

        hours_left = hours_to_deadline(m, cfg["tz"])
        p_yes = nws_yes_probability(
            nws_high, kind, lo, hi,
            hours_left=hours_left,
            observed_high=observed_high,
            calibration=calibration,
        )
        if p_yes is None:
            continue

        # Optional GFS reference values for the chart only.
        ensemble_median = ensemble_low = ensemble_high = None
        if ens is not None and d in ens.index:
            daily_members = pd.Series(ens.loc[d].values).dropna().astype(float)
            if observed_high is not None and not daily_members.empty:
                daily_members = daily_members.clip(lower=float(observed_high))
            if not daily_members.empty:
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
            sigma_source = "historical" if calibration.get(_lead_bucket(hours_left), {}).get("n", 0) >= 8 else "fallback"
            sigma_samples = calibration.get(_lead_bucket(hours_left), {}).get("n", 0)
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
                "observed_data_url": nws_observed_data_url(cfg.get("station_id")) if observed_high is not None else None,
                "nws_forecast": nrow.get("nws_detail"),
                "nws_forecast_url": nrow.get("nws_forecast_url"),
                "nws_sigma_f": nws_sigma_f(hours_left, calibration),
                "nws_sigma_source": "historical" if calibration.get(_lead_bucket(hours_left), {}).get("n", 0) >= 8 else "fallback",
                "nws_sigma_samples": calibration.get(_lead_bucket(hours_left), {}).get("n", 0),
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


def _chart_base_properties(chart, height=300):
    return chart.properties(height=height).configure_view(strokeWidth=0)



def forecast_range_summary_chart(row):
    """Readable daily-high summary with an explicit legend and temperature ticks."""
    def clean_number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    low = clean_number(row.get("ensemble_low_f"))
    high = clean_number(row.get("ensemble_high_f"))
    median = clean_number(row.get("ensemble_median_f"))
    projected = clean_number(row.get("nws_high_f"))
    observed = clean_number(row.get("observed_high_f"))
    values = [v for v in (low, high, median, projected, observed) if v is not None]
    if not values:
        return None

    domain_min = math.floor(min(values) - 2)
    domain_max = math.ceil(max(values) + 2)
    if domain_max - domain_min < 8:
        center = (domain_min + domain_max) / 2
        domain_min, domain_max = math.floor(center - 4), math.ceil(center + 4)
    ticks = list(range(domain_min, domain_max + 1, 1))

    colors = {
        "GFS 80% range": "#73E5F2",
        "GFS median": "#F1EDF7",
        "NWS projected high": "#B79CFF",
        "Observed high": "#FF8FCB",
    }
    domain, rng = list(colors), list(colors.values())
    layers = []

    if low is not None and high is not None:
        d = pd.DataFrame([{"low": low, "high": high, "lane": "Daily high", "Series": "GFS 80% range"}])
        layers.append(alt.Chart(d).mark_rule(strokeWidth=15, opacity=.75, strokeCap="round").encode(
            x=alt.X("low:Q", scale=alt.Scale(domain=[domain_min, domain_max]),
                    axis=alt.Axis(title="Temperature (°F)", values=ticks, tickSize=8, grid=True,
                                  labelExpr="datum.value + '°'")),
            x2="high:Q", y=alt.Y("lane:N", axis=None),
            color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("low:Q", title="GFS low", format=".1f"), alt.Tooltip("high:Q", title="GFS high", format=".1f")]))

    pts=[]
    for label, value in (("GFS median", median), ("NWS projected high", projected), ("Observed high", observed)):
        if value is not None: pts.append({"temperature":value,"lane":"Daily high","Series":label})
    if pts:
        d=pd.DataFrame(pts)
        layers.append(alt.Chart(d).mark_point(filled=True,size=230,stroke="#11121B",strokeWidth=2).encode(
            x=alt.X("temperature:Q", scale=alt.Scale(domain=[domain_min,domain_max]),
                    axis=alt.Axis(title="Temperature (°F)", values=ticks, tickSize=8, grid=True,
                                  labelExpr="datum.value + '°'")),
            y=alt.Y("lane:N",axis=None), color=alt.Color("Series:N",scale=alt.Scale(domain=domain,range=rng),legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("Series:N",title="Measure"),alt.Tooltip("temperature:Q",title="Temperature",format=".1f")]))
        layers.append(alt.Chart(d).mark_text(dy=-22,fontSize=15,fontWeight="bold").encode(
            x=alt.X("temperature:Q",scale=alt.Scale(domain=[domain_min,domain_max])), y=alt.Y("lane:N",axis=None),
            text=alt.Text("temperature:Q",format=".0f"), color=alt.Color("Series:N",scale=alt.Scale(domain=domain,range=rng),legend=None)))

    return alt.layer(*layers).resolve_scale(color="shared").properties(
        height=190, background="#11121B",
        title=alt.TitleParams(text="Forecast range & daily highs", subtitle="Reference view: GFS range/median, NWS projected high, observed high", anchor="start")
    ).configure_axis(labelFontSize=14,titleFontSize=16,labelColor="#F1EDF7",titleColor="#FAF7FF",gridColor="#777185",gridOpacity=.28,
                     domainColor="#AAA3B5",tickColor="#AAA3B5",tickWidth=1.5).configure_legend(
        orient="top",direction="horizontal",columns=2,title=None,labelFontSize=13,labelColor="#F1EDF7",symbolSize=180,padding=8
    ).configure_title(fontSize=19,subtitleFontSize=13,color="#FAF7FF",subtitleColor="#D4CEDD").configure_view(strokeWidth=0)

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
st.caption("NWS-vs-market weather opportunities, with live weather context.")

st.divider()

with st.sidebar:
    scan_mode = st.radio("Scan", ["All preset cities", "One city"], index=0)
    selected_city = st.selectbox("City", list(PRESETS.keys())) if scan_mode == "One city" else None
    top_n = st.slider("Top candidates", 3, 8, 5, 1)
    min_nws_chance = st.slider(
        "Minimum NWS-based chance",
        min_value=55,
        max_value=95,
        value=70,
        step=1,
        help="Hard safety cutoff. Bets below this NWS-based win probability are excluded before ranking.",
    ) / 100
    min_gap = st.slider("Minimum Weather Edge", 0, 30, 5, 1) / 100

cities = [selected_city] if selected_city else list(PRESETS.keys())
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

qualified = df[
    (df["nws_support"] == True)
    & (df["conservative_prob"] >= min_nws_chance)
    & (df["conservative_edge"] >= min_gap)
].copy()

qualified = qualified.sort_values(
    ["bet_quality_score", "conservative_edge", "opportunity_score", "volume"],
    ascending=[False, False, False, False],
).head(top_n)

if qualified.empty:
    st.info(
        f"No candidates currently pass the safety filters: "
        f"NWS-based chance ≥ {min_nws_chance*100:.0f}% and Weather Edge ≥ {min_gap*100:.0f} pp."
    )
else:
    st.subheader("Best candidates")

if not qualified.empty:
    bet_rows = [r for _, r in qualified.iterrows()]
    bet_labels = [
        f"#{i} · {r['city']} · {r['side']} · {r['bet_quality_score']:.0f}/100"
        for i, r in enumerate(bet_rows, start=1)
    ]
    selected_label = st.radio(
        "Top bets",
        bet_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="top_bet_selector",
    )
    selected_idx = bet_labels.index(selected_label)
    r = bet_rows[selected_idx]
    rank = selected_idx + 1

    st.markdown("<div class='bet-shell'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card-title'>#{rank} · {r['city']} · {r['date_label']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='bet-callout'>"
        f"<div class='bet-callout-label'>BET TO PLACE</div>"
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

    st.link_button("Open this bet on Kalshi ↗", r["kalshi_event_url"], use_container_width=True)
    st.caption(f"Settlement location: **{r['station_hint']}**")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latest NWS high", "—" if pd.isna(r["nws_high_f"]) else f"{int(r['nws_high_f'])}°F")
        obs = r.get("observed_high_f")
        st.metric("Observed high so far", "—" if obs is None or pd.isna(obs) else f"{obs:.0f}°F")
        implied = r.get("kalshi_implied_temp_f")
        st.metric("Kalshi implied temp (approx.)", "—" if implied is None or pd.isna(implied) else f"{implied:.1f}°F")
    with c2:
        st.metric("Kalshi price", f"{r['ask']*100:.0f}¢")
        st.metric("NWS-based chance", fmt_pct(r["conservative_prob"]))
        st.metric("Weather Edge", f"{r['conservative_edge']*100:+.1f} pp", help=f"NWS-based chance minus the live Kalshi {r['side']} ask. Positive values favor {r['side']}.")
        st.caption(f"{r['conservative_edge']*100:+.1f} pp toward {r['side']}")

    mismatch = r.get("temperature_mismatch_f")
    hours_left = r.get("hours_to_settlement")
    mismatch_text = "—" if mismatch is None or pd.isna(mismatch) else f"{mismatch:+.1f}°F"
    time_text = "—" if hours_left is None or pd.isna(hours_left) else (f"{hours_left:.1f}h" if hours_left < 48 else f"{hours_left/24:.1f}d")

    # Compact signal row: present these like the other app categories, but keep the
    # two decision-critical values together so they can be read in one glance.
    mismatch_sub = "NWS minus Kalshi implied temp"
    if mismatch is not None and not pd.isna(mismatch):
        mismatch_sub = "NWS warmer than market" if mismatch > 0 else ("NWS cooler than market" if mismatch < 0 else "NWS and market aligned")
    time_sub = "Until market settlement"
    if hours_left is not None and not pd.isna(hours_left):
        if hours_left <= 6:
            time_sub = "Very close to settlement"
        elif hours_left <= 24:
            time_sub = "Settles within 24 hours"
        elif hours_left <= 48:
            time_sub = "Settles within 2 days"

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
    st.success("Latest NWS forecast supports this side")
    if r["nws_forecast"]:
        st.caption(f"NWS: {r['nws_forecast']}")
    observed_url = r.get("observed_data_url")
    if observed_url and isinstance(observed_url, str) and observed_url.startswith(("http://", "https://")):
        # Markdown link is more tolerant across Streamlit versions than st.link_button.
        st.markdown(f"[View official observed data ↗]({observed_url})")
    if r["suspicious"]:
        st.warning("Large gap. Verify the live contract before betting.")

    # Put the original forecast-range summary back beside the selected bet.
    range_chart = forecast_range_summary_chart(r)
    if range_chart is not None:
        st.markdown("<div class='section-kicker'>FORECAST RANGE</div>", unsafe_allow_html=True)
        st.altair_chart(range_chart, use_container_width=True)

    # Keep the snapshot-driven weather evidence beside the recommendation it supports.
    render_bet_forecast(r["city"], r["date"])

    if r.get("nws_forecast_url"):
        st.link_button("Open NWS forecast", r["nws_forecast_url"], use_container_width=True)
    st.caption(
        f"Kalshi: find **{r['market_subtitle']}** and choose **{r['side']}** · ticker `{r['market_ticker']}`"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("See rejected / conflicting contracts"):
    rejected = df[~df.index.isin(qualified.index)].copy()
    if rejected.empty:
        st.write("None.")
    else:
        rejected["Status"] = rejected["agreement"]
        rejected["Price"] = rejected["ask"].map(lambda x: f"{x*100:.0f}¢")
        rejected["NWS station forecast"] = rejected["nws_high_f"].map(lambda x: "—" if pd.isna(x) else f"{int(x)}°F")
        rejected["Observed high"] = rejected["observed_high_f"].map(lambda x: "—" if pd.isna(x) else f"{x:.0f}°F")
        rejected["Kalshi implied"] = rejected["kalshi_implied_temp_f"].map(lambda x: "—" if x is None or pd.isna(x) else f"{x:.1f}°F")
        rejected["Temp mismatch"] = rejected["temperature_mismatch_f"].map(lambda x: "—" if x is None or pd.isna(x) else f"{x:+.1f}°F")
        rejected["Weather Edge"] = rejected["conservative_edge"].map(lambda x: f"{x*100:+.1f} pp")
        rejected["Bet Quality"] = rejected["bet_quality_score"].map(lambda x: f"{x:.0f}/100")
        rejected["Safety cutoff"] = rejected["conservative_prob"].map(
            lambda x: "PASS" if x >= min_nws_chance else "FAIL"
        )
        st.dataframe(
            rejected[
                ["city", "date_label", "side", "market_subtitle", "Status", "NWS station forecast", "Observed high", "Kalshi implied", "Temp mismatch", "Price", "Weather Edge", "Bet Quality", "Safety cutoff"]
            ].rename(columns={
                "city": "City",
                "date_label": "Date",
                "side": "Side",
                "market_subtitle": "Outcome",
            }),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("How to read this"):
    st.markdown(
        """
**Latest NWS high** is the newest official point forecast for the settlement-station area.

**NWS-based chance** converts that single NWS high into a probability distribution. Its uncertainty is calibrated from stored cron snapshots versus final observed station highs for the same city and lead-time bucket once at least 8 historical samples exist. Until then, WeatherEdge uses the previous conservative fallback uncertainty. GFS is not used in this probability.

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
