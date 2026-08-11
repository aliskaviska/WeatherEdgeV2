
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


def nws_sigma_f(hours_left):
    """Heuristic NWS daily-high uncertainty. It narrows as settlement approaches."""
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


def normal_cdf(x, mean, sigma):
    if sigma <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (sigma * math.sqrt(2.0))))


def nws_yes_probability(nws_high, kind, lo, hi, hours_left=None, observed_high=None):
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
    sigma = nws_sigma_f(hours_left)
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

def build_city_rows(city, cfg):
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
                "nws_sigma_f": nws_sigma_f(hours_left),
                "hours_to_settlement": hours_left,
                "kalshi_implied_temp_f": implied_temp,
                "temperature_mismatch_f": temp_gap,
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
    Read every stored hourly projection for one city + contract date.

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
    params = {
        "select": "*",
        "city": f"eq.{city}",
        "contract_date": f"eq.{contract_date.isoformat()}",
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
    """Compact daily-high summary: GFS 80% range, median, NWS projected high, and observed high."""
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
        domain_min = math.floor(center - 4)
        domain_max = math.ceil(center + 4)

    color_domain = ["GFS 80% range", "GFS median", "Projected high", "Observed high"]
    color_range = ["#73E5F2", "#EAE6F2", "#BCA8FF", "#F4A9D2"]

    layers = []
    if low is not None and high is not None:
        range_df = pd.DataFrame([{
            "low": low, "high": high, "lane": "Daily high", "Series": "GFS 80% range"
        }])
        layers.append(
            alt.Chart(range_df)
            .mark_rule(strokeWidth=13, opacity=0.68, strokeCap="round")
            .encode(
                x=alt.X("low:Q", title="Temperature (°F)", scale=alt.Scale(domain=[domain_min, domain_max]),
                        axis=alt.Axis(values=list(range(domain_min, domain_max + 1)), tickSize=7, grid=True, labelExpr="datum.value + '°'")),
                x2="high:Q",
                y=alt.Y("lane:N", axis=None),
                color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=color_domain, range=color_range)),
                tooltip=[
                    alt.Tooltip("low:Q", title="GFS 10th percentile", format=".1f"),
                    alt.Tooltip("high:Q", title="GFS 90th percentile", format=".1f"),
                ],
            )
        )

    point_rows = []
    for label, value in (("GFS median", median), ("Projected high", projected), ("Observed high", observed)):
        if value is not None:
            point_rows.append({"temperature": value, "lane": "Daily high", "Series": label})

    if point_rows:
        points = pd.DataFrame(point_rows)
        layers.append(
            alt.Chart(points)
            .mark_point(filled=True, size=170, stroke="#11121B", strokeWidth=1.5)
            .encode(
                x=alt.X("temperature:Q", title="Temperature (°F)", scale=alt.Scale(domain=[domain_min, domain_max]),
                        axis=alt.Axis(values=list(range(domain_min, domain_max + 1)), tickSize=7, grid=True, labelExpr="datum.value + '°'")),
                y=alt.Y("lane:N", axis=None),
                color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=color_domain, range=color_range)),
                tooltip=[
                    alt.Tooltip("Series:N", title="Measure"),
                    alt.Tooltip("temperature:Q", title="Temperature", format=".1f"),
                ],
            )
        )
        layers.append(
            alt.Chart(points)
            .mark_text(dy=-18, fontSize=13, fontWeight="bold")
            .encode(
                x=alt.X("temperature:Q", scale=alt.Scale(domain=[domain_min, domain_max])),
                y=alt.Y("lane:N", axis=None),
                text=alt.Text("temperature:Q", format=".0f"),
                color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            )
        )

    if not layers:
        return None

    return (
        alt.layer(*layers)
        .resolve_scale(color="shared")
        .properties(
            height=165,
            background="#11121B",
            title=alt.TitleParams(
                text="Forecast range & daily highs",
                subtitle="Reference only: GFS range + median, latest NWS high, and observed high",
                anchor="start",
            ),
        )
        .configure_axis(
            labelFontSize=14, titleFontSize=15, labelColor="#DED9EA", titleColor="#F5F1FA",
            gridColor="#5D5870", gridOpacity=0.22, domainColor="#6E687E", tickColor="#6E687E",
        )
        .configure_legend(
            orient="top", direction="horizontal", title=None, labelFontSize=14,
            labelColor="#F1EDF7", symbolSize=150, columns=2,
        )
        .configure_title(
            fontSize=18, subtitleFontSize=13, color="#FAF7FF", subtitleColor="#C9C3D5",
        )
        .configure_view(strokeWidth=0)
    )

def latest_projection_chart(snapshot_df, observed_df, previous_observed_df, tz_name, target_date):
    """Previous-evening + current-day observations with the newest stored hourly forecast."""
    if snapshot_df.empty:
        return None, None

    tz = ZoneInfo(tz_name)
    latest_key = snapshot_df["snapshot_key"].max()
    latest = snapshot_df[snapshot_df["snapshot_key"] == latest_key].copy()
    latest["time"] = latest["valid_at"].dt.tz_convert(tz)
    latest = latest[latest["time"].dt.date == target_date].sort_values("time")
    if latest.empty:
        return None, latest_key

    forecast_data = latest[["time", "temp_f"]].copy()
    forecast_data["Series"] = "Latest NWS forecast"

    # Keep earlier stored NWS forecast paths for the same target day. These are
    # intentionally faint so the newest forecast remains the visual focus while
    # still showing how the hourly projection evolved through time.
    prior_forecasts = snapshot_df[snapshot_df["snapshot_key"] != latest_key].copy()
    if not prior_forecasts.empty:
        prior_forecasts["time"] = prior_forecasts["valid_at"].dt.tz_convert(tz)
        prior_forecasts = prior_forecasts[prior_forecasts["time"].dt.date == target_date].copy()
        prior_forecasts["snapshot_local"] = prior_forecasts["snapshot_key"].dt.tz_convert(tz)
        prior_forecasts["snapshot_label"] = prior_forecasts["snapshot_local"].dt.strftime("%b %-d, %-I:%M %p")
        prior_forecasts = prior_forecasts.sort_values(["snapshot_key", "time"])

    # Add a little history before midnight so the user can see how the station
    # entered the forecast day. Keep only the final 6 hours to avoid crowding mobile charts.
    prev_obs = pd.DataFrame(columns=["time", "temp_f"])
    if previous_observed_df is not None and not previous_observed_df.empty:
        prev_obs = previous_observed_df.copy()
        prev_obs["time"] = pd.to_datetime(prev_obs["time"], utc=True, errors="coerce").dt.tz_convert(tz)
        prev_obs = prev_obs.dropna(subset=["time", "temp_f"])
        previous_date = target_date - timedelta(days=1)
        cutoff = datetime.combine(previous_date, datetime.min.time(), tzinfo=tz) + timedelta(hours=18)
        prev_obs = prev_obs[
            (prev_obs["time"].dt.date == previous_date)
            & (prev_obs["time"] >= cutoff)
        ].sort_values("time")

    layers = []

    if not prior_forecasts.empty:
        historical_forecasts = (
            alt.Chart(prior_forecasts)
            .mark_line(strokeWidth=1.5, opacity=0.24, color="#B79CFF")
            .encode(
                x=alt.X("time:T", title="Local time", axis=alt.Axis(format="%-I %p", labelAngle=0, tickCount=6, grid=False)),
                y=alt.Y("temp_f:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
                detail="snapshot_key:N",
                tooltip=[
                    alt.Tooltip("snapshot_label:N", title="Earlier NWS snapshot"),
                    alt.Tooltip("time:T", title="Valid time", format="%b %-d, %-I:%M %p"),
                    alt.Tooltip("temp_f:Q", title="Projected", format=".1f"),
                ],
            )
        )
        layers.append(historical_forecasts)

    forecast = (
        alt.Chart(forecast_data)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=36), strokeWidth=2.5, strokeDash=[7, 4])
        .encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%-I %p", labelAngle=0, tickCount=6, grid=False)),
            y=alt.Y("temp_f:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=["Previous day observed", "Observed", "Latest NWS forecast"], range=["#7899A6", "#FF8FCB", "#B79CFF"])),
            tooltip=[
                alt.Tooltip("time:T", title="Time", format="%b %-d, %-I:%M %p"),
                alt.Tooltip("temp_f:Q", title="Projected", format=".1f"),
            ],
        )
    )
    layers.append(forecast)

    if not prev_obs.empty:
        prev_obs["Series"] = "Previous day observed"
        previous_observed = (
            alt.Chart(prev_obs)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=30), strokeWidth=2.2, opacity=0.72)
            .encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%b %-d, %-I %p", labelAngle=-30, tickCount=7, grid=False)),
                y=alt.Y("temp_f:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Series:N", title=None,
                    scale=alt.Scale(
                        domain=["Previous day observed", "Observed", "Latest NWS forecast"],
                        range=["#7899A6", "#FF8FCB", "#B79CFF"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("time:T", title="Previous day", format="%b %-d, %-I:%M %p"),
                    alt.Tooltip("temp_f:Q", title="Observed", format=".1f"),
                ],
            )
        )
        layers.insert(0, previous_observed)

    if observed_df is not None and not observed_df.empty:
        obs = observed_df.copy()
        obs["time"] = pd.to_datetime(obs["time"], utc=True, errors="coerce").dt.tz_convert(tz)
        obs = obs.dropna(subset=["time", "temp_f"])
        obs = obs[obs["time"].dt.date == target_date].sort_values("time")
        if not obs.empty:
            obs["Series"] = "Observed"
            observed = (
                alt.Chart(obs)
                .mark_line(point=alt.OverlayMarkDef(filled=True, size=38), strokeWidth=3)
                .encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%-I %p", labelAngle=0, tickCount=6, grid=False)),
                    y=alt.Y("temp_f:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
                    color=alt.Color("Series:N", title=None, scale=alt.Scale(domain=["Previous day observed", "Observed", "Latest NWS forecast"], range=["#7899A6", "#FF8FCB", "#B79CFF"])),
                    tooltip=[
                        alt.Tooltip("time:T", title="Observed", format="%b %-d, %-I:%M %p"),
                        alt.Tooltip("temp_f:Q", title="Temperature", format=".1f"),
                    ],
                )
            )
            layers.insert(0, observed)

    chart = alt.layer(*layers).resolve_scale(color="shared").properties(
        height=300,
        title=alt.TitleParams(
            text="Temperature through the day",
            subtitle="Observed station readings + latest NWS hourly forecast · faint purple = earlier stored NWS forecasts",
            anchor="start",
        ),
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=15,
        labelColor="#DED9EA",
        titleColor="#F5F1FA",
        gridColor="#5D5870",
        gridOpacity=0.20,
        domainColor="#6E687E",
        tickColor="#6E687E",
    ).configure_legend(
        orient="top",
        direction="horizontal",
        title=None,
        labelFontSize=14,
        labelColor="#F1EDF7",
        symbolSize=130,
    ).configure_title(
        fontSize=18,
        subtitleFontSize=14,
        color="#FAF7FF",
        subtitleColor="#C9C3D5",
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

    st.markdown("<div class='section-kicker'>WEATHER TRAJECTORY</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Latest projected high", "—" if pd.isna(latest_high) else f"{latest_high:.0f}°F")
    m2.metric("Observed high so far", "—" if observed_high is None or pd.isna(observed_high) else f"{observed_high:.0f}°F")
    m3.metric("Stored snapshots", f"{history.shape[0]:,}")

    if latest_chart is not None:
        st.altair_chart(latest_chart, use_container_width=True)
        if latest_key is not None:
            latest_local = latest_key.tz_convert(ZoneInfo(cfg["tz"]))
            st.caption(f"Latest stored projection: {latest_local:%b %-d at %-I:%M %p %Z}.")

    if history_chart is not None:
        st.altair_chart(history_chart, use_container_width=True)
    else:
        st.info("Forecast-history will appear as additional snapshots are collected.")




st.set_page_config(page_title="WeatherEdge", page_icon="🌦️", layout="centered")

st.markdown("""
<style>
:root { --ink:#FAF8FF; --muted:#D2CDDA; --violet:#BCA8FF; --pink:#F4A9D2; --cyan:#8CEAF2; }
.stApp {
  background:
    radial-gradient(circle at 12% 5%, rgba(188,168,255,.15), transparent 31rem),
    radial-gradient(circle at 90% 18%, rgba(140,234,242,.09), transparent 29rem),
    linear-gradient(160deg, #0B0B12 0%, #12111B 48%, #0B1117 100%);
  color: var(--ink);
  font-size: 17px;
  line-height: 1.55;
}
.block-container {max-width: 900px; padding-top: 1.25rem; padding-bottom: 3rem;}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], .stCaption, p, li {
  color:#E7E2ED; font-size:1.02rem; line-height:1.58;
}
h1 {font-size:3rem!important; letter-spacing:-.035em!important; line-height:1.05!important;}
h2 {font-size:2rem!important; letter-spacing:-.02em!important;}
h3 {font-size:1.45rem!important;}
label, [data-testid="stWidgetLabel"] p {color:#F0ECF5!important; font-size:1rem!important; font-weight:650!important;}
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(255,255,255,.075), rgba(188,168,255,.04));
  border: 1px solid rgba(188,168,255,.20); padding: 14px; border-radius: 16px;
  box-shadow: 0 10px 34px rgba(0,0,0,.14);
}
[data-testid="stMetricLabel"] p {color:#D9D3E2!important; font-size:.95rem!important;}
[data-testid="stMetricValue"] {color:#FAF8FF!important; font-size:1.65rem!important;}
[data-testid="stAlert"] p {font-size:1rem!important; line-height:1.5!important; color:#F1EDF7!important;}
[data-testid="stCaptionContainer"] {color:#CEC8D7!important;}
.stButton button, [data-testid="stLinkButton"] a {font-size:1rem!important; font-weight:650!important;}
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) {gap:.7rem;}
[data-testid="stRadio"] > div {gap:.45rem;}
[data-testid="stRadio"] label {
  background:rgba(255,255,255,.055); border:1px solid rgba(188,168,255,.22);
  border-radius:999px; padding:.38rem .72rem; transition:.15s ease;
  color:#F5F1FA!important;
}
[data-testid="stRadio"] label p {font-size:.98rem!important; color:#F5F1FA!important;}
[data-testid="stRadio"] label:hover {border-color:rgba(140,234,242,.62); background:rgba(140,234,242,.08);}
.small-note {font-size:.98rem; color:#CEC8D7; line-height:1.5;}
.card-title {font-size:1.65rem; font-weight:780; margin-bottom:.15rem; letter-spacing:-.02em; color:#FCFAFF;}
.card-sub {font-size:1.05rem; color:#D6D0DF; margin-bottom:.8rem; line-height:1.5;}
.section-kicker {font-size:.78rem; letter-spacing:.14em; color:#C8B8FF; font-weight:780; margin:1.25rem 0 .55rem;}
.bet-shell {padding:.25rem 0 .5rem;}
.bet-callout {
  margin:.35rem 0 .75rem; padding:1rem 1.05rem; border-radius:18px;
  background:linear-gradient(135deg, rgba(183,156,255,.14), rgba(125,231,242,.08));
  border:1px solid rgba(183,156,255,.30); box-shadow:0 12px 34px rgba(0,0,0,.16);
}
.bet-callout-label {font-size:.76rem; letter-spacing:.13em; color:#C8B8FF; font-weight:800; margin-bottom:.28rem;}
.bet-callout-main {font-size:1.28rem; line-height:1.35; color:#FCFAFF; font-weight:760;}
.bet-callout-sub {font-size:.96rem; color:#D6D0DF; margin-top:.28rem;}
code {color:#B9F2D0!important; font-size:.92em!important;}
hr {border-color:rgba(188,168,255,.16)!important;}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ WeatherEdge")
st.caption("NWS-vs-market weather opportunities, with live weather context.")

st.divider()

with st.sidebar:
    scan_mode = st.radio("Scan", ["All preset cities", "One city"], index=0)
    selected_city = st.selectbox("City", list(PRESETS.keys())) if scan_mode == "One city" else None
    top_n = st.slider("Top candidates", 3, 8, 5, 1)
    min_gap = st.slider("Minimum NWS/market gap", 0, 30, 5, 1) / 100

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
    & (df["conservative_prob"] >= 0.55)
    & (df["conservative_edge"] >= min_gap)
].copy()

qualified = qualified.sort_values(
    ["opportunity_score", "conservative_edge", "volume"],
    ascending=[False, False, False],
).head(top_n)

if qualified.empty:
    st.info("No strong forecast-aligned candidates right now.")
else:
    st.subheader("Best candidates")

if not qualified.empty:
    bet_rows = [r for _, r in qualified.iterrows()]
    bet_labels = [
        f"#{i} · {r['city']} · {r['side']} · {r['ask']*100:.0f}¢"
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
        st.metric("NWS / market gap", f"{r['conservative_edge']*100:+.1f} pp")

    mismatch = r.get("temperature_mismatch_f")
    hours_left = r.get("hours_to_settlement")
    mismatch_text = "—" if mismatch is None or pd.isna(mismatch) else f"{mismatch:+.1f}°F"
    time_text = "—" if hours_left is None or pd.isna(hours_left) else (f"{hours_left:.1f}h" if hours_left < 48 else f"{hours_left/24:.1f}d")
    st.info(f"NWS − Kalshi temperature mismatch: **{mismatch_text}** · time to settlement: **{time_text}**")
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
        rejected["Gap"] = rejected["conservative_edge"].map(lambda x: f"{x*100:+.1f} pp")
        st.dataframe(
            rejected[
                ["city", "date_label", "side", "market_subtitle", "Status", "NWS station forecast", "Observed high", "Kalshi implied", "Temp mismatch", "Price", "Gap"]
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

**NWS-based chance** converts that single NWS high into a probability distribution. The uncertainty band automatically narrows as settlement approaches. GFS is not used in this probability.

**Observed high so far** is treated as a hard floor on same-day markets because the final daily high cannot finish below a temperature already observed.

**Kalshi implied temp (approx.)** is reconstructed from the live prices of the event's temperature brackets. It is useful for spotting NWS-vs-market temperature disagreement, but it may differ slightly from the forecast number displayed in Kalshi's app.

**NWS / market gap** is NWS-based contract probability minus the live Kalshi ask.

**Time to settlement** matters because the NWS uncertainty used by WeatherEdge gets tighter as the outcome gets closer. The opportunity ranking prioritizes contract edge, then larger NWS-vs-Kalshi temperature mismatches, with an extra boost as settlement gets close.

The GFS ensemble remains on the forecast-range chart only as optional context. It does **not** affect probabilities, qualification, gaps, or rankings.
"""
    )

st.caption(
    "Research tool only. Forecasts can be wrong, prices can move, and settlement rules matter."
)
