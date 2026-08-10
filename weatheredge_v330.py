
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

# These are convenient starting coordinates for the observing-site area.
# Always verify the settlement source shown in the app before trading.
PRESETS = {
    "New York": {
        "series": "KXHIGHNY",
        "lat": 40.78335, "lon": -73.96497,
        "tz": "America/New_York",
        "station": "Central Park (KNYC)",
        "station_id": "KNYC",
    },
    "Chicago": {
        "series": "KXHIGHCHI",
        "lat": 41.78417, "lon": -87.75528,
        "tz": "America/Chicago",
        "station": "Chicago Midway Airport (KMDW)",
        "station_id": "KMDW",
    },
    "Miami": {
        "series": "KXHIGHMIA",
        # Settlement-aligned anchor: Miami International Airport / KMIA.
        # NWS identifies KMIA at about 25.79 N, 80.32 W.
        "lat": 25.7952, "lon": -80.3254,
        "tz": "America/New_York",
        "station": "Miami International Airport (KMIA / CLIMIA)",
        "station_id": "KMIA",
    },
    "Los Angeles": {
        "series": "KXHIGHLAX",
        "lat": 33.93806, "lon": -118.38889,
        "tz": "America/Los_Angeles",
        "station": "Los Angeles International Airport (KLAX)",
        "station_id": "KLAX",
    },
    "Denver": {
        "series": "KXHIGHDEN",
        "lat": 39.85, "lon": -104.66,
        "tz": "America/Denver",
        "station": "Denver International Airport (KDEN)",
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

@st.cache_data(ttl=20)
def get_kalshi_orderbook(market_ticker):
    """
    Read Kalshi's public live order book and derive bid/ask/spread/depth.

    Kalshi returns YES bids and NO bids. In a binary market:
      YES ask = 1 - best NO bid
      NO ask  = 1 - best YES bid
    """
    data = get_json(f"{KALSHI_BASE}/markets/{market_ticker}/orderbook")
    ob = data.get("orderbook_fp", {})
    yes_levels = ob.get("yes_dollars") or []
    no_levels = ob.get("no_dollars") or []

    def levels(raw):
        out = []
        for row in raw:
            if not row or len(row) < 2:
                continue
            try:
                out.append((float(row[0]), float(row[1])))
            except (TypeError, ValueError):
                continue
        return sorted(out, key=lambda x: x[0])

    yes = levels(yes_levels)
    no = levels(no_levels)

    best_yes_bid = yes[-1][0] if yes else None
    best_yes_bid_size = yes[-1][1] if yes else None
    best_no_bid = no[-1][0] if no else None
    best_no_bid_size = no[-1][1] if no else None

    best_yes_ask = (1.0 - best_no_bid) if best_no_bid is not None else None
    best_yes_ask_size = best_no_bid_size
    best_no_ask = (1.0 - best_yes_bid) if best_yes_bid is not None else None
    best_no_ask_size = best_yes_bid_size

    yes_spread = (
        best_yes_ask - best_yes_bid
        if best_yes_ask is not None and best_yes_bid is not None
        else None
    )
    no_spread = (
        best_no_ask - best_no_bid
        if best_no_ask is not None and best_no_bid is not None
        else None
    )

    # Executable ask-side depth within 5 cents of the best ask.
    # YES asks are generated from NO bids; NO asks are generated from YES bids.
    yes_depth_5c = 0.0
    if best_no_bid is not None:
        cutoff = best_no_bid - 0.05
        yes_depth_5c = sum(qty for price, qty in no if price >= cutoff)

    no_depth_5c = 0.0
    if best_yes_bid is not None:
        cutoff = best_yes_bid - 0.05
        no_depth_5c = sum(qty for price, qty in yes if price >= cutoff)

    return {
        "YES": {
            "bid": best_yes_bid,
            "bid_size": best_yes_bid_size,
            "ask": best_yes_ask,
            "ask_size": best_yes_ask_size,
            "spread": yes_spread,
            "depth_5c": yes_depth_5c,
        },
        "NO": {
            "bid": best_no_bid,
            "bid_size": best_no_bid_size,
            "ask": best_no_ask,
            "ask_size": best_no_ask_size,
            "spread": no_spread,
            "depth_5c": no_depth_5c,
        },
    }


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

@st.cache_data(ttl=300)
def get_nws_daily(lat, lon, tz_name):
    """
    Build the CURRENT predicted daily high from the latest NWS hourly forecast.

    This is intentionally different from simply taking the NWS daytime-period
    headline temperature. We take the maximum hourly forecast temperature for
    each local calendar day so WeatherEdge reflects the latest predicted
    highest temperature for that day.
    """
    point = get_json(f"https://api.weather.gov/points/{lat},{lon}")
    props = point["properties"]
    local_tz = ZoneInfo(tz_name)

    hourly_url = props["forecastHourly"]
    daily_url = props["forecast"]

    hourly = get_json(hourly_url)
    daily = get_json(daily_url)

    # Short forecast text is still useful for display.
    details_by_date = {}
    for period in daily["properties"]["periods"]:
        if period.get("isDaytime"):
            start = datetime.fromisoformat(period["startTime"])
            d = start.astimezone(local_tz).date()
            details_by_date[d] = period.get("shortForecast")

    temps_by_date = {}
    for period in hourly["properties"]["periods"]:
        start = datetime.fromisoformat(period["startTime"])
        d = start.astimezone(local_tz).date()
        temp = period.get("temperature")
        unit = period.get("temperatureUnit")

        if temp is None:
            continue

        temp = float(temp)
        if unit == "C":
            temp = temp * 9 / 5 + 32

        temps_by_date.setdefault(d, []).append(temp)

    rows = []
    for d, temps in sorted(temps_by_date.items()):
        if not temps:
            continue
        rows.append({
            "date": d,
            "nws_high_f": max(temps),
            "nws_detail": details_by_date.get(d),
            "nws_forecast_url": daily_url,
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
        props = feature.get("properties") or {}
        ts = props.get("timestamp")
        if ts:
            try:
                obs_dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(tz)
                if obs_dt.date() != target_date:
                    continue
            except Exception:
                # If a malformed timestamp cannot be verified, do not let it
                # contaminate a date-specific daily-high calculation.
                continue

        value_c = ((props.get("temperature") or {}).get("value"))
        if value_c is not None:
            try:
                vals.append(float(value_c) * 9 / 5 + 32)
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None

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
    # Open-Meteo returns local wall-clock times because `timezone=tz_name`
    # is requested above. Each row is therefore assigned to exactly one local
    # calendar date before any daily maximum is calculated.
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
    ens = get_gfs_ensemble_daily_highs(cfg["lat"], cfg["lon"], cfg["tz"])
    nws = {r["date"]: r for r in get_nws_daily(cfg["lat"], cfg["lon"], cfg["tz"])}

    event_cache = {}
    rows = []

    for m in markets:
        d = infer_market_date(m, cfg["tz"])
        if d is None or d not in ens.index:
            continue

        kind, lo, hi, bracket = market_condition(m)
        if kind is None:
            continue

        # DATE LOCK: only the ensemble daily-high values for this exact
        # Kalshi contract date are allowed into this contract's probability.
        # No prior-day or next-day temperatures are mixed in.
        daily_members = pd.Series(ens.loc[d].values).dropna().astype(float)

        # For same-day markets, the temperature already observed at the settlement
        # station is a hard floor on the eventual daily high. This prevents the model
        # from assigning probability to brackets the station has already exceeded.
        observed_high = None
        try:
            observed_high = get_observed_high_so_far(cfg.get("station_id"), cfg["tz"], d)
        except Exception:
            observed_high = None
        if observed_high is not None and not daily_members.empty:
            daily_members = daily_members.clip(lower=float(observed_high))

        p_yes, n = probability(daily_members.values, kind, lo, hi)
        if p_yes is None:
            continue
        ensemble_median = float(daily_members.median()) if not daily_members.empty else None
        ensemble_low = float(daily_members.quantile(0.10)) if not daily_members.empty else None
        ensemble_high = float(daily_members.quantile(0.90)) if not daily_members.empty else None

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

        try:
            live_book = get_kalshi_orderbook(m.get("ticker"))
        except Exception:
            live_book = {}

        side_data = [
            ("YES", p_yes),
            ("NO", 1 - p_yes),
        ]

        for side, p in side_data:
            book_side = live_book.get(side, {})
            orderbook_ask = to_float(book_side.get("ask"))
            market_fallback_ask = to_float(
                m.get("yes_ask_dollars") if side == "YES" else m.get("no_ask_dollars")
            )
            ask = orderbook_ask if orderbook_ask is not None else market_fallback_ask

            if ask is None or not (0 < ask < 1):
                continue

            bid = to_float(book_side.get("bid"))
            spread = to_float(book_side.get("spread"))
            ask_size = to_float(book_side.get("ask_size"))
            depth_5c = to_float(book_side.get("depth_5c"))
            midpoint = (
                (bid + ask) / 2
                if bid is not None and 0 <= bid <= ask <= 1
                else None
            )
            p_low = wilson_lower(p, n)
            edge = p - ask
            conservative_edge = p_low - ask if p_low is not None else None

            nrow = nws.get(d, {})
            nws_remaining_high = nrow.get("nws_high_f")

            # Today's final high cannot be lower than a temperature already observed.
            # Projected daily high = max(observed high so far, remaining NWS hourly peak).
            if observed_high is not None and not pd.isna(observed_high):
                if nws_remaining_high is None or pd.isna(nws_remaining_high):
                    projected_daily_high = float(observed_high)
                else:
                    projected_daily_high = max(float(observed_high), float(nws_remaining_high))
            else:
                projected_daily_high = nws_remaining_high

            nws_support = side_supported_by_point(projected_daily_high, side, kind, lo, hi)
            median_support = side_supported_by_point(ensemble_median, side, kind, lo, hi)
            forecasts_agree = (nws_support is True and median_support is True)

            # Strict candidate rule: both NWS and ensemble median must support
            # the same side, the ensemble must be meaningfully confident, and
            # the conservative estimate must still exceed the live ask.
            qualifies = (
                forecasts_agree
                and p >= 0.65
                and p_low is not None
                and p_low >= 0.55
                and conservative_edge is not None
                and conservative_edge >= 0.05
            )

            suspicious = (
                conservative_edge is not None and conservative_edge >= 0.30
            )

            rows.append({
                "city": city,
                "station_hint": cfg["station"],
                "station_id": cfg.get("station_id"),
                "nws_public_url": f"https://forecast.weather.gov/MapClick.php?FcstType=graphical&lat={cfg['lat']}&lon={cfg['lon']}&unit=0&lg=english",
                "nws_observed_url": (
                    f"https://www.weather.gov/wrh/timeseries?site={cfg.get('station_id')}"
                    if cfg.get("station_id") else None
                ),
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
                "kalshi_bid": bid,
                "kalshi_mid": midpoint,
                "kalshi_spread": spread,
                "kalshi_ask_size": ask_size,
                "kalshi_depth_5c": depth_5c,
                "price_source": "orderbook" if orderbook_ask is not None else "market snapshot",
                "model_prob": p,
                "conservative_prob": p_low,
                "edge": edge,
                "conservative_edge": conservative_edge,
                "expected_roi": edge / ask,
                "n_members": n,
                "nws_high_f": projected_daily_high,
                "nws_remaining_high_f": nws_remaining_high,
                "observed_high_f": observed_high,
                "nws_forecast": nrow.get("nws_detail"),
                "ensemble_median_f": ensemble_median,
                "ensemble_low_f": ensemble_low,
                "ensemble_high_f": ensemble_high,
                "nws_support": nws_support,
                "median_support": median_support,
                "forecasts_agree": forecasts_agree,
                "qualifies": qualifies,
                "agreement": agreement_label(nws_support, median_support),
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




st.set_page_config(page_title="WeatherEdge", page_icon="🌦️", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp {
    background: radial-gradient(circle at 30% 0%, #0d1722 0%, #070b11 38%, #05080d 100%);
    color: #f6f7fb;
}
.block-container {
    max-width: 1180px;
    padding-top: 1.1rem;
    padding-bottom: 5rem;
}
section[data-testid="stSidebar"] { background:#081019; }

.we-topbar {
    display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.2rem;
}
.we-badge {
    display:inline-block; padding:.45rem .8rem; border-radius:10px;
    background:#072a18; border:1px solid #0d6b37; color:#38e47c;
    font-weight:800; letter-spacing:.03em; font-size:.9rem;
}
.we-actions { display:flex; gap:.55rem; }
.we-action {
    border:1px solid #2c3541; border-radius:10px; padding:.45rem .65rem; color:#f1f5f9;
    background:#0b1118; font-weight:700; font-size:.9rem;
}
.we-hero { display:flex; align-items:center; gap:1rem; margin:.5rem 0 .15rem 0; }
.we-logo { font-size:3.2rem; line-height:1; }
.we-title { font-size:4rem; font-weight:900; letter-spacing:-.05em; line-height:1; }
.we-title .edge { color:#62e879; }
.we-sub { color:#b8c0cc; font-size:1.25rem; margin:1rem 0 2rem 4.5rem; }
.we-divider { border-top:1px solid #2a3340; margin:1.4rem 0 1.8rem; }

.we-section-head { display:flex; align-items:center; gap:.8rem; margin-bottom:.25rem; }
.we-section-icon { color:#27df68; font-size:2rem; }
.we-section-title { font-size:2rem; font-weight:850; }
.we-live {
    margin-left:.4rem; padding:.25rem .55rem; border-radius:9px; background:#0d2d1b;
    color:#3ce87f; font-weight:800; font-size:.9rem;
}
.we-section-sub { color:#a9b2bf; margin-left:3.2rem; margin-bottom:1.4rem; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(19,27,37,.98), rgba(14,20,28,.98));
    border:1px solid #35404d !important;
    border-left:4px solid #27df68 !important;
    border-radius:16px !important;
    box-shadow:0 14px 35px rgba(0,0,0,.22);
}
.we-cardhead { padding:.15rem 0 .7rem; }
.we-rank {
    display:inline-flex; align-items:center; justify-content:center;
    background:#0f3c21; color:#4cef88; min-width:52px; height:48px;
    border-radius:10px; font-size:1.35rem; font-weight:900; margin-right:.75rem;
}
.we-city { font-size:1.9rem; font-weight:900; letter-spacing:-.02em; }
.we-date { color:#c3cad4; font-size:1rem; margin-left:.8rem; }
.we-bet { color:#c9d0da; font-size:1.12rem; margin:.7rem 0; }
.we-bet b { color:#34e577; }
.we-recommend {
    margin:1.15rem 0 .55rem;
    padding:1rem 1rem .95rem;
    border:1px solid #27df68;
    border-radius:14px;
    background:rgba(19,72,39,.22);
}
.we-recommend-label {
    color:#91a0ae;
    font-size:.76rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.08em;
    margin-bottom:.28rem;
}
.we-recommend-bet {
    color:#f5f8fb;
    font-size:1.55rem;
    font-weight:950;
    line-height:1.15;
}
.we-recommend-bet .side { color:#42e77e; }
.we-recommend-note {
    color:#aeb9c5;
    font-size:.82rem;
    margin-top:.35rem;
}
.we-station { color:#c5ccd6; font-size:1.05rem; margin:.4rem 0 1rem; }
.we-dash { border-top:1px dashed #5b6571; opacity:.6; margin:1rem 0; }

.we-nwsbox {
    border:1px solid #27df68; border-radius:10px;
    background:linear-gradient(90deg, rgba(11,63,32,.30), rgba(10,30,20,.18));
    padding:.9rem 1rem; margin:.7rem 0 .25rem;
}

.we-sourcegrid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:.65rem;
    margin:.7rem 0 .9rem;
}
.we-sourcebox {
    border:1px solid #27df68;
    border-radius:10px;
    background:linear-gradient(90deg, rgba(11,63,32,.30), rgba(10,30,20,.18));
    padding:.85rem .9rem;
}
.we-sourcebox.obs {
    border-color:#f0c95a;
    background:linear-gradient(90deg, rgba(90,70,10,.25), rgba(35,28,8,.18));
}
.we-sourcebox a {
    text-decoration:none !important;
    font-weight:850;
    font-size:.98rem;
    display:block;
}
.we-sourcebox.forecast a { color:#51ea85 !important; }
.we-sourcebox.obs a { color:#f4d76c !important; }
.we-sourcehelp {
    color:#98a7b7;
    font-size:.74rem;
    margin-top:.35rem;
}

.we-nwsbox a { color:#51ea85 !important; text-decoration:none !important; font-weight:850; font-size:1.12rem; }
.we-url { color:#38dd72; font-size:.9rem; overflow-wrap:anywhere; margin:.55rem 0 1rem; }

[data-testid="stMetric"] {
    background:#111923;
    border:1px solid #36404b;
    padding:16px 14px;
    border-radius:12px;
    min-height:142px;
}
[data-testid="stMetricLabel"] { color:#c8ced7; font-size:.95rem; }
[data-testid="stMetricValue"] { color:#f8fafc; font-weight:850; }
.metric-caption { color:#aeb6c2; font-size:.85rem; }

.we-probgrid {
    display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1rem 0;
}
.we-probbox {
    background:#121b25; border:1px solid #38424d; border-radius:12px; padding:1.2rem; text-align:center;
}
.we-probtitle { color:#cfd5dd; font-size:1rem; margin-bottom:.45rem; }
.we-probvalue { font-size:2rem; font-weight:900; color:#4ab3ff; }
.we-gapvalue { font-size:2rem; font-weight:900; color:#5fe67f; }
.we-agree { color:#56e77e; font-size:.9rem; margin-top:.45rem; }
.we-warning { color:#f4d20b; font-size:.9rem; margin-top:.45rem; }

div.stLinkButton > a {
    background:#f6f7f9 !important; color:#121820 !important;
    border:1px solid #ffffff !important; border-radius:10px !important;
    font-weight:850 !important; min-height:48px;
}
.we-footerline { color:#c0c7d0; margin-top:.55rem; }
.we-ticker {
    display:inline-block; background:#0d351f; color:#4ae77f; border-radius:8px;
    padding:.18rem .45rem; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}
.we-bottom {
    position:fixed; left:0; right:0; bottom:0; z-index:999;
    display:grid; grid-template-columns:repeat(4,1fr); background:rgba(5,10,15,.96);
    border-top:1px solid #25303a; backdrop-filter:blur(8px); padding:.45rem .7rem .6rem;
}
.we-tab { text-align:center; color:#b8c0ca; font-size:.82rem; }
.we-tab.active {
    background:#0d2f1d; color:#3fe879; border-radius:10px; padding:.45rem;
}
@media (max-width: 760px) {
    .we-title { font-size:2.8rem; }
    .we-sub { margin-left:0; font-size:1rem; }
    .we-topbar { align-items:flex-start; }
    .we-actions { display:none; }
    .we-city { font-size:1.5rem; }
    .we-date { display:block; margin:.35rem 0 0 0; }
    .we-probgrid { grid-template-columns:1fr; }
}

.we-metricgrid {
    display:grid;
    grid-template-columns:repeat(4, minmax(0, 1fr));
    gap:.75rem;
    margin:.9rem 0 1rem;
}
.we-metric {
    background:#111923;
    border:1px solid #36404b;
    border-radius:12px;
    padding:1rem .8rem;
    min-height:126px;
}
.we-metric-icon { font-size:1.05rem; margin-bottom:.55rem; }
.we-metric-label { color:#bfc7d2; font-size:.82rem; line-height:1.15; min-height:2rem; }
.we-metric-value { color:#fff; font-size:1.65rem; font-weight:900; margin-top:.35rem; }
.we-nwsbox {
    box-shadow:0 0 0 1px rgba(39,223,104,.08), 0 8px 24px rgba(0,0,0,.14);
}
.we-nwsbox a {
    display:block;
    width:100%;
}


.we-range-card {
    margin: .9rem 0 1.05rem;
    padding: .9rem 1rem 1rem;
    background: #0d1620;
    border: 1px solid #2d3b49;
    border-radius: 14px;
}
.we-range-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:.75rem;
    margin-bottom:.8rem;
}
.we-range-title {
    color:#e8edf3;
    font-size:.92rem;
    font-weight:800;
}
.we-range-spread {
    color:#9caabc;
    font-size:.78rem;
}
.we-range-axis {
    position:relative;
    height:48px;
    margin:0 .15rem;
}
.we-range-track {
    position:absolute;
    left:0;
    right:0;
    top:19px;
    height:10px;
    background:#202c38;
    border-radius:999px;
    overflow:hidden;
}
.we-range-band {
    position:absolute;
    top:19px;
    height:10px;
    background:linear-gradient(90deg,#3378ff,#7ba8ff);
    border-radius:999px;
}
.we-range-marker {
    position:absolute;
    top:11px;
    width:3px;
    height:26px;
    border-radius:2px;
    transform:translateX(-50%);
}
.we-range-marker.median { background:#dce8ff; }
.we-range-marker.nws { background:#3ee67e; }
.we-range-marker.obs { background:#f4c542; }
.we-range-label {
    position:absolute;
    top:35px;
    transform:translateX(-50%);
    color:#aeb9c6;
    font-size:.70rem;
    white-space:nowrap;
}
.we-range-legend {
    display:flex;
    flex-wrap:wrap;
    gap:.5rem .9rem;
    margin-top:.55rem;
    color:#9eabba;
    font-size:.76rem;
}
.we-dot {
    display:inline-block;
    width:8px;
    height:8px;
    border-radius:50%;
    margin-right:.3rem;
}
.we-dot.band { background:#5b8fff; }
.we-dot.median { background:#dce8ff; }
.we-dot.nws { background:#3ee67e; }
.we-dot.obs { background:#f4c542; }



.we-bookgrid {
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:.65rem;
    margin:.8rem 0 1rem;
}
.we-bookcell {
    background:#101923;
    border:1px solid #334251;
    border-radius:11px;
    padding:.75rem .7rem;
}
.we-booklabel {
    color:#96a5b5;
    font-size:.72rem;
    margin-bottom:.28rem;
}
.we-bookvalue {
    color:#f6f8fb;
    font-size:1.08rem;
    font-weight:850;
}
.we-booknote {
    color:#8c9aaa;
    font-size:.72rem;
    margin-top:.28rem;
}
.we-liquidity-good { color:#52e886; }
.we-liquidity-warn { color:#f0c95a; }
.we-liquidity-bad { color:#ff7e7e; }

.we-axis-tick {
    position:absolute;
    top:16px;
    width:1px;
    height:16px;
    background:#536273;
    transform:translateX(-50%);
}
.we-axis-number {
    position:absolute;
    top:34px;
    transform:translateX(-50%);
    color:#c0cad5;
    font-size:.72rem;
    font-weight:700;
    white-space:nowrap;
}
.we-marker-tag {
    position:absolute;
    top:-10px;
    transform:translateX(-50%);
    padding:.12rem .34rem;
    border-radius:6px;
    font-size:.66rem;
    font-weight:800;
    white-space:nowrap;
    background:#172330;
    border:1px solid #3b4a5a;
    color:#e7edf5;
}
.we-marker-tag.nws { color:#57ed88; border-color:#27884d; }
.we-marker-tag.obs { color:#f6d45e; border-color:#8d782d; }
.we-marker-tag.median { color:#dce8ff; }


/* Top-opportunity navigation: make the collapsed rows the visual focus */
[data-testid="stExpander"] details {
    border: 1px solid #2f3e4c;
    border-radius: 14px;
    background: rgba(11, 19, 28, .78);
    margin-bottom: .75rem;
}
[data-testid="stExpander"] details > summary {
    padding: 1.08rem 1rem !important;
    font-size: 1.34rem !important;
    font-weight: 950 !important;
    color: #ffffff !important;
    line-height: 1.25 !important;
    letter-spacing:-.015em !important;
}
[data-testid="stExpander"] details > summary:hover {
    background: rgba(38, 223, 104, .06);
}
.we-coregrid {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.75rem;
    margin:.9rem 0 .45rem;
}
.we-core {
    background:linear-gradient(180deg, rgba(19,24,38,.96), rgba(12,17,29,.96));
    border:1px solid rgba(111,226,255,.22);
    border-radius:12px;
    padding:.78rem .62rem;
}
.we-core-label {
    color:#c6d0dc;
    font-size:.70rem;
    margin-bottom:.3rem;
    line-height:1.15;
}
.we-core-value {
    color:#ffffff;
    font-size:1.28rem;
    font-weight:900;
    line-height:1.05;
}
.we-core-sub {
    color:#9eabc0;
    font-size:.67rem;
    margin-top:.25rem;
    line-height:1.1;
}
.we-section-mini {
    color:#e8edf3;
    font-size:.95rem;
    font-weight:850;
    margin:1rem 0 .55rem;
}
.we-market-card {
    margin-top:1.2rem;
    padding:1rem;
    border:1px solid #344454;
    border-radius:14px;
    background:#0d1620;
}
.we-market-title {
    color:#eef3f8;
    font-size:1rem;
    font-weight:900;
    margin-bottom:.7rem;
}
.we-marketgrid {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.65rem;
}
.we-marketcell {
    background:#101923;
    border:1px solid #2f3d4a;
    border-radius:10px;
    padding:.75rem;
}
.we-marketlabel {
    color:#93a1b1;
    font-size:.7rem;
    margin-bottom:.25rem;
}
.we-marketvalue {
    color:#f5f8fb;
    font-size:1.02rem;
    font-weight:850;
}

@media (max-width:760px) {
    .we-coregrid { grid-template-columns:repeat(3,minmax(0,1fr)); gap:.45rem; }
    .we-marketgrid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .we-sourcegrid { grid-template-columns:1fr; }
    .we-bookgrid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .we-metricgrid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .we-metric { min-height:112px; }
    .we-metric-value { font-size:1.5rem; }
}

/* restrained vaporwave layer */
.stApp {
    background:
      radial-gradient(circle at 12% 0%, rgba(80, 80, 255, .12), transparent 26rem),
      radial-gradient(circle at 88% 12%, rgba(255, 76, 181, .10), transparent 22rem),
      linear-gradient(180deg, #070914 0%, #090d18 48%, #070b13 100%);
}
.block-container { max-width: 780px; }

.we-title {
    text-shadow:0 0 18px rgba(117,236,255,.16);
}
.we-title .edge {
    color:#77efff;
    text-shadow:0 0 14px rgba(119,239,255,.24), 0 0 28px rgba(255,95,196,.10);
}
.we-divider {
    background:linear-gradient(90deg, transparent, rgba(119,239,255,.35), rgba(255,95,196,.28), transparent);
    height:1px;
    border:0;
}
.we-live {
    color:#ff78cf;
    background:rgba(255,88,196,.08);
    border-color:rgba(255,88,196,.28);
}
[data-testid="stExpander"] details {
    border:1px solid rgba(107, 233, 255, .22);
    background:
      linear-gradient(180deg, rgba(15,22,36,.92), rgba(10,15,27,.92));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.02);
}
[data-testid="stExpander"] details > summary:hover {
    background:linear-gradient(90deg, rgba(91,231,255,.07), rgba(255,94,193,.05));
}
.we-rank {
    background:linear-gradient(180deg, rgba(70,228,255,.18), rgba(70,228,255,.08));
    color:#8cf4ff;
    border:1px solid rgba(99,235,255,.24);
}
.we-recommend {
    border:1px solid rgba(255,94,193,.46);
    background:linear-gradient(135deg, rgba(255,79,185,.10), rgba(88,224,255,.07));
    box-shadow:0 0 0 1px rgba(255,255,255,.01) inset;
}
.we-recommend-label { color:#ff8dd5; }
.we-recommend-bet .side {
    color:#81f3ff;
    text-shadow:0 0 12px rgba(129,243,255,.18);
}
.we-sourcebox.forecast {
    border-color:rgba(103,235,255,.42);
    background:rgba(58,167,196,.08);
}
.we-sourcebox.forecast a { color:#8cf4ff !important; }
.we-sourcebox.obs {
    border-color:rgba(255,101,199,.38);
    background:rgba(170,48,128,.08);
}
.we-sourcebox.obs a { color:#ff9fdc !important; }
.we-probbox {
    border-color:rgba(111,226,255,.22);
    background:linear-gradient(180deg, rgba(18,25,39,.96), rgba(12,18,30,.96));
}
.we-probvalue { color:#8cf4ff; }
.we-gapvalue { color:#ff8fd4; }
.we-market-card {
    border-color:rgba(148,110,255,.22);
    background:linear-gradient(180deg, rgba(14,20,33,.95), rgba(10,15,25,.95));
}
.we-range-card {
    border-color:rgba(111,226,255,.22);
    background:linear-gradient(180deg, rgba(15,22,35,.96), rgba(10,16,27,.96));
}
.we-range-band {
    background:#74e8ff;
    box-shadow:0 0 14px rgba(116,232,255,.20);
}
.we-range-marker.nws { background:#78f2ff; }
.we-range-marker.obs { background:#ff80ce; }
.we-range-marker.median { background:#f6f2ff; }
.we-dot.band { background:#74e8ff; }
.we-dot.nws { background:#78f2ff; }
.we-dot.obs { background:#ff80ce; }
.we-dot.median { background:#f6f2ff; }


@media (max-width:760px) {
    [data-testid="stExpander"] details > summary {
        font-size:1.18rem !important;
        padding:.95rem .8rem !important;
    }
    .we-core { padding:.65rem .45rem; }
    .we-core-label { font-size:.62rem; }
    .we-core-value { font-size:1.08rem; }
    .we-core-sub { font-size:.60rem; }
}


/* v2.8: stronger vaporwave art direction, data-first */
.stApp {
    background:
      linear-gradient(rgba(7,8,22,.88), rgba(7,8,22,.94)),
      repeating-linear-gradient(0deg, transparent 0 31px, rgba(111,226,255,.045) 32px),
      repeating-linear-gradient(90deg, transparent 0 31px, rgba(255,95,196,.035) 32px),
      radial-gradient(ellipse at 50% -8%, rgba(170,90,255,.28), transparent 38rem),
      linear-gradient(180deg, #08091b 0%, #0c0a1b 45%, #070b17 100%);
}
.block-container {
    max-width:800px;
}
.we-title {
    letter-spacing:-.045em;
    text-shadow:
      2px 0 0 rgba(255,82,193,.20),
      -2px 0 0 rgba(91,232,255,.17),
      0 0 28px rgba(137,99,255,.16);
}
.we-title .edge {
    background:linear-gradient(90deg,#78efff,#c795ff,#ff82cf);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}
.we-subtitle {
    letter-spacing:.025em;
    color:#aeb6ca;
}
.we-divider {
    height:2px;
    background:linear-gradient(90deg, transparent 0%, #72eaff 26%, #ad83ff 52%, #ff78cb 76%, transparent 100%);
    opacity:.72;
}
.we-section-title {
    text-transform:uppercase;
    letter-spacing:.09em !important;
    color:#f6f4ff;
    text-shadow:0 0 16px rgba(127,231,255,.12);
}
[data-testid="stExpander"] details {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(122,231,255,.26);
    background:linear-gradient(145deg, rgba(17,23,43,.96), rgba(13,12,34,.95));
}
[data-testid="stExpander"] details::before {
    content:"";
    position:absolute;
    left:0;
    top:0;
    bottom:0;
    width:3px;
    background:linear-gradient(180deg,#74eaff,#a887ff,#ff79cd);
    opacity:.8;
}
[data-testid="stExpander"] details > summary {
    text-shadow:0 0 14px rgba(116,232,255,.08);
}
[data-testid="stExpander"] details > summary:hover {
    background:linear-gradient(90deg,rgba(83,226,255,.10),rgba(170,101,255,.07),rgba(255,91,194,.07));
}
.we-cardhead {
    border-bottom:1px solid rgba(138,121,255,.16);
    padding-bottom:.7rem;
}
.we-recommend {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(255,105,205,.52);
    background:
      linear-gradient(120deg,rgba(255,84,190,.13),rgba(138,97,255,.09) 48%,rgba(81,224,255,.10));
}
.we-recommend::after {
    content:"";
    position:absolute;
    width:120px;
    height:120px;
    right:-65px;
    top:-75px;
    border:1px solid rgba(116,232,255,.18);
    border-radius:50%;
    box-shadow:0 0 0 15px rgba(255,105,205,.025),0 0 0 30px rgba(144,103,255,.02);
}
.we-recommend-label {
    letter-spacing:.13em;
}
.we-recommend-bet {
    text-shadow:0 0 18px rgba(255,255,255,.04);
}
.we-core {
    border-color:rgba(132,224,255,.24);
    background:linear-gradient(155deg,rgba(18,27,48,.96),rgba(18,14,38,.96));
}
.we-core:nth-child(2) {
    border-color:rgba(169,126,255,.30);
}
.we-core:nth-child(3) {
    border-color:rgba(255,116,207,.28);
}
.we-core-value {
    text-shadow:0 0 15px rgba(112,233,255,.10);
}
.we-range-card {
    position:relative;
    overflow:hidden;
    border:1px solid rgba(130,224,255,.25);
    background:linear-gradient(155deg,rgba(15,24,43,.97),rgba(15,12,34,.97));
}
.we-range-card::after {
    content:"";
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:1px;
    background:linear-gradient(90deg,transparent,#74e8ff,#a98cff,#ff7ccc,transparent);
    opacity:.6;
}
.we-range-title {
    text-transform:uppercase;
    letter-spacing:.11em;
    font-size:.8rem;
    color:#dfe9f4;
}
.we-range-track {
    background:rgba(255,255,255,.09);
}
.we-axis-number {
    color:#b7c1d1;
}
.we-section-mini {
    text-transform:uppercase;
    letter-spacing:.11em;
    font-size:.76rem;
    color:#c8d2df;
    padding-left:.55rem;
    border-left:2px solid #a486ff;
}
.we-sourcebox {
    backdrop-filter:blur(3px);
}
.we-probbox {
    background:linear-gradient(155deg,rgba(17,27,46,.96),rgba(17,14,37,.96));
}
.we-market-card {
    position:relative;
    overflow:hidden;
    border-color:rgba(164,128,255,.28);
}
.we-market-title {
    text-transform:uppercase;
    letter-spacing:.11em;
    font-size:.82rem;
}
.we-marketcell {
    background:rgba(13,20,35,.86);
    border-color:rgba(119,221,255,.14);
}
.we-bottom {
    border-top:1px solid rgba(151,124,255,.20);
}


/* Keep opportunity header visible and legible in both collapsed and expanded states */
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] details[open] > summary {
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
    color:#f9fbff !important;
    min-height:3.25rem !important;
}
[data-testid="stExpander"] details[open] > summary {
    border-bottom:1px solid rgba(183,226,255,.14);
    background:linear-gradient(90deg,rgba(111,220,255,.09),rgba(184,145,255,.07),rgba(255,165,222,.06));
}
[data-testid="stExpander"] details[open] > summary p,
[data-testid="stExpander"] details > summary p {
    color:#f9fbff !important;
    opacity:1 !important;
}


/* v2.9 Frosted Dream: translucent, icy, soft-focus vaporwave */
.stApp {
    background:
      radial-gradient(circle at 18% 8%, rgba(174,235,255,.20), transparent 26rem),
      radial-gradient(circle at 82% 18%, rgba(218,178,255,.18), transparent 25rem),
      radial-gradient(circle at 50% 78%, rgba(255,187,226,.10), transparent 32rem),
      linear-gradient(180deg,#090d1d 0%,#0b1022 44%,#0a0d1b 100%);
}
.stApp::before {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    opacity:.30;
    background:
      radial-gradient(circle at 8% 28%, rgba(255,255,255,.20) 0 1px, transparent 1.5px),
      radial-gradient(circle at 72% 12%, rgba(255,255,255,.16) 0 1px, transparent 1.5px),
      radial-gradient(circle at 42% 66%, rgba(201,239,255,.13) 0 1px, transparent 1.5px);
    background-size:83px 83px,117px 117px,151px 151px;
}
.we-title .edge {
    background:linear-gradient(90deg,#c8f7ff,#cbb8ff,#ffd0ed);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    filter:drop-shadow(0 0 8px rgba(190,236,255,.18));
}
[data-testid="stExpander"] details {
    background:linear-gradient(145deg,rgba(220,242,255,.075),rgba(181,169,255,.055));
    border:1px solid rgba(210,239,255,.20);
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,.10),
      0 8px 30px rgba(3,7,18,.14);
    backdrop-filter:blur(16px) saturate(125%);
    -webkit-backdrop-filter:blur(16px) saturate(125%);
}
[data-testid="stExpander"] details::before {
    background:linear-gradient(180deg,rgba(199,246,255,.88),rgba(197,177,255,.72),rgba(255,194,230,.70));
}
.we-recommend,
.we-core,
.we-range-card,
.we-probbox,
.we-market-card,
.we-marketcell,
.we-sourcebox {
    backdrop-filter:blur(14px) saturate(120%);
    -webkit-backdrop-filter:blur(14px) saturate(120%);
}
.we-recommend {
    background:linear-gradient(135deg,rgba(217,242,255,.095),rgba(209,180,255,.08),rgba(255,201,232,.07));
    border-color:rgba(220,241,255,.25);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.09);
}
.we-recommend-label { color:#e8c9ff; }
.we-recommend-bet .side {
    color:#c9f8ff;
    text-shadow:0 0 15px rgba(190,241,255,.24);
}
.we-core {
    background:linear-gradient(155deg,rgba(218,242,255,.075),rgba(184,175,255,.055));
    border-color:rgba(211,239,255,.20);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
}
.we-core:nth-child(2) { border-color:rgba(211,192,255,.22); }
.we-core:nth-child(3) { border-color:rgba(255,207,234,.20); }
.we-range-card {
    background:linear-gradient(150deg,rgba(215,240,255,.07),rgba(180,174,255,.055));
    border-color:rgba(211,239,255,.20);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
}
.we-range-band {
    background:#bcefff;
    box-shadow:0 0 14px rgba(188,239,255,.22);
}
.we-sourcebox.forecast {
    background:rgba(194,239,255,.07);
    border-color:rgba(193,238,255,.24);
}
.we-sourcebox.obs {
    background:rgba(255,202,234,.055);
    border-color:rgba(255,211,238,.20);
}
.we-market-card {
    background:linear-gradient(150deg,rgba(207,235,255,.06),rgba(191,174,255,.05));
    border-color:rgba(211,203,255,.19);
}
.we-marketcell {
    background:rgba(216,239,255,.045);
    border-color:rgba(215,237,255,.12);
}
.we-divider {
    background:linear-gradient(90deg,transparent,rgba(199,244,255,.72),rgba(210,190,255,.55),rgba(255,205,234,.48),transparent);
}
.we-section-mini {
    border-left-color:#cbbaff;
    color:#e3e9f3;
}
.we-live {
    color:#ffd1ed;
    background:rgba(255,205,236,.07);
    border-color:rgba(255,213,239,.20);
}

/* compact glass Model Signal */
.we-signal-strip {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:.45rem;
    margin:.45rem 0 .8rem;
    padding:.45rem;
    border:1px solid rgba(211,239,255,.17);
    border-radius:13px;
    background:linear-gradient(135deg,rgba(216,241,255,.065),rgba(198,181,255,.05));
    backdrop-filter:blur(14px) saturate(120%);
    -webkit-backdrop-filter:blur(14px) saturate(120%);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
}
.we-signal-item {
    min-width:0;
    padding:.5rem .55rem;
    text-align:center;
    border-right:1px solid rgba(220,240,255,.10);
}
.we-signal-item:last-child { border-right:0; }
.we-signal-label {
    display:block;
    color:#9eabc0;
    font-size:.66rem;
    text-transform:uppercase;
    letter-spacing:.07em;
    margin-bottom:.2rem;
}
.we-signal-value {
    display:block;
    color:#f8fbff;
    font-size:1.05rem;
    font-weight:900;
    line-height:1.05;
}
.we-signal-item.accent .we-signal-value {
    color:#d6c6ff;
    text-shadow:0 0 12px rgba(207,188,255,.16);
}

@media (max-width:760px) {
    .we-signal-label { font-size:.58rem; letter-spacing:.04em; }
    .we-signal-value { font-size:.92rem; }
    .we-signal-item { padding:.45rem .25rem; }
}


/* v3.0 Vaporwave Adventure: glassier, richer, playful, still readable */
:root {
    --ice:#dffbff;
    --aqua:#7cecff;
    --lav:#bba6ff;
    --pink:#ff9bd8;
    --deep:#090c1a;
}
.stApp {
    background:
      radial-gradient(circle at 15% 8%, rgba(124,236,255,.22), transparent 22rem),
      radial-gradient(circle at 82% 16%, rgba(187,166,255,.22), transparent 24rem),
      radial-gradient(circle at 58% 72%, rgba(255,155,216,.13), transparent 30rem),
      linear-gradient(180deg,#070a16 0%,#0a1021 44%,#090b18 100%);
}
.stApp::before {
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    opacity:.40;
    background:
      linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.014) 1px, transparent 1px),
      radial-gradient(circle at 30% 20%, rgba(223,251,255,.18) 0 1px, transparent 1.6px),
      radial-gradient(circle at 72% 54%, rgba(255,206,235,.12) 0 1px, transparent 1.6px);
    background-size:34px 34px,34px 34px,113px 113px,151px 151px;
}
.block-container {
    max-width:820px;
    padding-top:1.2rem;
}
.we-hero {
    position:relative;
    padding:1.1rem 1rem 1.25rem;
    border:1px solid rgba(223,251,255,.14);
    border-radius:22px;
    background:linear-gradient(135deg,rgba(223,251,255,.065),rgba(187,166,255,.06),rgba(255,155,216,.05));
    backdrop-filter:blur(18px) saturate(135%);
    -webkit-backdrop-filter:blur(18px) saturate(135%);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.08), 0 18px 50px rgba(2,5,15,.18);
    overflow:hidden;
}
.we-hero::after {
    content:"✦";
    position:absolute;
    right:18px;
    top:10px;
    color:rgba(255,255,255,.22);
    font-size:2rem;
    text-shadow:0 0 24px rgba(185,228,255,.25);
}
.we-logo {
    filter:drop-shadow(0 0 16px rgba(124,236,255,.22));
}
.we-title {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
    font-weight:950;
    letter-spacing:-.055em;
    text-transform:none;
}
.we-title .edge {
    background:linear-gradient(90deg,#dffbff 0%,#85efff 32%,#c1adff 62%,#ffa6dc 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}
.we-sub {
    color:#b8c4d6;
    font-size:.94rem;
    letter-spacing:.02em;
}
.we-section-head {
    margin-top:1.35rem;
    padding:.1rem .2rem .35rem;
}
.we-section-title {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
    letter-spacing:.06em !important;
    font-size:1.45rem !important;
    text-transform:uppercase;
}
.we-live::before { content:"✦ "; }
[data-testid="stExpander"] details {
    border-radius:18px;
    border:1px solid rgba(220,246,255,.20);
    background:
      linear-gradient(145deg,rgba(224,246,255,.07),rgba(197,177,255,.055)),
      rgba(12,17,31,.62);
    backdrop-filter:blur(20px) saturate(145%);
    -webkit-backdrop-filter:blur(20px) saturate(145%);
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,.08),
      0 16px 38px rgba(2,6,18,.14);
}
[data-testid="stExpander"] details > summary {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
    letter-spacing:-.02em !important;
    text-shadow:0 0 18px rgba(190,241,255,.08);
}
[data-testid="stExpander"] details[open] > summary {
    background:linear-gradient(90deg,rgba(124,236,255,.08),rgba(187,166,255,.065),rgba(255,155,216,.055));
}
.we-rank {
    border-radius:14px;
    background:linear-gradient(180deg,rgba(209,248,255,.16),rgba(132,221,255,.08));
    border:1px solid rgba(210,247,255,.20);
    color:#d8faff;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
}
.we-city {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
}
.we-date::before { content:""; }
.we-recommend {
    border-radius:18px;
    border:1px solid rgba(255,209,237,.24);
    background:
      radial-gradient(circle at 92% 5%,rgba(255,255,255,.12),transparent 9rem),
      linear-gradient(135deg,rgba(218,247,255,.09),rgba(190,169,255,.075),rgba(255,174,222,.07));
    backdrop-filter:blur(18px) saturate(145%);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.10), 0 14px 36px rgba(4,7,20,.12);
}
.we-recommend-label::before { content:"✦ "; }
.we-recommend-label {
    color:#ffd2ec;
    font-size:.72rem;
}
.we-recommend-bet {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
    font-size:1.7rem;
}
.we-recommend-bet .side {
    color:#c8f7ff;
    text-shadow:0 0 18px rgba(167,239,255,.18);
}
.we-coregrid {
    gap:.55rem;
}
.we-core {
    border-radius:16px;
    background:
      radial-gradient(circle at 100% 0%,rgba(255,255,255,.07),transparent 8rem),
      linear-gradient(155deg,rgba(219,245,255,.075),rgba(187,174,255,.055));
    border:1px solid rgba(217,242,255,.18);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.07);
}
.we-core:nth-child(1) .we-core-label::before { content:"☁︎ "; }
.we-core:nth-child(2) .we-core-label::before { content:"✧ "; }
.we-core:nth-child(3) .we-core-label::before { content:"☼ "; }
.we-core-value {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
}
.we-range-card {
    border-radius:18px;
    background:
      linear-gradient(150deg,rgba(219,244,255,.065),rgba(184,173,255,.05)),
      rgba(10,16,29,.68);
    backdrop-filter:blur(18px) saturate(140%);
    -webkit-backdrop-filter:blur(18px) saturate(140%);
}
.we-range-title::before { content:"◌ "; color:#bfefff; }
.we-range-title {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
}
.we-range-track {
    background:rgba(255,255,255,.085);
    box-shadow:inset 0 1px 1px rgba(0,0,0,.20);
}
.we-range-band {
    background:#aeeeff;
    box-shadow:0 0 16px rgba(174,238,255,.22);
}
.we-marker-tag {
    background:rgba(16,22,39,.80);
    backdrop-filter:blur(8px);
}
.we-section-mini {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
    border-left:0;
    padding-left:0;
    letter-spacing:.10em;
}
.we-section-mini::before {
    content:"✦ ";
    color:#cdbbff;
}
.we-sourcebox {
    border-radius:16px;
    background:rgba(217,243,255,.055);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
}
.we-sourcebox.forecast a::before { content:"☁️ "; }
.we-sourcebox.obs a::before { content:"🌡️ "; }
.we-sourcebox.forecast a,
.we-sourcebox.obs a {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
}
.we-signal-strip {
    border-radius:16px;
    background:
      linear-gradient(135deg,rgba(222,246,255,.065),rgba(190,176,255,.05));
    box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
}
.we-signal-value {
    font-family:"Trebuchet MS","Arial Rounded MT Bold",system-ui,sans-serif;
}
.we-market-card {
    border-radius:18px;
    background:
      linear-gradient(150deg,rgba(218,242,255,.055),rgba(185,172,255,.045));
}
.we-market-title::before { content:"🪩 "; }
.we-marketcell {
    border-radius:12px;
    background:rgba(219,241,255,.04);
}
.we-bottom {
    backdrop-filter:blur(20px) saturate(130%);
    -webkit-backdrop-filter:blur(20px) saturate(130%);
}
.we-tab.active::before { content:"✦ "; }
@media (max-width:760px) {
    .we-recommend-bet { font-size:1.42rem; }
    .we-title { font-size:2.65rem; }
}


/* v3.1: intentional alignment + stronger vaporwave scene */
.we-topbar {
    display:flex;
    align-items:center;
    justify-content:flex-end;
    width:100%;
}
.we-hero {
    display:grid !important;
    grid-template-columns:auto 1fr;
    align-items:center !important;
    column-gap:.85rem;
    text-align:left !important;
}
.we-logo {
    margin:0 !important;
    line-height:1;
}
.we-title {
    margin:0 !important;
    text-align:left !important;
}
.we-sub {
    grid-column:2;
    margin:.15rem 0 0 !important;
    text-align:left !important;
    max-width:34rem;
}
.we-section-head {
    display:grid !important;
    grid-template-columns:1fr auto;
    align-items:center;
    gap:.75rem;
    width:100%;
}
.we-section-title {
    margin:0 !important;
    text-align:left !important;
}
.we-live {
    justify-self:end;
    margin:0 !important;
}
.we-section-sub {
    text-align:left !important;
    margin-left:0 !important;
    width:100%;
}
.we-cardhead {
    display:grid;
    grid-template-columns:auto 1fr auto;
    align-items:center;
    gap:.75rem;
}
.we-rank { margin:0 !important; }
.we-city { margin:0 !important; }
.we-date {
    margin:0 !important;
    justify-self:end;
    text-align:right;
}
.we-station {
    grid-column:1 / -1;
    margin:.15rem 0 .15rem !important;
}
.we-range-head {
    display:block !important;
}
.we-range-title {
    width:100%;
    text-align:left;
}
.we-section-mini {
    width:100%;
    text-align:left;
}

/* Decorative vaporwave scene: sun, horizon, glass shards, floating orbs */
.vw-scene {
    position:relative;
    height:128px;
    margin:.75rem 0 1rem;
    border:1px solid rgba(216,244,255,.18);
    border-radius:22px;
    overflow:hidden;
    background:
      linear-gradient(180deg, rgba(255,183,230,.10) 0%, rgba(160,130,255,.08) 45%, rgba(82,219,255,.05) 100%),
      rgba(13,16,32,.46);
    backdrop-filter:blur(18px) saturate(135%);
    -webkit-backdrop-filter:blur(18px) saturate(135%);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
}
.vw-sun {
    position:absolute;
    left:50%;
    top:16px;
    width:86px;
    height:86px;
    border-radius:50%;
    transform:translateX(-50%);
    background:
      repeating-linear-gradient(
        to bottom,
        rgba(255,191,226,.95) 0 6px,
        rgba(255,191,226,.95) 6px 9px,
        rgba(255,255,255,0) 9px 13px
      ),
      linear-gradient(180deg,#ffd2ec,#ff96d2);
    box-shadow:0 0 30px rgba(255,151,211,.28);
    opacity:.92;
}
.vw-horizon {
    position:absolute;
    left:0;
    right:0;
    bottom:0;
    height:62px;
    background:
      linear-gradient(rgba(126,233,255,.14) 1px, transparent 1px),
      linear-gradient(90deg, rgba(126,233,255,.12) 1px, transparent 1px);
    background-size:100% 14px, 34px 100%;
    transform:perspective(180px) rotateX(58deg);
    transform-origin:bottom;
    opacity:.9;
}
.vw-palm {
    position:absolute;
    bottom:10px;
    color:rgba(11,14,28,.92);
    font-size:58px;
    filter:drop-shadow(0 0 10px rgba(122,233,255,.05));
}
.vw-palm.left { left:18px; transform:rotate(-8deg); }
.vw-palm.right { right:18px; transform:scaleX(-1) rotate(-8deg); }
.vw-orb {
    position:absolute;
    border-radius:50%;
    border:1px solid rgba(221,248,255,.22);
    background:radial-gradient(circle at 30% 25%,rgba(255,255,255,.26),rgba(165,227,255,.05) 52%,rgba(255,255,255,0) 70%);
    backdrop-filter:blur(5px);
}
.vw-orb.one { width:28px;height:28px;left:18%;top:18px; }
.vw-orb.two { width:18px;height:18px;right:20%;top:32px; }
.vw-shard {
    position:absolute;
    width:70px;
    height:34px;
    border:1px solid rgba(220,241,255,.18);
    background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(179,165,255,.03));
    backdrop-filter:blur(9px);
    transform:skewX(-18deg) rotate(-9deg);
}
.vw-shard.a { left:8%; top:69px; }
.vw-shard.b { right:8%; top:72px; transform:skewX(18deg) rotate(9deg); }

/* More glass, but cleaner geometry */
[data-testid="stExpander"] details,
.we-recommend,
.we-core,
.we-range-card,
.we-sourcebox,
.we-signal-strip,
.we-market-card,
.we-marketcell {
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,.09),
      inset 0 -1px 0 rgba(255,255,255,.025),
      0 12px 34px rgba(2,5,15,.10);
}
[data-testid="stExpander"] details {
    border-color:rgba(215,244,255,.22);
}
.we-core {
    min-height:94px;
}
.we-recommend {
    overflow:hidden;
}
.we-recommend::before {
    content:"";
    position:absolute;
    left:-15%;
    top:-70%;
    width:55%;
    height:220%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
    transform:rotate(16deg);
    pointer-events:none;
}
.we-range-card::before {
    content:"";
    position:absolute;
    width:90px;
    height:90px;
    border-radius:50%;
    right:-35px;
    top:-38px;
    background:radial-gradient(circle,rgba(204,186,255,.10),transparent 68%);
    pointer-events:none;
}
.we-section-mini::before {
    content:"✧ ";
    color:#bdefff;
}
@media (max-width:760px) {
    .we-hero {
        grid-template-columns:auto 1fr;
        column-gap:.65rem;
    }
    .we-title { font-size:2.4rem !important; }
    .we-logo { font-size:1.7rem !important; }
    .we-sub { grid-column:1 / -1; margin-top:.45rem !important; }
    .we-cardhead {
        grid-template-columns:auto 1fr;
    }
    .we-date {
        grid-column:1 / -1;
        justify-self:start;
        text-align:left;
        margin-top:.1rem !important;
    }
    .vw-scene { height:108px; }
    .vw-sun { width:70px;height:70px;top:14px; }
    .vw-palm { font-size:44px; }
}


/* v3.2 Immersive Vaporwave: motifs carried throughout, near-black atmospheric base */
.stApp {
    background:
      radial-gradient(ellipse at 15% 4%, rgba(255,87,188,.15), transparent 28rem),
      radial-gradient(ellipse at 88% 18%, rgba(117,225,255,.12), transparent 30rem),
      radial-gradient(ellipse at 48% 62%, rgba(159,104,255,.09), transparent 34rem),
      linear-gradient(180deg,#050508 0%,#090711 42%,#05060a 100%) !important;
}
.stApp::before {
    opacity:.28 !important;
    background:
      radial-gradient(circle at 20% 12%, rgba(255,226,246,.16) 0 1px, transparent 1.4px),
      radial-gradient(circle at 76% 33%, rgba(201,244,255,.14) 0 1px, transparent 1.4px),
      radial-gradient(circle at 40% 78%, rgba(224,210,255,.10) 0 1px, transparent 1.4px) !important;
    background-size:127px 127px,173px 173px,211px 211px !important;
}
.stApp::after {
    content:"";
    position:fixed;
    left:0; right:0; bottom:-80px;
    height:300px;
    pointer-events:none;
    opacity:.20;
    background:
      linear-gradient(rgba(117,232,255,.22) 1px,transparent 1px),
      linear-gradient(90deg,rgba(255,112,202,.15) 1px,transparent 1px);
    background-size:100% 30px,56px 100%;
    transform:perspective(380px) rotateX(64deg);
    transform-origin:bottom;
    mask-image:linear-gradient(to top,black,transparent 82%);
    -webkit-mask-image:linear-gradient(to top,black,transparent 82%);
}
.block-container { position:relative; z-index:1; }

/* Remove spinner-like hero icon and center composition cleanly */
.we-hero {
    display:block !important;
    text-align:center !important;
    padding:1.3rem 1rem 1.15rem !important;
}
.we-logo { display:none !important; }
.we-title {
    text-align:center !important;
    margin:0 auto !important;
}
.we-sub {
    display:flex;
    justify-content:center;
    align-items:center;
    gap:.65rem;
    grid-column:auto !important;
    margin:.35rem auto 0 !important;
    text-align:center !important;
    max-width:38rem;
}
.we-sub::before,
.we-sub::after {
    content:"✦";
    color:rgba(221,194,255,.72);
    font-size:.72rem;
    text-shadow:0 0 10px rgba(205,176,255,.30);
}

/* Carry the visual scene language through every major section */
.we-section-head {
    position:relative;
    overflow:hidden;
    padding:.75rem .85rem !important;
    border:1px solid rgba(222,241,255,.13);
    border-radius:15px;
    background:
      linear-gradient(135deg,rgba(255,123,204,.045),rgba(124,225,255,.04)),
      rgba(255,255,255,.018);
    backdrop-filter:blur(14px);
}
.we-section-head::after {
    content:"";
    position:absolute;
    right:-18px;
    bottom:-18px;
    width:82px;
    height:50px;
    opacity:.30;
    background:
      linear-gradient(rgba(122,231,255,.28) 1px,transparent 1px),
      linear-gradient(90deg,rgba(255,123,207,.22) 1px,transparent 1px);
    background-size:100% 10px,16px 100%;
    transform:perspective(90px) rotateX(55deg);
}
[data-testid="stExpander"] details {
    position:relative;
}
[data-testid="stExpander"] details::after {
    content:"✦";
    position:absolute;
    right:14px;
    bottom:10px;
    color:rgba(203,184,255,.10);
    font-size:1.1rem;
    pointer-events:none;
}
.we-recommend::after {
    content:"🌴" !important;
    width:auto !important; height:auto !important;
    right:16px !important; top:auto !important; bottom:7px !important;
    border:0 !important; border-radius:0 !important; box-shadow:none !important;
    font-size:2.5rem;
    opacity:.10;
    filter:grayscale(1);
}
.we-core {
    position:relative;
    overflow:hidden;
}
.we-core::after {
    content:"";
    position:absolute;
    right:-14px;
    bottom:-14px;
    width:48px;
    height:48px;
    border-radius:50%;
    border:1px solid rgba(222,245,255,.08);
    box-shadow:0 0 0 8px rgba(198,175,255,.018);
}
.we-range-card::after {
    height:2px !important;
    background:linear-gradient(90deg,transparent,#8befff,#c2a8ff,#ff9edb,transparent) !important;
}
.we-sourcegrid {
    position:relative;
}
.we-sourcegrid::before {
    content:"";
    position:absolute;
    left:8%; right:8%; top:-5px;
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(139,238,255,.18),rgba(255,158,219,.14),transparent);
}
.we-signal-strip {
    position:relative;
    overflow:hidden;
}
.we-signal-strip::after {
    content:"";
    position:absolute;
    width:62px; height:62px;
    right:-24px; top:-27px;
    border-radius:50%;
    background:repeating-linear-gradient(to bottom,rgba(255,164,220,.11) 0 4px,transparent 4px 7px);
}
.we-market-card::after {
    content:"";
    position:absolute;
    left:0; right:0; bottom:0;
    height:34px;
    opacity:.14;
    background:
      linear-gradient(rgba(119,231,255,.24) 1px,transparent 1px),
      linear-gradient(90deg,rgba(255,128,209,.18) 1px,transparent 1px);
    background-size:100% 9px,24px 100%;
    transform:perspective(100px) rotateX(55deg);
    transform-origin:bottom;
    pointer-events:none;
}
.we-bottom::before {
    content:"🌴  ✦  ◇  ✦  🌴";
    display:block;
    text-align:center;
    margin-bottom:.55rem;
    opacity:.34;
    letter-spacing:.55rem;
    filter:grayscale(.35);
}

/* Frosted glass gets a brighter edge against black */
.we-hero,
.vw-scene,
[data-testid="stExpander"] details,
.we-recommend,
.we-core,
.we-range-card,
.we-sourcebox,
.we-signal-strip,
.we-market-card {
    border-color:rgba(221,244,255,.17) !important;
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,.08),
      0 16px 44px rgba(0,0,0,.20) !important;
}
.we-hero {
    background:
      radial-gradient(circle at 50% 0%,rgba(255,168,222,.08),transparent 14rem),
      linear-gradient(135deg,rgba(218,246,255,.045),rgba(191,165,255,.045),rgba(255,157,218,.035)) !important;
}
.vw-scene {
    background:
      radial-gradient(ellipse at 50% 35%,rgba(255,139,209,.10),transparent 13rem),
      linear-gradient(180deg,rgba(255,143,214,.055),rgba(143,111,255,.045) 48%,rgba(91,224,255,.035)),
      rgba(5,5,10,.48) !important;
}
.we-market-card { position:relative; overflow:hidden; }

@media (max-width:760px) {
    .we-hero { padding:1.05rem .75rem 1rem !important; }
    .we-sub { font-size:.82rem; gap:.45rem; }
    .we-section-head { padding:.65rem .7rem !important; }
}


/* v3.3 NOIR VAPORWAVE: black smoked glass + integrated hero + richer card scenery */
.stApp {
    background:
      radial-gradient(ellipse at 12% 8%, rgba(255,72,178,.13), transparent 26rem),
      radial-gradient(ellipse at 88% 15%, rgba(196,116,255,.11), transparent 27rem),
      radial-gradient(ellipse at 52% 55%, rgba(255,143,193,.055), transparent 34rem),
      radial-gradient(ellipse at 78% 78%, rgba(112,238,255,.045), transparent 28rem),
      linear-gradient(180deg,#020203 0%,#070508 44%,#020203 100%) !important;
}
.stApp::after {
    opacity:.12 !important;
}
.block-container { max-width:820px; }

/* One integrated album-cover-style hero */
.we-hero {
    position:relative;
    min-height:205px;
    display:flex !important;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center !important;
    overflow:hidden;
    border-radius:24px !important;
    background:
      radial-gradient(circle at 50% 47%,rgba(255,118,194,.10),transparent 11rem),
      linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.012)) !important;
}
.we-hero::before {
    content:"";
    position:absolute;
    left:50%;
    top:25px;
    width:112px;
    height:112px;
    border-radius:50%;
    transform:translateX(-50%);
    background:
      repeating-linear-gradient(to bottom,
        rgba(255,167,215,.68) 0 7px,
        rgba(255,167,215,.68) 7px 10px,
        transparent 10px 15px);
    box-shadow:0 0 40px rgba(255,104,190,.15);
    opacity:.42;
    z-index:0;
}
.we-hero::after {
    content:"";
    position:absolute;
    left:-8%; right:-8%; bottom:-42px;
    height:100px;
    background:
      linear-gradient(rgba(142,236,255,.19) 1px,transparent 1px),
      linear-gradient(90deg,rgba(255,111,199,.15) 1px,transparent 1px);
    background-size:100% 15px,36px 100%;
    transform:perspective(150px) rotateX(59deg);
    transform-origin:bottom;
    opacity:.50;
    z-index:0;
}
.we-title,.we-sub { position:relative; z-index:2; }
.we-title {
    margin-top:8px !important;
    font-size:3.35rem !important;
    text-shadow:0 2px 24px rgba(0,0,0,.62);
}
.we-sub {
    background:rgba(3,3,4,.32);
    border:1px solid rgba(255,255,255,.07);
    border-radius:999px;
    padding:.38rem .75rem;
    backdrop-filter:blur(9px);
}
.we-sub::before,.we-sub::after { content:"✦"; }

/* Hide old separate decorative scene: its language now lives inside hero/cards */
.vw-scene { display:none !important; }

/* Near-black smoked glass instead of navy */
[data-testid="stExpander"] details,
.we-recommend,.we-core,.we-range-card,.we-sourcebox,
.we-signal-strip,.we-market-card,.we-marketcell,.we-section-head {
    background-color:rgba(5,4,6,.60) !important;
}
[data-testid="stExpander"] details {
    background:
      radial-gradient(circle at 94% 12%,rgba(255,117,199,.055),transparent 9rem),
      radial-gradient(circle at 5% 88%,rgba(130,235,255,.035),transparent 10rem),
      linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.015)),
      rgba(4,3,5,.72) !important;
    border-color:rgba(255,223,242,.15) !important;
}
[data-testid="stExpander"] details::before {
    width:2px !important;
    background:linear-gradient(180deg,#ff83c9,#b790ff,#8cecff) !important;
    opacity:.72 !important;
}
/* pronounced but non-interfering visual details in expanded cards */
[data-testid="stExpander"] details[open]::after {
    content:"";
    position:absolute;
    right:-34px;
    top:86px;
    width:120px;
    height:120px;
    border-radius:50%;
    background:
      repeating-linear-gradient(to bottom,rgba(255,140,205,.10) 0 5px,transparent 5px 9px);
    opacity:.52;
    pointer-events:none;
}
.we-cardhead {
    position:relative;
}
.we-cardhead::after {
    content:"◇  ✦  ◇";
    position:absolute;
    right:2px;
    bottom:-4px;
    color:rgba(221,190,255,.19);
    letter-spacing:.3rem;
    font-size:.68rem;
}
.we-recommend {
    background:
      linear-gradient(125deg,rgba(255,121,198,.075),rgba(186,139,255,.05),rgba(117,234,255,.035)),
      rgba(5,4,6,.66) !important;
}
.we-recommend::after {
    opacity:.18 !important;
    filter:grayscale(.25) saturate(.75) !important;
}
.we-core {
    background:
      radial-gradient(circle at 100% 0%,rgba(255,255,255,.06),transparent 6rem),
      rgba(7,6,8,.68) !important;
}
.we-core:nth-child(1) { border-color:rgba(255,163,214,.15) !important; }
.we-core:nth-child(2) { border-color:rgba(192,154,255,.17) !important; }
.we-core:nth-child(3) { border-color:rgba(136,235,255,.14) !important; }
.we-range-card {
    background:
      radial-gradient(circle at 88% 5%,rgba(193,153,255,.06),transparent 10rem),
      rgba(6,5,7,.68) !important;
}
.we-range-card::before {
    width:120px !important;height:120px !important;
    background:
      repeating-linear-gradient(to bottom,rgba(255,139,205,.07) 0 5px,transparent 5px 9px) !important;
}
.we-sourcebox {
    background:rgba(8,7,9,.66) !important;
}
.we-signal-strip {
    background:
      linear-gradient(90deg,rgba(255,122,196,.035),rgba(183,144,255,.035),rgba(126,234,255,.025)),
      rgba(7,6,8,.66) !important;
}
.we-market-card {
    background:
      radial-gradient(circle at 96% 0%,rgba(255,131,202,.05),transparent 11rem),
      rgba(6,5,7,.72) !important;
}
.we-marketcell { background:rgba(255,255,255,.025) !important; }

/* More intentional section ornaments */
.we-section-mini {
    display:flex;
    align-items:center;
    gap:.55rem;
}
.we-section-mini::before {
    content:"";
    width:26px;
    height:1px;
    background:linear-gradient(90deg,#ff8dcc,#bba2ff);
}
.we-section-mini::after {
    content:"";
    flex:1;
    height:1px;
    background:linear-gradient(90deg,rgba(139,235,255,.22),transparent);
}
.we-section-head::before {
    content:"";
    position:absolute;
    left:14px;
    bottom:0;
    width:88px;
    height:1px;
    background:linear-gradient(90deg,#ff89c9,#b696ff,transparent);
}

/* Remove confusing bottom decoration entirely */
.we-bottom::before { content:none !important; display:none !important; }

/* Keep bottom calm and clean */
.we-bottom {
    border-top:1px solid rgba(255,255,255,.07) !important;
    background:transparent !important;
}

@media (max-width:760px) {
    .we-hero { min-height:180px; }
    .we-title { font-size:2.65rem !important; }
    .we-hero::before { width:94px;height:94px;top:24px; }
    .we-sub { font-size:.76rem; padding:.32rem .58rem; }
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="we-topbar">
  <div class="we-actions">
    <div class="we-action">↗ Share</div>
    <div class="we-action">☆</div>
    <div class="we-action">◉</div>
    <div class="we-action">☼</div>
  </div>
</div>
<div class="we-hero">
  <div class="we-logo">◌</div>
  <div class="we-title">Weather<span class="edge">Edge</span></div>
</div>
<div class="we-sub">Weather markets, refracted through forecast data.</div>
<div class="we-divider"></div>
<div class="vw-scene" aria-hidden="true">
  <div class="vw-sun"></div>
  <div class="vw-orb one"></div>
  <div class="vw-orb two"></div>
  <div class="vw-shard a"></div>
  <div class="vw-shard b"></div>
  <div class="vw-palm left">🌴</div>
  <div class="vw-palm right">🌴</div>
  <div class="vw-orb" style="width:12px;height:12px;left:35%;top:16px;"></div>
  <div class="vw-orb" style="width:9px;height:9px;right:33%;top:21px;"></div>
  <div class="vw-horizon"></div>
</div>
<div class="we-section-head">
  <div class="we-section-title" style="font-size:1.35rem;font-weight:950;letter-spacing:-.02em;">Top opportunities ✨</div>
  <div class="we-live">Live</div>
</div>
<div class="we-section-sub">Sorted by model/market gap, strongest first.</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Scanner")
    scan_mode = st.radio("Scan", ["All preset cities", "One city"], index=0)
    selected_city = st.selectbox("City", list(PRESETS.keys())) if scan_mode == "One city" else None
    top_n = st.slider("Top candidates", 3, 8, 5, 1)
    min_gap = st.slider("Minimum model/market gap", 0, 30, 5, 1) / 100

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
    (df["forecasts_agree"] == True)
    & (df["model_prob"] >= 0.65)
    & (df["conservative_prob"] >= 0.55)
    & (df["conservative_edge"] >= min_gap)
].copy()

qualified = qualified.sort_values(
    ["conservative_edge", "model_prob", "volume"],
    ascending=[False, False, False],
).head(top_n)

if qualified.empty:
    st.info("No strong forecast-aligned candidates right now.")

for rank, (_, r) in enumerate(qualified.iterrows(), start=1):
    expander_label = (
        f"#{rank}  {r['city']} • {r['date_label']}  |  "
        f"{r['side']} {r['market_subtitle']}  |  +{max(0, r['conservative_edge']*100):.1f} pp"
    )

    with st.expander(expander_label, expanded=False):
        observed_value = r.get("observed_high_f")
        nws_txt = "—" if pd.isna(r["nws_high_f"]) else f"{int(round(float(r['nws_high_f'])))}°F"
        obs_txt = (
            "—"
            if observed_value is None or pd.isna(observed_value)
            else f"{int(round(float(observed_value)))}°F"
        )
        med_txt = "—" if pd.isna(r["ensemble_median_f"]) else f"{r['ensemble_median_f']:.1f}°F"
        low_txt = "—" if pd.isna(r["ensemble_low_f"]) else f"{r['ensemble_low_f']:.1f}°F"
        high_txt = "—" if pd.isna(r["ensemble_high_f"]) else f"{r['ensemble_high_f']:.1f}°F"
        range_txt = "—" if low_txt == "—" or high_txt == "—" else f"{low_txt}–{high_txt}"

        price_txt = f"{r['ask']*100:.0f}¢"
        bid_v = r.get("kalshi_bid")
        mid_v = r.get("kalshi_mid")
        spread_v = r.get("kalshi_spread")
        ask_size_v = r.get("kalshi_ask_size")
        depth_v = r.get("kalshi_depth_5c")

        bid_txt = "—" if bid_v is None or pd.isna(bid_v) else f"{bid_v*100:.0f}¢"
        mid_txt = "—" if mid_v is None or pd.isna(mid_v) else f"{mid_v*100:.1f}¢"
        spread_txt = "—" if spread_v is None or pd.isna(spread_v) else f"{spread_v*100:.1f}¢"
        ask_size_txt = "—" if ask_size_v is None or pd.isna(ask_size_v) else f"{ask_size_v:.0f}"
        depth_txt = "—" if depth_v is None or pd.isna(depth_v) else f"{depth_v:.0f}"

        if spread_v is None or pd.isna(spread_v):
            liquidity_label = "Unknown"
            liquidity_class = "we-liquidity-warn"
        elif spread_v <= 0.03 and (depth_v or 0) >= 50:
            liquidity_label = "Good"
            liquidity_class = "we-liquidity-good"
        elif spread_v <= 0.08 and (depth_v or 0) >= 15:
            liquidity_label = "Fair"
            liquidity_class = "we-liquidity-warn"
        else:
            liquidity_label = "Thin / wide"
            liquidity_class = "we-liquidity-bad"

        # ---- 1. Recommendation first ----
        st.markdown(
            f"""
            <div class="we-cardhead">
              <span class="we-rank">#{rank}</span>
              <span class="we-city">{r['city']}</span>
              <span class="we-date">📅 {r['date_label']}</span>
              <div class="we-station">⌖ {r['station_hint']}</div>
            </div>
            <div class="we-recommend">
              <div class="we-recommend-label">Recommended bet</div>
              <div class="we-recommend-bet"><span class="side">{r['side']}</span> on “{r['market_subtitle']}”</div>
              <div class="we-recommend-note">Live Kalshi ask {price_txt} • Model chance {fmt_pct(r['model_prob'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button(
            "📊 Open exact market on Kalshi ↗",
            r["kalshi_event_url"],
            use_container_width=True,
        )

        # ---- 2. Core weather evidence ----
        st.markdown('<div class="we-section-mini">Temperature outlook</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="we-coregrid">
              <div class="we-core">
                <div class="we-core-label">Projected Daily High</div>
                <div class="we-core-value">{nws_txt}</div>
                <div class="we-core-sub">NWS forecast + observed floor</div>
              </div>
              <div class="we-core">
                <div class="we-core-label">GFS Model Median + Range</div>
                <div class="we-core-value">{med_txt}</div>
                <div class="we-core-sub">{range_txt}</div>
              </div>
              <div class="we-core">
                <div class="we-core-label">Observed High So Far</div>
                <div class="we-core-value">{obs_txt}</div>
                <div class="we-core-sub">Exact settlement station</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual comparison strip with readable temperature ticks.
        low_v = None if pd.isna(r["ensemble_low_f"]) else float(r["ensemble_low_f"])
        high_v = None if pd.isna(r["ensemble_high_f"]) else float(r["ensemble_high_f"])
        median_v = None if pd.isna(r["ensemble_median_f"]) else float(r["ensemble_median_f"])
        nws_v = None if pd.isna(r["nws_high_f"]) else float(r["nws_high_f"])
        obs_v = None if observed_value is None or pd.isna(observed_value) else float(observed_value)

        if low_v is not None and high_v is not None:
            visible_vals = [low_v, high_v]
            for value in (median_v, nws_v, obs_v):
                if value is not None:
                    visible_vals.append(value)

            axis_min = math.floor(min(visible_vals) - 1)
            axis_max = math.ceil(max(visible_vals) + 1)
            if axis_max <= axis_min:
                axis_max = axis_min + 2
            span = axis_max - axis_min
            tick_step = 1 if span <= 10 else 2
            axis_min = tick_step * math.floor(axis_min / tick_step)
            axis_max = tick_step * math.ceil(axis_max / tick_step)

            def pct(v):
                return max(0.0, min(100.0, 100.0 * (v - axis_min) / (axis_max - axis_min)))

            band_left = pct(low_v)
            band_width = max(1.0, pct(high_v) - band_left)
            median_pos = pct(median_v) if median_v is not None else None
            nws_pos = pct(nws_v) if nws_v is not None else None
            obs_pos = pct(obs_v) if obs_v is not None else None
            spread = high_v - low_v

            tick_html = ""
            t = axis_min
            while t <= axis_max + 0.001:
                pos = pct(t)
                tick_html += (
                    f'<div class="we-axis-tick" style="left:{pos:.2f}%"></div>'
                    f'<div class="we-axis-number" style="left:{pos:.2f}%">{t:.0f}°</div>'
                )
                t += tick_step

            marker_html = ""
            if median_pos is not None:
                marker_html += (
                    f'<div class="we-range-marker median" style="left:{median_pos:.2f}%"></div>'
                    f'<div class="we-marker-tag median" style="left:{median_pos:.2f}%">GFS {median_v:.1f}°</div>'
                )
            if nws_pos is not None:
                marker_html += (
                    f'<div class="we-range-marker nws" style="left:{nws_pos:.2f}%"></div>'
                    f'<div class="we-marker-tag nws" style="left:{nws_pos:.2f}%">Proj {nws_v:.0f}°</div>'
                )
            if obs_pos is not None and (nws_v is None or abs(obs_v - nws_v) >= 0.5):
                marker_html += (
                    f'<div class="we-range-marker obs" style="left:{obs_pos:.2f}%"></div>'
                    f'<div class="we-marker-tag obs" style="left:{obs_pos:.2f}%">Obs {obs_v:.0f}°</div>'
                )

            st.markdown(
                f"""
                <div class="we-range-card">
                  <div class="we-range-head">
                    <div class="we-range-title" style="width:100%;">Forecast comparison</div>
                  </div>
                  <div class="we-range-axis" style="height:64px;margin-top:.55rem;">
                    <div class="we-range-track"></div>
                    {tick_html}
                    <div class="we-range-band" style="left:{band_left:.2f}%;width:{band_width:.2f}%"></div>
                    {marker_html}
                  </div>
                  <div class="we-range-legend" style="margin-top:.85rem;">
                    <span><i class="we-dot band"></i>GFS 80% range</span>
                    <span><i class="we-dot median"></i>GFS median</span>
                    <span><i class="we-dot nws"></i>Projected high</span>
                    <span><i class="we-dot obs"></i>Observed high</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- 3. Source links immediately after weather picture ----
        st.markdown('<div class="we-section-mini">Verify the weather data</div>', unsafe_allow_html=True)
        nws_url = r["nws_public_url"]
        observed_url = r.get("nws_observed_url")
        observed_link_html = (
            f"""
            <div class="we-sourcebox obs">
              <a href="{observed_url}" target="_blank">🌡️ Open NWS observed data ↗</a>
              <div class="we-sourcehelp">Station {r.get('station_id')} observations for the contract date.</div>
            </div>
            """
            if observed_url else ""
        )
        st.markdown(
            f"""
            <div class="we-sourcegrid">
              <div class="we-sourcebox forecast">
                <a href="{nws_url}" target="_blank">🌤️ Open NWS forecast data ↗</a>
                <div class="we-sourcehelp">Forecast hours for this settlement location and contract date.</div>
              </div>
              {observed_link_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- 4. Probability summary ----
        st.markdown('<div class="we-section-mini">Model signal</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="we-signal-strip">
              <div class="we-signal-item">
                <span class="we-signal-label">Model chance</span>
                <span class="we-signal-value">{fmt_pct(r['model_prob'])}</span>
              </div>
              <div class="we-signal-item">
                <span class="we-signal-label">Conservative</span>
                <span class="we-signal-value">{fmt_pct(r['conservative_prob'])}</span>
              </div>
              <div class="we-signal-item accent">
                <span class="we-signal-label">Market gap</span>
                <span class="we-signal-value">{r['conservative_edge']*100:+.1f} pp</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- 5. Market details last, cleaned up ----
        st.markdown(
            f"""
            <div class="we-market-card">
              <div class="we-market-title">Market details</div>
              <div class="we-marketgrid">
                <div class="we-marketcell">
                  <div class="we-marketlabel">Best Bid</div>
                  <div class="we-marketvalue">{bid_txt}</div>
                </div>
                <div class="we-marketcell">
                  <div class="we-marketlabel">Best Ask</div>
                  <div class="we-marketvalue">{price_txt}</div>
                </div>
                <div class="we-marketcell">
                  <div class="we-marketlabel">Midpoint</div>
                  <div class="we-marketvalue">{mid_txt}</div>
                </div>
                <div class="we-marketcell">
                  <div class="we-marketlabel">Bid–Ask Spread</div>
                  <div class="we-marketvalue">{spread_txt}</div>
                </div>
                <div class="we-marketcell">
                  <div class="we-marketlabel">Liquidity</div>
                  <div class="we-marketvalue {liquidity_class}">{liquidity_label}</div>
                </div>
                <div class="we-marketcell">
                  <div class="we-marketlabel">Depth Near Ask</div>
                  <div class="we-marketvalue">{depth_txt}</div>
                </div>
              </div>
              <div class="we-footerline" style="margin-top:.8rem;">
                {ask_size_txt} contracts at the best ask • ticker
                <span class="we-ticker">{r['market_ticker']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"📅 Date lock: every weather input above is restricted to {r['date_label']} "
            f"in the settlement location's local timezone."
        )

st.markdown("""
<div class="we-bottom">
  <div class="we-tab active">◎<br>Top Picks</div>
  <div class="we-tab">▤<br>All Markets</div>
  <div class="we-tab">☆<br>Watchlist</div>
  <div class="we-tab">ⓘ<br>How It Works</div>
</div>
""", unsafe_allow_html=True)
