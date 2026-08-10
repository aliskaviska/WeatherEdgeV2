
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
        "lat": 40.7812, "lon": -73.9665,
        "tz": "America/New_York",
        "station": "Central Park / NYC settlement area",
        "station_id": "KNYC",
    },
    "Chicago": {
        "series": "KXHIGHCHI",
        "lat": 41.7868, "lon": -87.7522,
        "tz": "America/Chicago",
        "station": "Chicago Midway / KMDW settlement area",
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

@st.cache_data(ttl=300)
def get_nws_daily(lat, lon):
    """
    Build the CURRENT predicted daily high from the latest NWS hourly forecast.

    This is intentionally different from simply taking the NWS daytime-period
    headline temperature. We take the maximum hourly forecast temperature for
    each local calendar day so WeatherEdge reflects the latest predicted
    highest temperature for that day.
    """
    point = get_json(f"https://api.weather.gov/points/{lat},{lon}")
    props = point["properties"]

    hourly_url = props["forecastHourly"]
    daily_url = props["forecast"]

    hourly = get_json(hourly_url)
    daily = get_json(daily_url)

    # Short forecast text is still useful for display.
    details_by_date = {}
    for period in daily["properties"]["periods"]:
        if period.get("isDaytime"):
            d = datetime.fromisoformat(period["startTime"]).date()
            details_by_date[d] = period.get("shortForecast")

    temps_by_date = {}
    for period in hourly["properties"]["periods"]:
        start = datetime.fromisoformat(period["startTime"])
        d = start.date()
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
        value_c = (((feature.get("properties") or {}).get("temperature") or {}).get("value"))
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
    nws = {r["date"]: r for r in get_nws_daily(cfg["lat"], cfg["lon"])}

    event_cache = {}
    rows = []

    for m in markets:
        d = infer_market_date(m, cfg["tz"])
        if d is None or d not in ens.index:
            continue

        kind, lo, hi, bracket = market_condition(m)
        if kind is None:
            continue

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

        side_data = [
            ("YES", to_float(m.get("yes_ask_dollars")), p_yes),
            ("NO", to_float(m.get("no_ask_dollars")), 1 - p_yes),
        ]

        for side, ask, p in side_data:
            # Never synthesize the opposite ask. Use the actual live side price only.
            if ask is None or not (0 < ask < 1):
                continue
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
                "nws_public_url": f"https://forecast.weather.gov/MapClick.php?FcstType=graphical&lat={cfg['lat']}&lon={cfg['lon']}&unit=0&lg=english",
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
.we-station { color:#c5ccd6; font-size:1.05rem; margin:.4rem 0 1rem; }
.we-dash { border-top:1px dashed #5b6571; opacity:.6; margin:1rem 0; }

.we-nwsbox {
    border:1px solid #27df68; border-radius:10px;
    background:linear-gradient(90deg, rgba(11,63,32,.30), rgba(10,30,20,.18));
    padding:.9rem 1rem; margin:.7rem 0 .25rem;
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

@media (max-width:760px) {
    .we-metricgrid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .we-metric { min-height:112px; }
    .we-metric-value { font-size:1.5rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="we-topbar">
  <div class="we-badge">⚡ WEATHEREDGE v2.3.4 • GFS RANGE + CONTRACT ODDS</div>
  <div class="we-actions">
    <div class="we-action">↗ Share</div>
    <div class="we-action">☆</div>
    <div class="we-action">◉</div>
    <div class="we-action">☼</div>
  </div>
</div>
<div class="we-hero">
  <div class="we-logo">🌦️</div>
  <div class="we-title">Weather<span class="edge">Edge</span></div>
</div>
<div class="we-sub">Forecast-aligned Kalshi weather candidates, simplified.</div>
<div class="we-divider"></div>
<div class="we-section-head">
  <div class="we-section-icon">◎</div>
  <div class="we-section-title">Top opportunities</div>
  <div class="we-live">Live</div>
</div>
<div class="we-section-sub">Sorted by model/market gap (high to low) • NWS high = maximum of the latest hourly NWS forecast for that day</div>
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
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="we-cardhead">
              <span class="we-rank">#{rank}</span>
              <span class="we-city">{r['city']}</span>
              <span class="we-date">▣ {r['date_label']}</span>
              <div class="we-bet">Market: <b>{r['side']}</b> on “{r['market_subtitle']}”</div>
              <div class="we-station">⌖ {r['station_hint']}</div>
              <div class="we-dash"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nws_url = r["nws_public_url"]
        st.markdown(
            f"""
            <div class="we-nwsbox">
              <a href="{nws_url}" target="_blank">🌤️ Open official NWS hourly forecast ↗</a>
              <div style="color:#a9b7c5;font-size:.82rem;margin-top:.35rem;">
                This page shows the remaining NWS hourly forecast. WeatherEdge combines that with the observed high so far to project the final daily high.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        st.markdown(
            f"""
            <div class="we-metricgrid">
              <div class="we-metric">
                <div class="we-metric-icon">🌡️</div>
                <div class="we-metric-label">Projected Daily High</div>
                <div class="we-metric-value">{nws_txt}</div>
              </div>
              <div class="we-metric">
                <div class="we-metric-icon">↗</div>
                <div class="we-metric-label">Observed High</div>
                <div class="we-metric-value">{obs_txt}</div>
              </div>
              <div class="we-metric">
                <div class="we-metric-icon">▮▮</div>
                <div class="we-metric-label">GFS Likely Range</div>
                <div class="we-metric-value" style="font-size:1.28rem;">{range_txt}</div>
                <div style="color:#8fa0b3;font-size:.78rem;margin-top:.35rem;">Median {med_txt}</div>
              </div>
              <div class="we-metric">
                <div class="we-metric-icon">🏷️</div>
                <div class="we-metric-label">Kalshi Price</div>
                <div class="we-metric-value">{price_txt}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        remaining = r.get("nws_remaining_high_f")
        remaining_txt = (
            "—"
            if remaining is None or pd.isna(remaining)
            else f"{int(round(float(remaining)))}°F"
        )
        observed_txt = obs_txt
        st.caption(
            f"Projected daily high = max(observed high so far, remaining NWS hourly forecast). "
            f"Observed: {observed_txt} • Remaining NWS peak: {remaining_txt}"
        )

        st.markdown(
            f"""
            <div class="we-probgrid">
              <div class="we-probbox">
                <div class="we-probtitle">GFS Chance of This Side</div>
                <div class="we-probvalue">{fmt_pct(r['model_prob'])}</div>
                <div class="we-agree">Conservative estimate: {fmt_pct(r['conservative_prob'])}</div>
              </div>
              <div class="we-probbox">
                <div class="we-probtitle">Model/Market Gap</div>
                <div class="we-gapvalue">{r['conservative_edge']*100:+.1f} pp</div>
                <div class="we-warning">Compared with the live Kalshi ask</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"GFS 80% likely range: {range_txt} • Median: {med_txt} • "
            f"Model chance for {r['side']} on “{r['market_subtitle']}”: {fmt_pct(r['model_prob'])} • "
            f"Conservative chance: {fmt_pct(r['conservative_prob'])} • Kalshi ask: {r['ask']*100:.0f}¢"
        )

        st.link_button(
            "📊 Open exact market on Kalshi ↗",
            r["kalshi_event_url"],
            use_container_width=True,
        )
        st.markdown(
            f"<div class='we-footerline'>Find {r['market_subtitle']} and choose <b>{r['side']}</b> · ticker "
            f"<span class='we-ticker'>{r['market_ticker']}</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("""
<div class="we-bottom">
  <div class="we-tab active">◎<br>Top Picks</div>
  <div class="we-tab">▤<br>All Markets</div>
  <div class="we-tab">☆<br>Watchlist</div>
  <div class="we-tab">ⓘ<br>How It Works</div>
</div>
""", unsafe_allow_html=True)
