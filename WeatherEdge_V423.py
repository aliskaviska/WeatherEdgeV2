
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


def build_city_rows(city, cfg):
    """
    Fast live-scan path.

    GFS is excluded from this stage because it does not affect ranking.
    Shared weather/observation context is computed once per city/date.
    """
    calibration = build_nws_error_calibration(
        city, cfg.get("station_id"), cfg["tz"]
    )
    markets = get_kalshi_markets(cfg["series"])
    if not markets:
        return []

    series_info = get_series_info(cfg["series"])
    nws = {r["date"]: r for r in get_nws_daily(cfg["lat"], cfg["lon"])}

    markets_by_event = {}
    for market in markets:
        markets_by_event.setdefault(market.get("event_ticker"), []).append(market)
    implied_by_event = {
        event_ticker: market_implied_temperature(group)
        for event_ticker, group in markets_by_event.items()
    }

    event_cache = {}
    date_context_cache = {}
    rows = []

    def get_date_context(d):
        if d in date_context_cache:
            return date_context_cache[d]

        nrow = nws.get(d, {})
        nws_high = nrow.get("nws_high_f")
        if nws_high is None:
            date_context_cache[d] = None
            return None

        peak_context = expected_peak_context(calibration, d, cfg["tz"])
        hours_to_peak = peak_context["hours_to_peak"]
        trajectory_score = trajectory_agreement_score(city, cfg, d, nws_high)

        observed_high = observed_high_time = None
        previous_day_high = previous_day_high_time = None
        previous_3day_avg_high = None
        previous_day_prediction = None

        try:
            observed_high, observed_high_time = observed_daily_high_details(
                cfg.get("station_id"), cfg["tz"], d
            )
            (
                previous_day_high,
                previous_day_high_time,
                previous_3day_avg_high,
            ) = recent_observed_high_summary(
                cfg.get("station_id"), cfg["tz"], d
            )
            previous_day_prediction = previous_day_forecast_summary(
                city, cfg["tz"], d
            )
        except Exception:
            pass

        bucket = _lead_bucket(hours_to_peak)
        ctx = {
            "nrow": nrow,
            "nws_high": nws_high,
            "peak_context": peak_context,
            "hours_to_peak": hours_to_peak,
            "trajectory_score": trajectory_score,
            "observed_high": observed_high,
            "observed_high_time": observed_high_time,
            "previous_day_high": previous_day_high,
            "previous_day_high_time": previous_day_high_time,
            "previous_3day_avg_high": previous_3day_avg_high,
            "previous_day_prediction": previous_day_prediction,
            "sigma_source": (
                "historical"
                if calibration.get(bucket, {}).get("n", 0) >= 8
                else "fallback"
            ),
            "sigma_samples": calibration.get(bucket, {}).get("n", 0),
        }
        date_context_cache[d] = ctx
        return ctx

    for m in markets:
        d = infer_market_date(m, cfg["tz"])
        if d is None or d not in nws:
            continue

        kind, lo, hi, bracket = market_condition(m)
        if kind is None:
            continue

        ctx = get_date_context(d)
        if not ctx:
            continue

        nrow = ctx["nrow"]
        nws_high = ctx["nws_high"]
        hours_to_peak = ctx["hours_to_peak"]
        observed_high = ctx["observed_high"]

        hours_left = hours_to_deadline(m, cfg["tz"])
        p_yes = nws_yes_probability(
            nws_high,
            kind,
            lo,
            hi,
            hours_left=hours_left,
            observed_high=observed_high,
            calibration=calibration,
            hours_to_peak=hours_to_peak,
        )
        if p_yes is None:
            continue

        event_ticker = m.get("event_ticker")
        if event_ticker not in event_cache:
            try:
                event_cache[event_ticker] = get_event(event_ticker)
            except Exception:
                event_cache[event_ticker] = {}
        event_info = event_cache[event_ticker]

        title = (
            event_info.get("title")
            or m.get("title")
            or series_info.get("title")
            or f"{city} high temperature"
        )
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or bracket
        settlement = source_names(series_info, event_info)
        contract_url = series_info.get("contract_url")
        implied_temp = implied_by_event.get(event_ticker)
        temp_gap = (
            float(nws_high) - implied_temp
            if implied_temp is not None
            else None
        )
        nws_support_yes = point_forecast_supports_yes(
            nws_high, kind, lo, hi
        )

        for side, ask, p_nws in [
            ("YES", to_float(m.get("yes_ask_dollars")), p_yes),
            ("NO", to_float(m.get("no_ask_dollars")), 1 - p_yes),
        ]:
            if ask is None or not (0 < ask < 1):
                continue

            edge = p_nws - ask
            score = opportunity_score(edge, temp_gap, hours_left)
            quality_score = bet_quality_score(
                edge=edge,
                p_nws=p_nws,
                hours_left=hours_left,
                temp_gap=temp_gap,
                trajectory_score=ctx["trajectory_score"],
                sigma_source=ctx["sigma_source"],
                sigma_samples=ctx["sigma_samples"],
            )

            nws_support = (
                nws_support_yes
                if side == "YES"
                else (
                    not nws_support_yes
                    if nws_support_yes is not None
                    else None
                )
            )

            qualifies = (
                p_nws >= 0.55
                and edge >= 0.05
                and nws_support is True
            )
            suspicious = edge >= 0.30
            previous_day_prediction = ctx["previous_day_prediction"]
            peak_context = ctx["peak_context"]

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
                "model_prob": p_nws,
                "conservative_prob": p_nws,
                "edge": edge,
                "conservative_edge": edge,
                "expected_roi": edge / ask,
                "n_members": None,
                "nws_high_f": nws_high,
                "observed_high_f": observed_high,
                "observed_high_time_local": ctx["observed_high_time"],
                "previous_day_high_f": ctx["previous_day_high"],
                "previous_day_high_time_local": ctx["previous_day_high_time"],
                "previous_3day_avg_high_f": ctx["previous_3day_avg_high"],
                "previous_day_prediction_avg_f": (
                    previous_day_prediction.get("average_f")
                    if previous_day_prediction else None
                ),
                "previous_day_prediction_low_f": (
                    previous_day_prediction.get("low_f")
                    if previous_day_prediction else None
                ),
                "previous_day_prediction_high_f": (
                    previous_day_prediction.get("high_f")
                    if previous_day_prediction else None
                ),
                "previous_day_prediction_n": (
                    previous_day_prediction.get("n_snapshots")
                    if previous_day_prediction else 0
                ),
                "observed_data_url": nws_climate_url(cfg),
                "nws_forecast": nrow.get("nws_detail"),
                "nws_forecast_url": nrow.get("nws_forecast_url"),
                "nws_sigma_f": nws_sigma_f(hours_to_peak, calibration),
                "nws_bias_f": nws_bias_f(hours_to_peak, calibration),
                "nws_sigma_source": ctx["sigma_source"],
                "nws_sigma_samples": ctx["sigma_samples"],
                "hours_to_expected_peak": hours_to_peak,
                "expected_peak_local": peak_context["expected_peak_dt"],
                "peak_history_days": peak_context["n_peak_days"],
                "hours_to_settlement": hours_left,
                "kalshi_implied_temp_f": implied_temp,
                "temperature_mismatch_f": temp_gap,
                "trajectory_agreement_score": ctx["trajectory_score"],
                "bet_quality_score": quality_score,
                "bet_quality_label": bet_quality_label(quality_score),
                "opportunity_score": score,
                "ensemble_median_f": None,
                "ensemble_low_f": None,
                "ensemble_high_f": None,
                "ensemble_daily_highs_f": [],
                "nws_support": nws_support,
                "median_support": None,
                "forecasts_agree": nws_support is True,
                "qualifies": qualifies,
                "agreement": (
                    "✅ NWS supports this side"
                    if nws_support is True
                    else "❌ NWS opposes this side"
                ),
                "settlement_source": settlement,
                "contract_url": contract_url,
                "kalshi_event_url": kalshi_event_url(
                    cfg["series"], event_ticker
                ),
                "volume": to_float(m.get("volume_fp")) or 0,
                "open_interest": to_float(m.get("open_interest_fp")) or 0,
                "suspicious": suspicious,
            })

    return rows



@st.cache_data(ttl=300, show_spinner=False)
def scan_live_market_universe():
    """Refresh supported cities concurrently and cache the result for 5 minutes."""
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

    Historical Supabase snapshot caches are intentionally left alone because
    they change less often and are not the cause of routine navigation lag.
    """
    scan_live_market_universe.clear()

    # Fast-moving public inputs.
    for func in (
        get_kalshi_markets,
        get_event,
        get_nws_daily,
        get_station_observations,
        get_observed_high_so_far,
        get_gfs_ensemble_daily_highs,
        get_lazy_gfs_for_contract,
    ):
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
    Summarize stored NWS predictions for the PREVIOUS day's high, using forecasts
    issued during the day before that.

    Example for an Aug 12 market page:
      - previous observed day = Aug 11
      - source forecast day = Aug 10
      - collect stored snapshots issued Aug 10 whose valid times fall on Aug 11
      - return the mean projected high, and min/max if the projection changed

    Returns dict with average_f, low_f, high_f, n_snapshots, or None if unavailable.
    """
    rows, err = get_snapshot_rows(city, None)
    snap, norm_err = normalize_snapshot_rows(rows)
    if err or norm_err or snap.empty:
        return None

    tz = ZoneInfo(tz_name)
    previous_day = target_date - timedelta(days=1)
    source_day = target_date - timedelta(days=2)

    work = snap.copy()
    work["valid_local"] = work["valid_at"].dt.tz_convert(tz_name)
    work["snapshot_local"] = work["snapshot_at"].dt.tz_convert(tz_name)
    work["valid_date"] = work["valid_local"].dt.date
    work["snapshot_date"] = work["snapshot_local"].dt.date

    work = work[
        (work["valid_date"] == previous_day)
        & (work["snapshot_date"] == source_day)
    ].copy()
    if work.empty:
        return None

    projected = (
        work.groupby("snapshot_key", as_index=False)
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
        "n_snapshots": int(len(values)),
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

    # Historical reference: forecasts made the day before for the previous day's high.
    prev_pred_avg = clean_number(row.get("previous_day_prediction_avg_f"))
    prev_pred_low = clean_number(row.get("previous_day_prediction_low_f"))
    prev_pred_high = clean_number(row.get("previous_day_prediction_high_f"))
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
            "label": f"Avg prior-day prediction · {prev_pred_n} snapshots",
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
        def marker_color(label):
            if label.startswith("Raw NWS high"):
                return "#FFFFFF"
            if label.startswith("Calibrated NWS center"):
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
            # Integer Kalshi buckets use half-degree settlement boundaries.
            band_lo, band_hi = bet_low - 0.5, bet_high + 0.5
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
                    opacity=0.24,
                    stroke=top_color,
                    strokeWidth=3,
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
                    angle=270,
                    fontSize=13,
                    fontWeight="bold",
                    color="#FFFFFF",
                    stroke=top_color,
                    strokeWidth=3,
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
        items.append((obs_label, "#FF2D8D", "dash"))

    if present(row.get("kalshi_implied_temp_f")):
        items.append(("Kalshi implied temp", "#FFD400", "dash"))

    prev_pred_avg = row.get("previous_day_prediction_avg_f")
    prev_pred_low = row.get("previous_day_prediction_low_f")
    prev_pred_high = row.get("previous_day_prediction_high_f")
    prev_pred_n = int(row.get("previous_day_prediction_n") or 0)

    if present(prev_pred_avg):
        if present(prev_pred_low) and present(prev_pred_high) and abs(float(prev_pred_high) - float(prev_pred_low)) >= 0.05:
            items.append((
                f"Prior-day forecasts for previous high · {prev_pred_n} snapshots",
                "#36C2FF",
                "history_band",
            ))
        else:
            items.append((
                f"Prior-day prediction for previous high · {prev_pred_n} snapshot"
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
        items.append((prev_label, "#00E676", "dash"))

    if present(row.get("previous_3day_avg_high_f")):
        items.append(("Average high, previous 3 days", "#FF8A00", "dash"))

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
    st.caption("Live scan is cached for 5 minutes. Refreshes run cities in parallel, and GFS loads only when you open a probability chart.")

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
        chart_row = enrich_row_with_lazy_gfs(r)
        range_chart = forecast_range_summary_chart(
            chart_row,
            show_bet_overlay=(status_kind == "best"),
        )
        if range_chart is not None:
            st.markdown("<div class='section-kicker'>DAILY HIGH PROBABILITY</div>", unsafe_allow_html=True)
            st.altair_chart(range_chart, use_container_width=True)
            render_probability_chart_legend(chart_row)
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
        "Browse every configured Kalshi temperature city and open any active contract."
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

        city_df = city_df.sort_values(["date", "market_ticker", "side"])
        available_dates = list(city_df["date"].drop_duplicates())
        date_labels = [pd.Timestamp(d).strftime("%a %b %-d") for d in available_dates]

        # If a direct Best Bets/My Bets jump supplied a contract, choose its date.
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

        # Select the contract BEFORE rendering figures. The selected contract now
        # drives Kalshi implied temperature, Top Bet status, and chart shading.
        st.markdown("<div class='section-kicker'>AVAILABLE BETS</div>", unsafe_allow_html=True)
        contract_rows = [
            r for _, r in date_df.sort_values(
                ["market_subtitle", "side", "ask"],
                ascending=[True, True, True],
            ).iterrows()
        ]
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

        desired = st.session_state.get("explorer_contract")
        default_idx = contract_keys.index(desired) if desired in contract_keys else 0

        selected_contract_label = st.selectbox(
            "Open contract",
            contract_labels,
            index=default_idx,
            key=f"explorer_contract_select_{city}_{selected_date}",
        )
        contract_idx = contract_labels.index(selected_contract_label)
        selected_row = contract_rows[contract_idx]
        selected_key = contract_keys[contract_idx]
        st.session_state.explorer_contract = selected_key

        # The selected contract is the single source of truth for this bet page.
        representative = selected_row
        selected_status, selected_status_kind = contract_status(representative)

        st.markdown(f"<div class='card-title'>{city} · {representative['market_subtitle']}</div>", unsafe_allow_html=True)

        if selected_status_kind == "best":
            st.success(f"{selected_status} · {representative['side']}")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Latest NWS high",
            "—" if pd.isna(representative["nws_high_f"]) else f"{representative['nws_high_f']:.0f}°F",
        )
        obs = representative.get("observed_high_f")
        obs_time = representative.get("observed_high_time_local")
        obs_label = "Observed so far"
        if obs_time is not None and not pd.isna(obs_time):
            try:
                obs_label += f" ({pd.Timestamp(obs_time).strftime('%-I:%M %p')})"
            except Exception:
                pass
        m2.metric(
            obs_label,
            "—" if obs is None or pd.isna(obs) else f"{obs:.0f}°F",
        )
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
                st.link_button(
                    "Observed temperatures ↗",
                    observed_url,
                    use_container_width=True,
                )

        # The probability figure now receives the exact selected contract.
        chart_row = enrich_row_with_lazy_gfs(representative)
        range_chart = forecast_range_summary_chart(
            chart_row,
            show_bet_overlay=(selected_status_kind == "best"),
        )
        if range_chart is not None:
            st.markdown(
                "<div class='section-kicker'>DAILY HIGH PROBABILITY</div>",
                unsafe_allow_html=True,
            )
            st.altair_chart(range_chart, use_container_width=True)
            render_probability_chart_legend(chart_row)

        render_bet_forecast(city, selected_date)

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
