#!/usr/bin/env python3
"""
GeoMagVolcano Monitor v3.1 API-stable
APS 1540 - Isola di Vulcano (38.40 N, 14.97 E)

Avvio:
    streamlit run app_geomag_volcano_v3.py

Dipendenze consigliate:
    pip install streamlit pandas numpy requests plotly scipy

Note scientifiche sintetiche
---------------------------
Il segnale magnetico locale osservato e' trattato come:
    B_obs = B_main + B_external + B_local + noise
Dove B_main e' approssimato con una baseline IGRF locale, B_external e' controllato
tramite indici geomagnetici/vento solare, e B_local e' la componente potenzialmente
vulcanomagnetica. Lo stato vulcanico non viene stimato da un singolo parametro, ma
tramite un indice multiparametrico con penalizzazione della qualita' quando Kp, Dst,
Bz o il vento solare indicano condizioni geomagneticamente disturbate.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    from scipy.signal import butter, filtfilt, periodogram, spectrogram
except ImportError:  # pragma: no cover - handled in UI
    butter = None
    filtfilt = None
    periodogram = None
    spectrogram = None


# =============================================================================
# Configuration
# =============================================================================

APP_TITLE = "GeoMagVolcano Monitor v4.0 — Analisi notturna & GFZ overlay"
LAT_DEFAULT = 38.40
LON_DEFAULT = 14.97
ALT_DEFAULT = 120.0
IGRF_F_DEFAULT = 45_800.0
IGRF_D_DEFAULT = -2.3
IGRF_I_DEFAULT = -55.8
REQUEST_TIMEOUT = 20
USER_AGENT = "GeoMagVolcanoMonitor/3.0 scientific-volcano-monitoring"

NOAA = {
    # Endpoint attivi/verificati. Gli alias app4 sono mantenuti piu' sotto come fallback.
    "kp1m": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
    "kp3h": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "kpfcst": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json",
    "mag_1d": "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json",
    "plasma_1d": "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json",
    "xray_7d": "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
    "dst_1h": "https://services.swpc.noaa.gov/json/geospace/geospace_dst_1_hour.json",
    "f107": "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
}

# Endpoint recuperati dalla versione funzionante app4.py e mantenuti come codici API/fallback.
# Alcuni NOAA legacy possono restituire 404 oggi, ma sono lasciati qui per tracciabilita'.
NOAA_APP4_LEGACY = {
    "kp1m": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json",
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json",
    "xray": "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
    "f107": "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
    "kp3h": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "dst_legacy": "https://services.swpc.noaa.gov/products/geospace/dst-1-7-day.json",
    "kpfcst_legacy": "https://services.swpc.noaa.gov/json/planetary_k_index_forecast.json",
}

# SWPC retired /products/solar-wind/*.json on 2026-06-30, when DSCOVR ingest was
# stopped and SOLAR-1 became the primary RTSW instrument. The last 24 h of in situ
# data now live here as static JSON files (records, all spacecraft in one file).
NOAA_RTSW = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "wind": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    "ephemerides": "https://services.swpc.noaa.gov/json/rtsw/rtsw_ephemerides_1h.json",
}

NOAA_FALLBACKS = {
    "dst": [NOAA["dst_1h"], NOAA_APP4_LEGACY["dst_legacy"]],
    "kpfcst": [NOAA["kpfcst"], NOAA_APP4_LEGACY["kpfcst_legacy"]],
    "kp1m": [NOAA["kp1m"], NOAA_APP4_LEGACY["kp1m"]],
    "kp3h": [NOAA["kp3h"], NOAA_APP4_LEGACY["kp3h"]],
    # RTSW first; the legacy solar-wind URLs are kept only for traceability
    # (they return 404 since 2026-06-30).
    "mag": [NOAA_RTSW["mag"], NOAA["mag_1d"]],
    "plasma": [NOAA_RTSW["wind"], NOAA["plasma_1d"]],
    "xray": [NOAA["xray_7d"], NOAA_APP4_LEGACY["xray"]],
    "f107": [NOAA["f107"], NOAA_APP4_LEGACY["f107"]],
}

FDSN_ENDPOINTS = {
    "USGS global": "https://earthquake.usgs.gov/fdsnws/event/1/query",
    "INGV FDSN": "https://webservices.ingv.it/fdsnws/event/1/query",
    "EMSC SeismicPortal": "https://www.seismicportal.eu/fdsnws/event/1/query",
}

# API access copied from the working app4.py implementation.
NASA_DONKI_URL = "https://api.nasa.gov/DONKI"
NASA_DEMO_KEY = "nrssiOcAgmRoesKhzo33m4l76Tte27qefXfPrXOh"
OPEN_METEO_APP4_HOURLY = (
    "temperature_2m,relativehumidity_2m,surface_pressure,"
    "windspeed_10m,soil_temperature_0cm"
)

API_DATABASES = {
    "NOAA_SWPC": {
        "descrizione": "Space-weather real-time: Kp, IMF, vento solare, raggi X, Dst, F10.7",
        **NOAA,
        **{f"legacy_{k}": v for k, v in NOAA_APP4_LEGACY.items()},
    },
    "GFZ_POTSDAM": {
        "descrizione": "Indici geomagnetici storici/nowcast: Kp, ap, Ap, Cp, Hp30, Hp60, SN, Fobs, Fadj",
        "json_api": "https://kp.gfz-potsdam.de/app/json/",
        "parametri": "start, end, index, status",
    },
    "NASA_DONKI": {
        "descrizione": "Eventi solari e interplanetari: FLR, GST, CME, SEP, IPS, HSS, WSAEnlilSimulations",
        "base_url": NASA_DONKI_URL,
        "parametri": "startDate, endDate, api_key",
    },
    "INGV_FDSN": {
        "descrizione": "Catalogo terremoti INGV in formato text/QuakeML",
        "event_query": "https://webservices.ingv.it/fdsnws/event/1/query",
        "parametri": "starttime, endtime, minmag, latitude, longitude, maxradius, format=text",
    },
    "USGS_FDSN": {
        "descrizione": "Catalogo terremoti globale USGS, fallback GeoJSON",
        "event_query": "https://earthquake.usgs.gov/fdsnws/event/1/query",
        "parametri": "format=geojson, starttime, endtime, latitude, longitude, maxradius, minmagnitude",
    },
    "EMSC_FDSN": {
        "descrizione": "Catalogo terremoti EMSC/SeismicPortal",
        "event_query": "https://www.seismicportal.eu/fdsnws/event/1/query",
        "parametri": "format=text/geojson, starttime, endtime, latitude, longitude, maxradius",
    },
    "OPEN_METEO": {
        "descrizione": "Meteo orario no-key: temperatura, umidita', pressione, vento, temperatura suolo",
        "forecast": "https://api.open-meteo.com/v1/forecast",
        "hourly": OPEN_METEO_APP4_HOURLY,
    },
}


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    margin=dict(l=50, r=25, t=55, b=40),
    hovermode="x unified",
    font=dict(family="Inter, Arial, sans-serif", size=12),
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.mbox{background:#1a2235;border:1px solid #2d3748;border-radius:12px;padding:14px 18px;margin-bottom:10px}
.mlbl{font-size:11px;color:#93a4b7;text-transform:uppercase;letter-spacing:.07em;font-weight:700}
.mval{font-size:25px;font-weight:800;font-family:'JetBrains Mono',monospace}
.info{background:rgba(56,189,248,.08);border-left:4px solid #38bdf8;border-radius:0 10px 10px 0;padding:12px 16px;font-size:13px}
.green{background:rgba(74,222,128,.10);border:1px solid rgba(74,222,128,.35);border-radius:10px;padding:12px 16px}
.yellow{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.35);border-radius:10px;padding:12px 16px}
.orange{background:rgba(251,146,60,.10);border:1px solid rgba(251,146,60,.35);border-radius:10px;padding:12px 16px}
.red{background:rgba(248,113,113,.10);border:1px solid rgba(248,113,113,.35);border-radius:10px;padding:12px 16px}
.small{font-size:12px;color:#93a4b7}

/* ── Seismic recent-event flash cards ──────────────────────────────────────── */
@keyframes seis-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(248,113,113,0.70); border-color: rgba(248,113,113,0.90); }
  50%  { box-shadow: 0 0 18px 8px rgba(248,113,113,0.15); border-color: rgba(248,113,113,0.40); }
  100% { box-shadow: 0 0 0 0 rgba(248,113,113,0.70); border-color: rgba(248,113,113,0.90); }
}
@keyframes seis-pulse-orange {
  0%   { box-shadow: 0 0 0 0 rgba(251,146,60,0.70); border-color: rgba(251,146,60,0.90); }
  50%  { box-shadow: 0 0 18px 8px rgba(251,146,60,0.12); border-color: rgba(251,146,60,0.40); }
  100% { box-shadow: 0 0 0 0 rgba(251,146,60,0.70); border-color: rgba(251,146,60,0.90); }
}
@keyframes dot-blink {
  0%,100% { opacity:1; }
  50%      { opacity:0.15; }
}
.seis-card {
  background:#1a2235;border:1.5px solid rgba(248,113,113,0.80);border-radius:12px;
  padding:10px 16px;margin-bottom:8px;
  animation: seis-pulse 2.4s ease-in-out infinite;
  position:relative;
}
.seis-card.recent2 { animation: seis-pulse-orange 2.8s ease-in-out infinite; border-color:rgba(251,146,60,0.80); }
.seis-card.recent3 { animation: none; border-color:rgba(251,191,36,0.55); background:#1a2030; }
.seis-card.recent4, .seis-card.recent5 { animation: none; border-color:rgba(100,116,139,0.45); background:#161d2d; }
.seis-badge {
  display:inline-block;border-radius:6px;font-size:11px;font-weight:700;
  padding:2px 8px;margin-right:8px;font-family:'JetBrains Mono',monospace;
  letter-spacing:.04em;
}
.seis-badge.m-minor { background:rgba(74,222,128,.20); color:#4ade80; }
.seis-badge.m-light { background:rgba(251,191,36,.20); color:#fbbf24; }
.seis-badge.m-moderate { background:rgba(251,146,60,.25); color:#fb923c; }
.seis-badge.m-strong  { background:rgba(248,113,113,.30); color:#f87171; }
.seis-time { font-size:12px; color:#93a4b7; font-family:'JetBrains Mono',monospace; }
.seis-place { font-size:13px; color:#e2e8f0; margin-top:2px; }
.seis-meta  { font-size:11px; color:#64748b; margin-top:2px; }
.live-dot {
  display:inline-block;width:9px;height:9px;border-radius:50%;
  background:#f87171;margin-right:6px;vertical-align:middle;
  animation: dot-blink 1.4s ease-in-out infinite;
}
.live-dot.orange { background:#fb923c; }
.live-dot.yellow { background:#fbbf24; }
.live-dot.gray   { background:#64748b; animation:none; }

/* ── Swarm alert banner ──────────────────────────────────────────────────── */
@keyframes swarm-flash {
  0%,100% { background:rgba(248,113,113,0.12); }
  50%     { background:rgba(248,113,113,0.22); }
}
.swarm-alert {
  border:1.5px solid rgba(248,113,113,0.60);border-radius:10px;padding:14px 18px;
  animation: swarm-flash 1.8s ease-in-out infinite;
  font-weight:600; color:#f87171;
}

/* ── Stat mini-card row ───────────────────────────────────────────────────── */
.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }
.stat-card {
  background:#131c2e;border:1px solid #2d3748;border-radius:10px;
  padding:10px 16px;min-width:110px;flex:1;
}
.stat-card .sv { font-size:22px;font-weight:800;font-family:'JetBrains Mono',monospace; }
.stat-card .sl { font-size:10px;color:#93a4b7;text-transform:uppercase;letter-spacing:.06em;font-weight:700; }
</style>
"""


@dataclass
class VolcanoState:
    level: str
    css_class: str
    score: float
    confidence: float
    explanation: str
    recommended_action: str


# =============================================================================
# Robust IO and utility functions
# =============================================================================


def http_get_json(url: str, params: Optional[dict] = None) -> Optional[object]:
    """Return JSON payload with conservative timeout and explicit user agent."""
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.warning(f"API non raggiungibile: {url} ({exc})")
        return None
    except ValueError as exc:
        st.warning(f"Risposta JSON non valida da {url}: {exc}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_json_cached(url: str, params: Optional[dict] = None) -> Optional[object]:
    return http_get_json(url, params=params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_matrix_cached(url: str) -> pd.DataFrame:
    payload = fetch_json_cached(url)
    if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[0], list):
        return pd.DataFrame(payload[1:], columns=payload[0])
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_json_fallback_cached(urls: Tuple[str, ...], params: Optional[dict] = None) -> Optional[object]:
    """Try multiple API URLs and return the first valid JSON payload.

    This is useful for NOAA/SWPC products whose operational paths occasionally
    move. Legacy app4.py URLs are retained as fallback but active endpoints are
    tried first.
    """
    last_error = None
    for url in urls:
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code == 204:
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        st.warning(f"Nessun endpoint disponibile tra: {', '.join(urls)} ({last_error})")
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_matrix_fallback_cached(urls: Tuple[str, ...]) -> pd.DataFrame:
    payload = fetch_json_fallback_cached(urls)
    if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[0], list):
        return pd.DataFrame(payload[1:], columns=payload[0])
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return pd.DataFrame(payload)
    return pd.DataFrame()


def detect_separator(raw: str) -> str:
    counts = {",": raw.count(","), ";": raw.count(";"), "\t": raw.count("\t")}
    return max(counts, key=counts.get)


def read_uploaded_table(uploaded_file, decimal: str = ".") -> pd.DataFrame:
    """Read CSV/TXT with automatic delimiter detection and safe decoding."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        raw = uploaded_file.read().decode("utf-8", errors="replace")
        sep = detect_separator(raw)
        df = pd.read_csv(io.StringIO(raw), sep=sep, decimal=decimal, engine="python")
        df.columns = [str(col).strip().replace('"', "") for col in df.columns]
        return df
    except Exception as exc:  # noqa: BLE001 - Streamlit UI should not crash
        st.error(f"Errore lettura file: {exc}")
        return pd.DataFrame()


def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "datetime",
        "timestamp",
        "time",
        "date",
        "data",
        "utc",
        "origin_time",
    ]
    lower_map = {str(col).lower().strip(): col for col in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    for col in df.columns:
        text = str(col).lower()
        if "time" in text or "date" in text or "data" in text:
            return col
    return None


def prepare_time_index(df: pd.DataFrame, dayfirst: bool = True) -> pd.DataFrame:
    """Create a DatetimeIndex from common timestamp formats."""
    if df.empty:
        return df
    result = df.copy()
    if all(col in result.columns for col in ["Year", "Month", "Day", "Hour", "Min", "Sec"]):
        parts = result[["Year", "Month", "Day", "Hour", "Min", "Sec"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        parts.columns = ["year", "month", "day", "hour", "minute", "second"]
        result["Datetime"] = pd.to_datetime(parts, errors="coerce")
    else:
        time_col = find_datetime_column(result)
        if time_col is None:
            raise ValueError("Colonna temporale non trovata nel file.")
        result["Datetime"] = pd.to_datetime(result[time_col], dayfirst=dayfirst, errors="coerce")
    result = result.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    result = result[~result.index.duplicated(keep="first")]
    return result


def to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def unit_to_nt(series: pd.Series, unit: str) -> pd.Series:
    if unit == "mG":
        return series * 100.0
    if unit in {"G", "Gauss"}:
        return series * 100_000.0
    return series


def robust_zscore(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """Median/MAD z-score, robust against spikes and non-Gaussian tails."""
    s = pd.to_numeric(series, errors="coerce")
    if window and window >= 5:
        med = s.rolling(window, min_periods=max(3, window // 4), center=True).median()
        mad = (s - med).abs().rolling(window, min_periods=max(3, window // 4), center=True).median()
    else:
        med = pd.Series(s.median(), index=s.index)
        mad = pd.Series((s - s.median()).abs().median(), index=s.index)
    sigma = 1.4826 * mad.replace(0, np.nan)
    return ((s - med) / sigma).replace([np.inf, -np.inf], np.nan)


def hampel_filter(series: pd.Series, window: int = 11, n_sigmas: float = 4.0) -> pd.Series:
    """Replace isolated spikes using a Hampel filter."""
    if window < 3:
        return series
    s = pd.to_numeric(series, errors="coerce").copy()
    rolling_median = s.rolling(window, center=True, min_periods=3).median()
    diff = (s - rolling_median).abs()
    mad = diff.rolling(window, center=True, min_periods=3).median()
    threshold = n_sigmas * 1.4826 * mad
    outliers = diff > threshold
    s.loc[outliers] = np.nan
    return s.interpolate(limit_direction="both")


def rolling_anomaly(series: pd.Series, window_samples: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    baseline = s.rolling(window_samples, center=True, min_periods=max(3, window_samples // 5)).median()
    return s - baseline.interpolate(limit_direction="both").fillna(s.median())


def infer_sampling_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 60.0
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    median_dt = diffs[(diffs > 0) & np.isfinite(diffs)].median()
    if pd.isna(median_dt) or median_dt <= 0:
        return 60.0
    return float(median_dt)


def butterworth_filter(
    series: pd.Series,
    lowcut_hours: Optional[float],
    highcut_hours: Optional[float],
    order: int = 4,
) -> pd.Series:
    """Zero-phase Butterworth filter using periods expressed in hours.

    lowcut_hours removes variations slower than this period when used as high-pass.
    highcut_hours keeps variations slower than this period when used as low-pass.
    For a band-pass, use both and ensure lowcut_hours > highcut_hours.
    """
    if butter is None or filtfilt is None:
        st.warning("scipy non installato: filtro Butterworth disattivato.")
        return series

    s = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both")
    if len(s.dropna()) < 20:
        return series

    dt = infer_sampling_seconds(s.index)
    fs = 1.0 / dt
    nyquist = 0.5 * fs

    try:
        if lowcut_hours and highcut_hours:
            low_freq = 1.0 / (lowcut_hours * 3600.0)
            high_freq = 1.0 / (highcut_hours * 3600.0)
            band = sorted([low_freq / nyquist, high_freq / nyquist])
            band = [max(1e-6, min(0.999, val)) for val in band]
            if band[0] >= band[1]:
                return series
            b, a = butter(order, band, btype="band")
        elif lowcut_hours:
            cutoff = 1.0 / (lowcut_hours * 3600.0) / nyquist
            cutoff = max(1e-6, min(0.999, cutoff))
            b, a = butter(order, cutoff, btype="high")
        elif highcut_hours:
            cutoff = 1.0 / (highcut_hours * 3600.0) / nyquist
            cutoff = max(1e-6, min(0.999, cutoff))
            b, a = butter(order, cutoff, btype="low")
        else:
            return series
        filtered = filtfilt(b, a, s.to_numpy(dtype=float))
        return pd.Series(filtered, index=s.index, name=series.name)
    except ValueError as exc:
        st.warning(f"Filtro non applicabile ai dati correnti: {exc}")
        return series


def estimate_sq_from_quiet_hours(series: pd.Series, kp: Optional[pd.Series], kp_threshold: float) -> pd.Series:
    """Estimate solar quiet daily variation from hours with low Kp."""
    s = pd.to_numeric(series, errors="coerce")
    base = s.copy()
    if kp is not None and not kp.empty:
        try:
            kp_hourly = kp.resample("1h").mean()
            quiet_hours = kp_hourly[kp_hourly < kp_threshold].index
            mask = s.index.floor("1h").isin(quiet_hours)
            if mask.sum() > 24:
                base = s.loc[mask]
        except Exception:
            base = s
    if base.dropna().empty:
        return pd.Series(0.0, index=s.index)
    hourly_profile = base.groupby(base.index.hour).median()
    sq_values = [hourly_profile.get(ts.hour, base.median()) for ts in s.index]
    return pd.Series(sq_values, index=s.index, name="Sq_est")


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0
    phi1 = math.radians(lat1)
    lam1 = math.radians(lon1)
    phi2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lam2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return pd.Series(2.0 * radius * np.arcsin(np.sqrt(a)), index=lat2.index)


def seismic_energy_joule(magnitude: pd.Series) -> pd.Series:
    """Approximate radiated seismic energy: log10(E[J]) = 1.5 M + 4.8."""
    mag = pd.to_numeric(magnitude, errors="coerce")
    return 10.0 ** (1.5 * mag + 4.8)


def normalize_01(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def latest_numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if df.empty or column not in df.columns:
        return default
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.iloc[-1])


def kp_color(kp: float) -> str:
    if kp < 4:
        return "#4ade80"
    if kp < 5:
        return "#fbbf24"
    if kp < 7:
        return "#fb923c"
    return "#f87171"


# =============================================================================
# Nighttime filtering and volcanic anomaly discrimination
# =============================================================================


def filter_nighttime(series: pd.Series, night_start: int = 21, night_end: int = 6) -> pd.Series:
    """Return only samples falling within nighttime hours (UTC).

    Nighttime data are far less contaminated by ionospheric Sq variation and
    external field contributions, making them more suitable for identifying
    persistent volcanic crustal anomalies.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    if night_start > night_end:  # window wraps midnight (e.g. 21->06)
        mask = (series.index.hour >= night_start) | (series.index.hour < night_end)
    else:
        mask = (series.index.hour >= night_start) & (series.index.hour < night_end)
    return series[mask]


def filter_nighttime_df(df: pd.DataFrame, night_start: int = 21, night_end: int = 6) -> pd.DataFrame:
    """Return only rows falling within nighttime hours (UTC)."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if night_start > night_end:
        mask = (df.index.hour >= night_start) | (df.index.hour < night_end)
    else:
        mask = (df.index.hour >= night_start) & (df.index.hour < night_end)
    return df[mask]


def night_daily_stats(series: pd.Series, night_start: int = 21, night_end: int = 6) -> pd.DataFrame:
    """Daily statistics (median, MAD, max|value|, n_samples) for nighttime data only.

    Used to detect persistent night-time anomalies that cannot be explained by
    ionospheric variations (Sq, storm-time ring current) and are therefore more
    likely to be of volcanic/crustal origin.
    """
    night_s = filter_nighttime(series, night_start, night_end)
    if night_s.empty:
        return pd.DataFrame(columns=["night_median", "night_mad", "night_abs_max", "night_n"])
    df = night_s.to_frame(name="_v")
    stats = df.resample("1D").agg(
        night_median=("_v", "median"),
        night_mad=("_v", lambda x: float((x - x.median()).abs().median())),
        night_abs_max=("_v", lambda x: float(x.abs().max()) if not x.empty else 0.0),
        night_n=("_v", "count"),
    )
    return stats


def compute_crosscorr(x: pd.Series, y: pd.Series, max_lag_hours: int = 48, resample_minutes: int = 60) -> pd.DataFrame:
    """Normalised cross-correlation of two irregular time series at multiple lags.

    Both series are resampled to a common regular grid before computing CCF.
    Returns a DataFrame with columns lag_h and correlation.
    """
    freq = f"{max(1, resample_minutes)}min"
    x_r = pd.to_numeric(x, errors="coerce").resample(freq).mean().interpolate(limit_direction="both")
    y_r = pd.to_numeric(y, errors="coerce").resample(freq).mean().interpolate(limit_direction="both")
    common = pd.concat([x_r, y_r], axis=1, join="inner").dropna()
    if len(common) < 10:
        return pd.DataFrame(columns=["lag_h", "correlation"])
    xa = common.iloc[:, 0].to_numpy(dtype=float)
    ya = common.iloc[:, 1].to_numpy(dtype=float)
    xa = (xa - xa.mean()) / max(xa.std(), 1e-12)
    ya = (ya - ya.mean()) / max(ya.std(), 1e-12)
    lag_samples = int(max_lag_hours * 60 / max(1, resample_minutes))
    lags, ccf = [], []
    n = len(xa)
    for lag in range(-lag_samples, lag_samples + 1):
        if lag < 0:
            xi, yi = xa[:n + lag], ya[-lag:]
        elif lag > 0:
            xi, yi = xa[lag:], ya[:n - lag]
        else:
            xi, yi = xa, ya
        length = min(len(xi), len(yi))
        if length < 5:
            ccf.append(np.nan)
        else:
            ccf.append(float(np.corrcoef(xi[:length], yi[:length])[0, 1]))
        lags.append(lag * resample_minutes / 60.0)
    return pd.DataFrame({"lag_h": lags, "correlation": ccf})


def night_persistence_score(night_zscore: pd.Series, threshold: float = 2.0, min_days: int = 2) -> pd.DataFrame:
    """Daily indicator of persistent nighttime z-score anomalies.

    A volcanic signal is expected to persist across multiple nights (unlike
    ionospheric disturbances which are strongly correlated with Kp and last
    hours, not days). A day is flagged if the median |z| of its nighttime
    samples exceeds `threshold`.

    Returns a DataFrame with columns: median_z, flagged (bool), streak (consecutive flagged days).
    """
    if night_zscore.empty:
        return pd.DataFrame(columns=["median_z", "flagged", "streak"])
    daily_z = night_zscore.abs().resample("1D").median()
    flagged = daily_z >= threshold
    streak = flagged.astype(int)
    current = 0
    streak_vals = []
    for f in flagged:
        current = current + 1 if f else 0
        streak_vals.append(current)
    return pd.DataFrame({"median_z": daily_z, "flagged": flagged, "streak": streak_vals}, index=daily_z.index)


# =============================================================================
# Enhanced plot helpers
# =============================================================================


def filled_anomaly_traces(fig: go.Figure, series: pd.Series, name: str,
                           row: Optional[int] = None, col: Optional[int] = None,
                           color_pos: str = "rgba(74,222,128,0.20)",
                           color_neg: str = "rgba(248,113,113,0.20)",
                           line_color: str = "#60a5fa",
                           line_width: float = 1.5) -> None:
    """Add a filled anomaly trace: green fill above zero, red fill below zero.

    This makes polarity and amplitude of anomalies immediately visible
    without needing to inspect y-axis values.
    """
    kwargs = {"row": row, "col": col} if row is not None else {}
    s = pd.to_numeric(series, errors="coerce")

    # Filled areas
    fig.add_trace(go.Scatter(
        x=s.index, y=s.clip(lower=0),
        fill="tozeroy", fillcolor=color_pos,
        line=dict(width=0, color="rgba(0,0,0,0)"),
        name=f"{name} +", showlegend=False, hoverinfo="skip",
    ), **kwargs)
    fig.add_trace(go.Scatter(
        x=s.index, y=s.clip(upper=0),
        fill="tozeroy", fillcolor=color_neg,
        line=dict(width=0, color="rgba(0,0,0,0)"),
        name=f"{name} −", showlegend=False, hoverinfo="skip",
    ), **kwargs)
    # Main line on top
    fig.add_trace(go.Scatter(
        x=s.index, y=s,
        line=dict(color=line_color, width=line_width),
        name=name,
    ), **kwargs)


def add_anomaly_background_bands(fig: go.Figure, series: pd.Series,
                                  row: Optional[int] = None, col: Optional[int] = None) -> None:
    """Add horizontal background colour bands at ±2σ and ±3σ (robust).

    Bands make threshold crossings visible at a glance, which is essential
    for monitoring contexts where the analyst may be viewing the chart rapidly.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    median = float(s.median())
    mad = float((s - median).abs().median())
    sigma = 1.4826 * mad if mad > 0 else float(s.std())
    if not (np.isfinite(sigma) and sigma > 0):
        return
    kwargs = {"row": row, "col": col} if row is not None else {}
    # 2-3σ bands: yellow
    for sign in [1, -1]:
        fig.add_hrect(
            y0=median + sign * 2 * sigma,
            y1=median + sign * 3 * sigma,
            fillcolor="rgba(251,191,36,0.08)",
            line_width=0,
            **kwargs,
        )
        # >3σ bands: red
        extreme = median + sign * 3 * sigma
        edge = median + sign * 6 * sigma
        fig.add_hrect(
            y0=extreme, y1=edge,
            fillcolor="rgba(248,113,113,0.10)",
            line_width=0,
            **kwargs,
        )


def add_sigma_lines_enhanced(fig: go.Figure, series: pd.Series,
                              row: Optional[int] = None, col: Optional[int] = None) -> None:
    """Enhanced sigma lines with background bands and dotted thresholds."""
    add_anomaly_background_bands(fig, series, row=row, col=col)
    add_sigma_lines(fig, series, row=row, col=col)  # existing dotted lines


def overlay_gfz_secondary(fig: go.Figure, gfz_time: pd.Series, gfz_values: pd.Series,
                           idx_name: str, secondary_y: bool = True) -> None:
    """Overlay a GFZ index on an existing figure using a secondary y-axis.

    The secondary axis is rendered in a muted orange/amber to visually
    distinguish it from the primary local magnetometer signal.
    """
    fig.add_trace(go.Scatter(
        x=gfz_time, y=gfz_values,
        line=dict(color="#f59e0b", width=1.4, dash="dot"),
        name=f"GFZ {idx_name}",
        opacity=0.85,
        yaxis="y2" if secondary_y else "y",
    ))
    if secondary_y:
        fig.update_layout(
            yaxis2=dict(
                title=dict(text=f"GFZ {idx_name}", font=dict(color="#f59e0b", size=11)),
                tickfont=dict(color="#f59e0b", size=10),
                overlaying="y", side="right",
                showgrid=False,
                zeroline=False,
            )
        )


# =============================================================================
# API fetchers
# =============================================================================


@st.cache_data(ttl=900, show_spinner=False)
def fetch_gfz_index(start_iso: str, end_iso: str, index_name: str, status: str = "all") -> pd.DataFrame:
    params = {"start": start_iso, "end": end_iso, "index": index_name, "status": status}
    payload = fetch_json_cached("https://kp.gfz-potsdam.de/app/json/", params=params)
    if isinstance(payload, dict) and "datetime" in payload and index_name in payload:
        out = pd.DataFrame(
            {
                "time": pd.to_datetime(payload["datetime"], errors="coerce"),
                index_name: pd.to_numeric(payload[index_name], errors="coerce"),
            }
        )
        return out.dropna(subset=["time"]).sort_values("time")
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fdsn_text_cached(url: str, params: Optional[dict] = None) -> tuple[int, str]:
    """Fetch an FDSN text response.

    This mirrors the user's working INGV example: requests.get(url, params=params)
    with format='text'. It returns (status_code, text) so Streamlit can handle
    204 No Content without crashing.
    """
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        return response.status_code, response.text
    except requests.RequestException as exc:
        return 0, f"Connection error: {exc}"


def parse_fdsn_text_catalog(text_payload: str, latitude: float, longitude: float) -> pd.DataFrame:
    """Parse FDSN Event Service text format into the internal seismic schema.

    Typical FDSN text columns are pipe-separated and may start with a commented
    header such as '#EventID|Time|Latitude|Longitude|Depth/km|Author|Catalog|...'.
    The parser is intentionally tolerant because INGV, USGS and EMSC use small
    column-name differences.
    """
    if not text_payload or not text_payload.strip():
        return pd.DataFrame()

    lines = [line.strip() for line in text_payload.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()

    header = None
    data_lines = []
    for line in lines:
        if line.startswith("#") and "|" in line:
            header = [col.strip().lstrip("#") for col in line.split("|")]
        elif not line.startswith("#"):
            data_lines.append(line)

    if not data_lines:
        return pd.DataFrame()

    if header is None:
        # Conservative fallback based on common FDSN text ordering.
        header = [
            "EventID", "Time", "Latitude", "Longitude", "Depth/km",
            "Author", "Catalog", "Contributor", "ContributorID",
            "MagType", "Magnitude", "MagAuthor", "EventLocationName",
        ]

    try:
        raw = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep="|",
            names=header,
            engine="python",
        )
    except Exception:
        return pd.DataFrame()

    colmap = {str(col).lower().replace(" ", "").replace("_", ""): col for col in raw.columns}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in colmap:
                return colmap[key]
        return None

    time_col = pick("time", "origin_time", "datetime")
    lat_col = pick("latitude", "lat")
    lon_col = pick("longitude", "lon", "long")
    depth_col = pick("depth/km", "depth", "depthkm")
    mag_col = pick("magnitude", "mag")
    mag_type_col = pick("magtype", "mag_type", "magnitude type")
    id_col = pick("eventid", "event_id", "id")
    place_col = pick("eventlocationname", "place", "location")
    source_col = pick("author", "catalog", "contributor", "source")

    if time_col is None or lat_col is None or lon_col is None:
        return pd.DataFrame()

    events = pd.DataFrame({
        "time": pd.to_datetime(raw[time_col], errors="coerce"),
        "latitude": pd.to_numeric(raw[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(raw[lon_col], errors="coerce"),
        "depth_km": pd.to_numeric(raw[depth_col], errors="coerce") if depth_col else np.nan,
        "magnitude": pd.to_numeric(raw[mag_col], errors="coerce") if mag_col else np.nan,
        "mag_type": raw[mag_type_col].astype(str) if mag_type_col else "",
        "place": raw[place_col].astype(str) if place_col else "",
        "event_id": raw[id_col].astype(str) if id_col else "",
        "source": raw[source_col].astype(str) if source_col else "INGV/FDSN",
    })
    events = events.dropna(subset=["time", "latitude", "longitude"])
    if events.empty:
        return events
    events["distance_km"] = haversine_km(latitude, longitude, events["latitude"], events["longitude"])
    events["energy_j"] = seismic_energy_joule(events["magnitude"])
    return events.sort_values("time").reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fdsn_events(
    endpoint: str,
    start_time: str,
    end_time: str,
    latitude: float,
    longitude: float,
    max_radius_km: float,
    min_magnitude: float,
    limit: int,
) -> pd.DataFrame:
    """Fetch seismic events from FDSN.

    The first attempt uses the exact working INGV style supplied by the user:
    format='text'. This is generally more robust for INGV. If the server does
    not return text records, a GeoJSON fallback is attempted for USGS/EMSC.
    """
    degrees = max_radius_km / 111.19

    text_params = {
        "starttime": start_time,
        "endtime": end_time,
        "minmag": min_magnitude,
        "format": "text",
        "latitude": latitude,
        "longitude": longitude,
        "maxradius": degrees,
        "limit": limit,
        "orderby": "time-asc",
    }
    status_code, text_payload = fetch_fdsn_text_cached(endpoint, params=text_params)
    if status_code == 200:
        events = parse_fdsn_text_catalog(text_payload, latitude, longitude)
        if not events.empty:
            if "distance_km" in events.columns:
                events = events[events["distance_km"] <= max_radius_km]
            if "magnitude" in events.columns:
                events = events[events["magnitude"].fillna(-999) >= min_magnitude]
            return events.head(limit).sort_values("time")
    elif status_code == 204:
        return pd.DataFrame()

    # GeoJSON fallback for services that prefer it, especially USGS.
    json_params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "latitude": latitude,
        "longitude": longitude,
        "maxradius": degrees,
        "minmagnitude": min_magnitude,
        "limit": limit,
        "orderby": "time-asc",
    }
    payload = fetch_json_cached(endpoint, params=json_params)
    if not isinstance(payload, dict) or "features" not in payload:
        return pd.DataFrame()

    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry", {}) or {}
        coords = geom.get("coordinates", [np.nan, np.nan, np.nan])
        event_time = props.get("time")
        if isinstance(event_time, (int, float)):
            origin_time = pd.to_datetime(event_time, unit="ms", utc=True).tz_convert(None)
        else:
            origin_time = pd.to_datetime(event_time, errors="coerce")
        rows.append(
            {
                "time": origin_time,
                "latitude": coords[1] if len(coords) > 1 else np.nan,
                "longitude": coords[0] if len(coords) > 0 else np.nan,
                "depth_km": coords[2] if len(coords) > 2 else np.nan,
                "magnitude": props.get("mag"),
                "mag_type": props.get("magType"),
                "place": props.get("place"),
                "event_id": props.get("code") or props.get("ids"),
                "source": props.get("net") or props.get("sources"),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["time"] = pd.to_datetime(events["time"], errors="coerce")
    events["magnitude"] = pd.to_numeric(events["magnitude"], errors="coerce")
    events["depth_km"] = pd.to_numeric(events["depth_km"], errors="coerce")
    events["distance_km"] = haversine_km(latitude, longitude, events["latitude"], events["longitude"])
    events["energy_j"] = seismic_energy_joule(events["magnitude"])
    return events.dropna(subset=["time"]).sort_values("time")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo(latitude: float, longitude: float, forecast_days: int = 7) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": OPEN_METEO_APP4_HOURLY,
        "timezone": "Europe/Rome",
        "forecast_days": forecast_days,
    }
    payload = fetch_json_cached(url, params=params)
    if isinstance(payload, dict) and "hourly" in payload:
        df = pd.DataFrame(payload["hourly"])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            return df.dropna(subset=["time"]).sort_values("time")
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_donki(endpoint: str, start_date_iso: str, end_date_iso: str, api_key: str) -> pd.DataFrame:
    """Fetch NASA DONKI events using the same access pattern as app4.py.

    Valid endpoints include FLR, GST, CME, IPS, SEP, MPC, RBE, HSS and WSAEnlilSimulations.
    The function returns an empty DataFrame rather than crashing the dashboard.
    """
    key = api_key.strip() if api_key and api_key.strip() else NASA_DEMO_KEY
    url = f"{NASA_DONKI_URL}/{endpoint}"
    payload = fetch_json_cached(
        url,
        params={"startDate": start_date_iso, "endDate": end_date_iso, "api_key": key},
    )
    if isinstance(payload, list) and payload:
        return pd.DataFrame(payload)
    return pd.DataFrame()


# Plasma field names gained a "proton_" prefix in the RTSW schema.
RTSW_RENAMES = {
    "proton_speed": "speed",
    "proton_density": "density",
    "proton_temperature": "temperature",
}


def normalize_rtsw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt a SWPC RTSW payload to the legacy solar-wind schema.

    Since 2026-06-30 NOAA serves /json/rtsw/*.json instead of the retired
    /products/solar-wind/*.json files. Each record carries every available
    spacecraft (SOLAR-1, IMAP I-ALiRT, DSCOVR, ACE); only the one flagged
    ``active`` is operational, so rows must be filtered before plotting or the
    series interleaves two spacecraft. Magnetic field names are unchanged;
    plasma fields are renamed back. Legacy matrix payloads pass through
    untouched, so this is safe to call on either format.
    """
    if df.empty or "source" not in df.columns:
        return df
    out = df
    if "active" in out.columns:
        active = out[out["active"].fillna(False).astype(bool)]
        if not active.empty:
            out = active
    out = out.rename(columns={k: v for k, v in RTSW_RENAMES.items() if k in out.columns})
    if "time_tag" in out.columns:
        out = out.sort_values("time_tag").drop_duplicates(subset="time_tag", keep="last")
    return out.reset_index(drop=True)


def parse_f107_value(raw_payload: object) -> str:
    """Robust F10.7 parser copied in logic from app4.py."""
    try:
        if isinstance(raw_payload, list) and raw_payload:
            last = raw_payload[-1]
            if isinstance(last, list) and len(last) > 1:
                return str(last[1])
            if isinstance(last, dict):
                for key in ["flux", "value", "f107", "f10.7", "adjusted_flux", "observed_flux"]:
                    if key in last:
                        return str(last[key])
                values = list(last.values())
                if values:
                    return str(values[1] if len(values) > 1 else values[0])
    except Exception:
        return "N/A"
    return "N/A"


def parse_noaa_live() -> Dict[str, pd.DataFrame]:
    # NOAA: endpoint attivi + fallback legacy app4.py. I prodotti matrix hanno
    # intestazione nella prima riga; i prodotti JSON standard sono liste di dict.
    kp1m = pd.DataFrame(fetch_json_fallback_cached(tuple(NOAA_FALLBACKS["kp1m"])) or [])
    kp3h = fetch_matrix_fallback_cached(tuple(NOAA_FALLBACKS["kp3h"]))
    mag = normalize_rtsw_frame(fetch_matrix_fallback_cached(tuple(NOAA_FALLBACKS["mag"])))
    plasma = normalize_rtsw_frame(fetch_matrix_fallback_cached(tuple(NOAA_FALLBACKS["plasma"])))
    xray = pd.DataFrame(fetch_json_fallback_cached(tuple(NOAA_FALLBACKS["xray"])) or [])
    dst = pd.DataFrame(fetch_json_fallback_cached(tuple(NOAA_FALLBACKS["dst"])) or [])
    kpfcst_payload = fetch_json_fallback_cached(tuple(NOAA_FALLBACKS["kpfcst"]))
    if isinstance(kpfcst_payload, list) and kpfcst_payload and isinstance(kpfcst_payload[0], list):
        kpfcst = pd.DataFrame(kpfcst_payload[1:], columns=kpfcst_payload[0])
    else:
        kpfcst = pd.DataFrame(kpfcst_payload or [])
    f107_raw = fetch_json_fallback_cached(tuple(NOAA_FALLBACKS["f107"]))
    f107_value = parse_f107_value(f107_raw)

    for df in [kp1m, kp3h, mag, plasma, xray, dst, kpfcst]:
        for col in df.columns:
            if "time" in str(col).lower() or col in {"time_tag"}:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                try:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    # Keep original text columns such as time labels, but convert numeric-like columns.
                    if converted.notna().sum() > 0:
                        df[col] = converted
                except (TypeError, ValueError):
                    pass
    return {
        "kp1m": kp1m,
        "kp3h": kp3h,
        "mag": mag,
        "plasma": plasma,
        "xray": xray,
        "dst": dst,
        "kpfcst": kpfcst,
        "f107_value": f107_value,
    }


# =============================================================================
# Data preparation
# =============================================================================


def parse_magnetometer(df_raw: pd.DataFrame, unit: str, spike_sigma: float) -> pd.DataFrame:
    df = prepare_time_index(df_raw)
    aliases = {
        "X": ["X", "Bx", "HX", "Hx", "H", "north", "N"],
        "Y": ["Y", "By", "HY", "Hy", "D", "east", "E"],
        "Z": ["Z", "Bz", "HZ", "Hz", "vertical", "V"],
        "F": ["F", "Bt", "Total", "TotalField", "TMI", "Field"],
        "T_sensor": ["T", "Temp", "temperature", "TEMP", "sensor_temp"],
    }
    found: Dict[str, str] = {}
    for target, names in aliases.items():
        for name in names:
            if name in df.columns:
                found[target] = name
                break

    out = pd.DataFrame(index=df.index)
    for target, source in found.items():
        out[target] = to_numeric_series(df, source)
        if target in {"X", "Y", "Z", "F"}:
            out[target] = unit_to_nt(out[target], unit)

    if "F" not in out.columns and all(col in out.columns for col in ["X", "Y", "Z"]):
        out["F"] = np.sqrt(out["X"] ** 2 + out["Y"] ** 2 + out["Z"] ** 2)
    if "F" not in out.columns:
        raise ValueError("Servono F oppure X/Y/Z per calcolare il campo totale.")

    for col in [c for c in ["X", "Y", "Z", "F"] if c in out.columns]:
        out[col] = hampel_filter(out[col], window=11, n_sigmas=spike_sigma)
    return out.dropna(subset=["F"]).sort_index()


def parse_multiparameter_csv(df_raw: pd.DataFrame, dayfirst: bool = True) -> pd.DataFrame:
    df = prepare_time_index(df_raw, dayfirst=dayfirst)
    for col in df.columns:
        if col != "Datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].dropna(how="all")


def parse_seismic_csv(df_raw: pd.DataFrame, lat0: float, lon0: float, dayfirst: bool = True) -> pd.DataFrame:
    df = prepare_time_index(df_raw, dayfirst=dayfirst).reset_index().rename(columns={"Datetime": "time"})
    rename_candidates = {
        "latitude": ["latitude", "lat", "Lat", "LAT"],
        "longitude": ["longitude", "lon", "Lon", "LON", "long"],
        "depth_km": ["depth", "depth_km", "Depth", "prof", "profondita"],
        "magnitude": ["magnitude", "mag", "Magnitude", "Ml", "ML", "Mw", "MW"],
    }
    out = pd.DataFrame({"time": df["time"]})
    for target, names in rename_candidates.items():
        source = next((name for name in names if name in df.columns), None)
        if source:
            out[target] = pd.to_numeric(df[source], errors="coerce")
    if "latitude" in out.columns and "longitude" in out.columns:
        out["distance_km"] = haversine_km(lat0, lon0, out["latitude"], out["longitude"])
    if "magnitude" in out.columns:
        out["energy_j"] = seismic_energy_joule(out["magnitude"])
    return out.dropna(subset=["time"]).sort_values("time")


# =============================================================================
# Plot helpers
# =============================================================================


def add_sigma_lines(fig: go.Figure, series: pd.Series, row: Optional[int] = None, col: Optional[int] = None) -> None:
    z = pd.to_numeric(series, errors="coerce").dropna()
    if z.empty:
        return
    median = z.median()
    sigma = 1.4826 * (z - median).abs().median()
    sigma = float(sigma if sigma > 0 else z.std())
    if not np.isfinite(sigma) or sigma == 0:
        return
    kwargs = {"row": row, "col": col} if row is not None and col is not None else {}
    for mult, color in [(2, "#fbbf24"), (3, "#fb923c"), (4, "#f87171")]:
        fig.add_hline(
            y=median + mult * sigma,
            line_dash="dot",
            line_color=color,
            annotation_text=f"+{mult} robust σ",
            annotation_font_size=10,
            **kwargs,
        )
        fig.add_hline(
            y=median - mult * sigma,
            line_dash="dot",
            line_color=color,
            annotation_text=f"-{mult} robust σ",
            annotation_font_size=10,
            **kwargs,
        )


def make_multi_axis_timeseries(
    data: pd.DataFrame,
    time_col: Optional[str],
    variables: List[str],
    title: str,
    y_title: str = "value",
) -> go.Figure:
    fig = go.Figure()
    x = data[time_col] if time_col else data.index
    for var in variables:
        if var in data.columns:
            fig.add_trace(go.Scatter(x=x, y=data[var], mode="lines", name=var, line=dict(width=1.3)))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, yaxis_title=y_title)
    return fig




def finite_min_max(values: Iterable[pd.Series]) -> Tuple[float, float]:
    """Return finite min/max for one or more numeric series."""
    finite_parts = []
    for values_i in values:
        arr = pd.to_numeric(pd.Series(values_i).dropna(), errors="coerce")
        arr = arr[np.isfinite(arr)]
        if not arr.empty:
            finite_parts.append(arr)
    if not finite_parts:
        return 0.0, 1.0
    joined = pd.concat(finite_parts)
    y_min = float(joined.min())
    y_max = float(joined.max())
    if y_min == y_max:
        pad = max(abs(y_min) * 0.01, 1.0)
    else:
        pad = max((y_max - y_min) * 0.04, 1e-9)
    return y_min - pad, y_max + pad


def y_range_controls(key: str, label: str, series_list: Iterable[pd.Series]) -> Tuple[Optional[float], Optional[float]]:
    """Streamlit controls for y-axis range.

    Default behaviour is data-driven: the range is computed from the finite
    minimum and maximum of the plotted series. Manual mode is useful for
    comparing different time windows with the same ordinate scale.
    """
    y_min_auto, y_max_auto = finite_min_max(series_list)
    mode = st.selectbox(
        f"Range ordinate - {label}",
        ["Automatico su min/max dati", "Manuale"],
        index=0,
        key=f"{key}_mode",
    )
    if mode == "Manuale":
        c1, c2 = st.columns(2)
        with c1:
            y_min = st.number_input(
                f"Y min - {label}", value=float(y_min_auto), format="%.6g", key=f"{key}_min"
            )
        with c2:
            y_max = st.number_input(
                f"Y max - {label}", value=float(y_max_auto), format="%.6g", key=f"{key}_max"
            )
        if y_min >= y_max:
            st.warning("Y min deve essere minore di Y max. Uso il range automatico.")
            return y_min_auto, y_max_auto
        return float(y_min), float(y_max)
    return y_min_auto, y_max_auto


def apply_y_range(fig: go.Figure, y_min: Optional[float], y_max: Optional[float], row: Optional[int] = None) -> None:
    """Apply a y-axis range to a Plotly figure."""
    if y_min is None or y_max is None or not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    if row is None:
        fig.update_yaxes(range=[y_min, y_max])
    else:
        fig.update_yaxes(range=[y_min, y_max], row=row, col=1)

def _depth_to_hex(depth: float, depth_min: float, depth_max: float) -> str:
    """Interpolate the DEPTH_COLORSCALE to get a hex colour for a given depth value.

    Used to colour the animated pulse rings of recent events with the same
    perceptual scale as the main event layer, so depth information is never lost.
    """
    DEPTH_COLORSCALE_STOPS = [
        (0.00, (30,  144, 255)),   # #1e90ff
        (0.20, (0,   207, 255)),   # #00cfff
        (0.40, (87,  226, 154)),   # #57e29a
        (0.60, (249, 199,  79)),   # #f9c74f
        (0.80, (247, 127,   0)),   # #f77f00
        (1.00, (214,  40,  40)),   # #d62828
    ]
    span = max(depth_max - depth_min, 1.0)
    t = float(np.clip((depth - depth_min) / span, 0.0, 1.0))
    for i in range(len(DEPTH_COLORSCALE_STOPS) - 1):
        t0, rgb0 = DEPTH_COLORSCALE_STOPS[i]
        t1, rgb1 = DEPTH_COLORSCALE_STOPS[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / max(t1 - t0, 1e-9)
            r = int(rgb0[0] + alpha * (rgb1[0] - rgb0[0]))
            g = int(rgb0[1] + alpha * (rgb1[1] - rgb0[1]))
            b = int(rgb0[2] + alpha * (rgb1[2] - rgb0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ffffff"


def plot_event_map(events: pd.DataFrame, title: str, center_lat: float = LAT_DEFAULT, center_lon: float = LON_DEFAULT) -> go.Figure:
    """Map earthquake epicentres around the Aeolian Islands.

    The map is centred on Vulcano Island. Marker colour encodes hypocentral
    depth with a perceptually ordered colorscale (blue=shallow → red=deep)
    and marker size scales exponentially with magnitude for clear visual
    separation between small and large events.
    Additional features:
    - Distance rings at 10 / 20 / 30 km from Vulcano
    - Dark basemap (carto-darkmatter)
    - Magnitude reference entries in the legend
    - Refined colorbar and hover tooltip
    - Animated pulse rings on the 5 most recent events (colour = depth, size = magnitude)
    """
    fig = go.Figure()

    # Aeolian island reference coordinates (approximate island centres).
    aeolian_islands = pd.DataFrame(
        [
            {"name": "Vulcano",   "lat": 38.404, "lon": 14.962},
            {"name": "Lipari",    "lat": 38.467, "lon": 14.955},
            {"name": "Salina",    "lat": 38.560, "lon": 14.840},
            {"name": "Panarea",   "lat": 38.638, "lon": 15.077},
            {"name": "Stromboli", "lat": 38.789, "lon": 15.213},
            {"name": "Filicudi",  "lat": 38.562, "lon": 14.566},
            {"name": "Alicudi",   "lat": 38.536, "lon": 14.351},
        ]
    )

    # Perceptually ordered depth colorscale: blue (shallow) → cyan → green → yellow → orange → red (deep).
    DEPTH_COLORSCALE = [
        [0.00, "#1e90ff"],   # ~0 km   – blu intenso
        [0.20, "#00cfff"],   # ~10 km  – azzurro
        [0.40, "#57e29a"],   # ~20 km  – verde acqua
        [0.60, "#f9c74f"],   # ~30 km  – giallo
        [0.80, "#f77f00"],   # ~40 km  – arancio
        [1.00, "#d62828"],   # profond  – rosso
    ]

    # ── Distance rings around Vulcano ─────────────────────────────────────────
    def _geodesic_ring(clat: float, clon: float, radius_km: float, n: int = 120) -> Tuple[List[float], List[float]]:
        """Approximate geodesic circle as a closed polygon."""
        lats, lons = [], []
        for i in range(n + 1):
            angle = 2.0 * math.pi * i / n
            dlat = math.degrees(radius_km / 6371.0 * math.cos(angle))
            dlon = math.degrees(
                radius_km / 6371.0 * math.sin(angle) / max(math.cos(math.radians(clat)), 1e-9)
            )
            lats.append(clat + dlat)
            lons.append(clon + dlon)
        return lats, lons

    ring_styles = [
        (10, "rgba(255,255,255,0.30)", "10 km"),
        (20, "rgba(255,255,255,0.18)", "20 km"),
        (30, "rgba(255,255,255,0.10)", "30 km"),
    ]
    for r_km, ring_color, ring_label in ring_styles:
        r_lats, r_lons = _geodesic_ring(center_lat, center_lon, r_km)
        fig.add_trace(
            go.Scattermapbox(
                lat=r_lats, lon=r_lons,
                mode="lines",
                line=dict(color=ring_color, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Label placed at the top of each ring.
        fig.add_trace(
            go.Scattermapbox(
                lat=[center_lat + r_km / 111.0 + 0.005],
                lon=[center_lon],
                mode="text",
                text=[ring_label],
                textfont=dict(size=9, color="rgba(220,220,220,0.55)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # ── Seismic events ────────────────────────────────────────────────────────
    depth_min_global = 0.0
    depth_max_global = 50.0

    if not events.empty:
        plot_df = events.copy()
        plot_df["latitude"]    = pd.to_numeric(plot_df.get("latitude"),    errors="coerce")
        plot_df["longitude"]   = pd.to_numeric(plot_df.get("longitude"),   errors="coerce")
        plot_df["magnitude"]   = pd.to_numeric(plot_df.get("magnitude"),   errors="coerce")
        plot_df["depth_km"]    = pd.to_numeric(plot_df.get("depth_km"),    errors="coerce")
        plot_df["distance_km"] = pd.to_numeric(plot_df.get("distance_km"), errors="coerce")
        plot_df = plot_df.dropna(subset=["latitude", "longitude"])

        if not plot_df.empty:
            mag   = plot_df["magnitude"].fillna(1.0).clip(lower=0.1, upper=8.0)
            depth = plot_df["depth_km"].fillna(0.0)

            # Exponential size scaling: Mw 1 → ~7 px, Mw 3 → ~22 px, Mw 5 → ~58 px
            size = (3.5 + 2.8 * (mag ** 1.9)).clip(upper=80)

            depth_min_global = max(0.0, float(depth.min())) if depth.notna().any() else 0.0
            depth_max_global = float(depth.max()) if depth.notna().any() else 50.0
            if depth_max_global <= depth_min_global:
                depth_max_global = depth_min_global + 1.0

            place       = plot_df.get("place",  pd.Series("—", index=plot_df.index)).astype(str)
            source      = plot_df.get("source", pd.Series("—", index=plot_df.index)).astype(str)
            time_values = pd.to_datetime(plot_df.get("time"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")

            customdata = np.column_stack([
                plot_df["magnitude"].fillna(np.nan),
                depth,
                plot_df["distance_km"].fillna(np.nan),
                place,
                source,
                time_values.fillna("—"),
            ])

            fig.add_trace(
                go.Scattermapbox(
                    lat=plot_df["latitude"],
                    lon=plot_df["longitude"],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=depth,
                        colorscale=DEPTH_COLORSCALE,
                        cmin=depth_min_global,
                        cmax=depth_max_global,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Profondità (km)", side="right", font=dict(size=12)),
                            thickness=16,
                            len=0.60,
                            yanchor="middle",
                            y=0.50,
                            x=1.01,
                            tickfont=dict(size=11),
                            outlinewidth=0,
                            bgcolor="rgba(13,17,23,0.75)",
                            borderwidth=0,
                        ),
                        opacity=0.90,
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>Terremoto</b><br>"
                        "Tempo: %{customdata[5]}<br>"
                        "Magnitudo: <b>Mw %{customdata[0]:.2f}</b><br>"
                        "Profondità: <b>%{customdata[1]:.1f} km</b><br>"
                        "Distanza da Vulcano: %{customdata[2]:.1f} km<br>"
                        "Lat / Lon: %{lat:.4f}°, %{lon:.4f}°<br>"
                        "Località: %{customdata[3]}<br>"
                        "Fonte: %{customdata[4]}<extra></extra>"
                    ),
                    name="Sismi  (colore = profondità, dim = Mw)",
                )
            )

            # Magnitude reference entries (invisible points, legend only).
            for ref_mag in [1.0, 2.0, 3.0, 4.0, 5.0]:
                ref_size = float((3.5 + 2.8 * (ref_mag ** 1.9)))
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[None], lon=[None],
                        mode="markers",
                        marker=dict(size=ref_size, color="#b0b8c8", opacity=0.80),
                        name=f"Mw {ref_mag:.0f}",
                        showlegend=True,
                    )
                )

    # ── Aeolian island landmarks ───────────────────────────────────────────────
    fig.add_trace(
        go.Scattermapbox(
            lat=aeolian_islands["lat"],
            lon=aeolian_islands["lon"],
            mode="markers+text",
            marker=dict(size=8, color="#e2e8f0", opacity=0.85),
            text=aeolian_islands["name"],
            textposition="top center",
            textfont=dict(size=10, color="#e2e8f0"),
            hovertemplate="<b>%{text}</b><br>%{lat:.3f}°N, %{lon:.3f}°E<extra></extra>",
            name="Isole Eolie",
        )
    )

    # ── Monitoring station – Vulcano ───────────────────────────────────────────
    fig.add_trace(
        go.Scattermapbox(
            lat=[center_lat],
            lon=[center_lon],
            mode="markers+text",
            marker=dict(size=22, color="#f87171", opacity=1.0),
            text=["▲ APS 1540"],
            textposition="bottom right",
            textfont=dict(size=11, color="#f87171"),
            hovertemplate=(
                "<b>⛰ Isola di Vulcano</b><br>"
                "Stazione APS 1540<br>"
                "%{lat:.4f}°N, %{lon:.4f}°E<extra></extra>"
            ),
            name="Vulcano – APS 1540",
        )
    )

    # ── Animated pulse rings for the 5 most recent events ─────────────────────
    # Strategy: Plotly animation frames cycle through 8 opacity steps to create
    # a smooth "sonar ping" effect on the most recent events only.
    # Each event keeps its own depth colour and magnitude-proportional ring size
    # so no information is lost compared with the static layer below.
    # The distance from Vulcano is drawn as a line to the station marker so the
    # analyst can immediately read the threat proximity.
    if not events.empty:
        ev_sorted = events.copy()
        ev_sorted["time"] = pd.to_datetime(ev_sorted.get("time"), errors="coerce")
        ev_sorted = (
            ev_sorted
            .dropna(subset=["latitude", "longitude", "time"])
            .sort_values("time", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

        if not ev_sorted.empty:
            # Rank labels shown in hover
            rank_labels = ["#1 — più recente", "#2", "#3", "#4", "#5"]

            # Pulse opacities: 8 frames → smooth sine-like cycle
            N_FRAMES = 8
            opacities_cycle = [
                round(0.15 + 0.80 * math.sin(math.pi * k / (N_FRAMES - 1)) ** 2, 3)
                for k in range(N_FRAMES)
            ]
            # Ring size multipliers per frame (expand while fading)
            size_mults = [
                round(1.0 + 0.90 * math.sin(math.pi * k / (N_FRAMES - 1)), 3)
                for k in range(N_FRAMES)
            ]

            # Pre-compute per-event properties
            rec_lats, rec_lons, rec_sizes_base, rec_colors, rec_hover = [], [], [], [], []
            for _, row in ev_sorted.iterrows():
                rlat = float(row.get("latitude", 0.0))
                rlon = float(row.get("longitude", 0.0))
                rmag = float(row.get("magnitude", 1.0) or 1.0)
                rmag = max(0.1, min(rmag, 8.0))
                rdepth = float(row.get("depth_km", 0.0) or 0.0)
                rdist = row.get("distance_km", np.nan)
                rdist_str = f"{float(rdist):.1f} km" if pd.notna(rdist) and np.isfinite(float(rdist)) else "—"
                rtime = row.get("time", pd.NaT)
                try:
                    rtime_str = pd.Timestamp(rtime).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    rtime_str = "—"
                rplace = str(row.get("place", "—") or "—")
                rcolor = _depth_to_hex(rdepth, depth_min_global, depth_max_global)
                # Pulse ring base size: larger than the event marker to surround it
                base_size = float((3.5 + 2.8 * (rmag ** 1.9))) * 1.55
                rec_lats.append(rlat)
                rec_lons.append(rlon)
                rec_sizes_base.append(base_size)
                rec_colors.append(rcolor)
                rec_hover.append(
                    f"<b>⚡ Evento recente {rank_labels[len(rec_lats)-1]}</b><br>"
                    f"Tempo: {rtime_str}<br>"
                    f"Magnitudo: <b>{rmag:.1f}</b><br>"
                    f"Profondità: <b>{rdepth:.1f} km</b><br>"
                    f"Distanza da Vulcano: <b>{rdist_str}</b><br>"
                    f"Lat/Lon: {rlat:.4f}°, {rlon:.4f}°<br>"
                    f"Località: {rplace}<extra></extra>"
                )

            # ── Static distance lines: epicentre → station ─────────────────────
            # One thin line per event in the event's depth colour so the
            # analyst can visually read proximity at a glance.
            for i in range(len(ev_sorted)):
                rank_str = rank_labels[i]
                hex_c = rec_colors[i]
                rdist_val = ev_sorted.iloc[i].get("distance_km", np.nan)
                rdist_s = f"{float(rdist_val):.1f} km" if pd.notna(rdist_val) and np.isfinite(float(rdist_val)) else "—"
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[rec_lats[i], center_lat, None],
                        lon=[rec_lons[i], center_lon, None],
                        mode="lines",
                        line=dict(color=hex_c, width=1.6),
                        opacity=0.55,
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"dist_{rank_str}",
                    )
                )
                # Mid-point label for distance
                mid_lat = (rec_lats[i] + center_lat) / 2
                mid_lon = (rec_lons[i] + center_lon) / 2
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[mid_lat],
                        lon=[mid_lon],
                        mode="text",
                        text=[rdist_s],
                        textfont=dict(size=10, color=hex_c),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            # ── Animated pulse rings ───────────────────────────────────────────
            # Index of the animated trace in fig.data (added right after the static layers above)
            # We add it first in frame-0 state, then build frames that update it.
            fig.add_trace(
                go.Scattermapbox(
                    lat=rec_lats,
                    lon=rec_lons,
                    mode="markers",
                    marker=dict(
                        size=[s * size_mults[0] for s in rec_sizes_base],
                        color=rec_colors,
                        opacity=opacities_cycle[0],
                    ),
                    hovertemplate=rec_hover,
                    name="⚡ Ultimi 5 eventi",
                    showlegend=True,
                )
            )

            # Build animation frames
            animated_trace_idx = len(fig.data) - 1
            frames = []
            for k in range(N_FRAMES):
                frame_trace = go.Scattermapbox(
                    lat=rec_lats,
                    lon=rec_lons,
                    mode="markers",
                    marker=dict(
                        size=[s * size_mults[k] for s in rec_sizes_base],
                        color=rec_colors,
                        opacity=opacities_cycle[k],
                    ),
                    hovertemplate=rec_hover,
                    name="⚡ Ultimi 5 eventi",
                )
                frames.append(go.Frame(
                    data=[frame_trace],
                    traces=[animated_trace_idx],
                    name=str(k),
                ))
            fig.frames = frames

    map_layout = dict(PLOTLY_LAYOUT)
    map_layout["margin"] = dict(l=10, r=10, t=60, b=10)
    fig.update_layout(
        **map_layout,
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        height=800,
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8.5,
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(13,17,23,0.75)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            font=dict(size=11, color="#c8d0dc"),
            title=dict(text="Legenda", font=dict(size=11, color="#93a4b7")),
        ),
        # Animation controls: auto-play, loop, hidden slider/buttons
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            visible=False,   # hidden – animation starts automatically via JS below
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=350, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                    mode="immediate",
                    loop=True,
                )],
            )],
        )],
    )

    # Auto-play the animation as soon as the figure renders
    # (Plotly JS picks up the "autoplay" attribute on the first frame group)
    if fig.frames:
        fig.layout.sliders = [dict(
            active=0,
            visible=False,   # hide the scrubber bar
            steps=[dict(method="animate", args=[[f.name], dict(
                mode="immediate",
                frame=dict(duration=350, redraw=True),
                transition=dict(duration=0),
            )]) for f in fig.frames],
        )]

    return fig


# =============================================================================
# State model
# =============================================================================


def compute_space_weather_quality(kp: float, dst: float, bz: float, solar_speed: float) -> Tuple[float, List[str]]:
    """Return confidence multiplier from 0 to 1 and diagnostic flags."""
    quality = 1.0
    flags: List[str] = []
    if np.isfinite(kp) and kp >= 5:
        quality -= 0.35
        flags.append("Kp >= 5: contributo esterno dominante")
    elif np.isfinite(kp) and kp >= 4:
        quality -= 0.15
        flags.append("Kp 4-5: condizioni geomagnetiche attive")
    if np.isfinite(dst) and dst <= -50:
        quality -= 0.25
        flags.append("Dst <= -50 nT: tempesta geomagnetica moderata")
    if np.isfinite(bz) and bz <= -10:
        quality -= 0.15
        flags.append("Bz GSM <= -10 nT: accoppiamento solare-terrestre efficiente")
    if np.isfinite(solar_speed) and solar_speed >= 650:
        quality -= 0.10
        flags.append("vento solare veloce")
    return max(0.15, quality), flags


def compute_volcano_state(
    mag_z: float,
    seismic_rate_z: float,
    seismic_energy_z: float,
    geochem_z: float,
    deformation_z: float,
    temp_z: float,
    space_quality: float,
    space_flags: List[str],
) -> VolcanoState:
    weights = {
        "mag": 0.26,
        "seismic_rate": 0.22,
        "seismic_energy": 0.17,
        "geochem": 0.18,
        "deformation": 0.12,
        "temp": 0.05,
    }
    raw_score = (
        weights["mag"] * normalize_01(mag_z, 1.5, 5.0)
        + weights["seismic_rate"] * normalize_01(seismic_rate_z, 1.0, 4.0)
        + weights["seismic_energy"] * normalize_01(seismic_energy_z, 1.0, 4.0)
        + weights["geochem"] * normalize_01(geochem_z, 1.0, 4.0)
        + weights["deformation"] * normalize_01(deformation_z, 1.0, 4.0)
        + weights["temp"] * normalize_01(temp_z, 1.0, 4.0)
    )
    score = 100.0 * raw_score
    if score < 25:
        level = "VERDE - background / nessuna evidenza multiparametrica"
        css = "green"
        action = "Continuare il monitoraggio ordinario e validare la qualita' dei dati."
    elif score < 50:
        level = "GIALLO - anomalia debole o non persistente"
        css = "yellow"
        action = "Verificare sensori, meteo, space weather e coerenza temporale tra parametri."
    elif score < 75:
        level = "ARANCIO - anomalia multiparametrica significativa"
        css = "orange"
        action = "Incrementare frequenza di controllo, confrontare con stazioni di riferimento e dati INGV."
    else:
        level = "ROSSO - anomalia forte / possibile escalation"
        css = "red"
        action = "Attivare revisione esperta immediata; non usare l'indice come unico criterio decisionale."

    explanation = (
        f"Score={score:.1f}/100; z magnetico={mag_z:.2f}, z rate sismico={seismic_rate_z:.2f}, "
        f"z energia sismica={seismic_energy_z:.2f}, z geochimica={geochem_z:.2f}, "
        f"z deformazione={deformation_z:.2f}, z termico={temp_z:.2f}."
    )
    if space_flags:
        explanation += " Qualita' vulcanomagnetica ridotta: " + "; ".join(space_flags) + "."
    return VolcanoState(level, css, score, 100.0 * space_quality, explanation, action)


def compute_daily_seismic_metrics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["count", "energy_j", "max_mag"])
    ev = events.copy().set_index("time")
    metrics = pd.DataFrame()
    metrics["count"] = ev["magnitude"].resample("1D").count()
    metrics["energy_j"] = ev["energy_j"].resample("1D").sum(min_count=1)
    metrics["max_mag"] = ev["magnitude"].resample("1D").max()
    return metrics.fillna({"count": 0, "energy_j": 0})


def mag_badge_class(mag: float) -> str:
    """Return CSS badge class based on magnitude."""
    if mag < 2.0:
        return "m-minor"
    if mag < 3.0:
        return "m-light"
    if mag < 4.0:
        return "m-moderate"
    return "m-strong"


def mag_emoji(mag: float) -> str:
    if mag < 2.0:
        return "🟢"
    if mag < 3.0:
        return "🟡"
    if mag < 4.0:
        return "🟠"
    return "🔴"


def dot_class(rank: int) -> str:
    """Return live-dot CSS class for ranking 1-5."""
    if rank == 1:
        return "live-dot"
    if rank == 2:
        return "live-dot orange"
    if rank == 3:
        return "live-dot yellow"
    return "live-dot gray"


def card_class(rank: int) -> str:
    mapping = {1: "seis-card", 2: "seis-card recent2", 3: "seis-card recent3",
               4: "seis-card recent4", 5: "seis-card recent5"}
    return mapping.get(rank, "seis-card recent5")


def render_recent_events_flash(events: pd.DataFrame, n: int = 5) -> None:
    """Render the N most recent seismic events as animated flash cards.

    The most recent event pulses red, the second orange, the third yellow,
    and older ones are displayed statically. This allows rapid visual triage
    of the freshest seismicity without scanning a table.
    """
    if events.empty:
        return
    ev = events.copy()
    ev["time"] = pd.to_datetime(ev["time"], errors="coerce")
    ev = ev.dropna(subset=["time"]).sort_values("time", ascending=False).head(n).reset_index(drop=True)

    cards_html = "<div style='margin-bottom:6px'>"
    for i, row in ev.iterrows():
        rank = i + 1
        mag = float(row.get("magnitude", np.nan))
        mag_str = f"{mag:.1f}" if np.isfinite(mag) else "—"
        depth = float(row.get("depth_km", np.nan))
        depth_str = f"{depth:.1f} km" if np.isfinite(depth) else "—"
        dist = float(row.get("distance_km", np.nan))
        dist_str = f"{dist:.1f} km" if np.isfinite(dist) else "—"
        place = str(row.get("place", "—") or "—")
        source = str(row.get("source", "—") or "—")
        t = row["time"]
        try:
            time_str = pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            time_str = "—"
        mag_type = str(row.get("mag_type", "") or "").strip()
        if not mag_type or mag_type in {"nan", "None"}:
            mag_type = "Mw"
        label_text = "🕐 più recente" if rank == 1 else f"#{rank}"
        badge = mag_badge_class(mag) if np.isfinite(mag) else "m-minor"
        emoji = mag_emoji(mag) if np.isfinite(mag) else "⚫"
        dot = f"<span class='{dot_class(rank)}'></span>"
        cards_html += f"""
        <div class='{card_class(rank)}'>
          <div>{dot}<span class='seis-badge {badge}'>{mag_type} {mag_str}</span>
               {emoji} <b style='font-size:13px;color:#e2e8f0'>{label_text}</b></div>
          <div class='seis-time'>⏱ {time_str}</div>
          <div class='seis-place'>📍 {place}</div>
          <div class='seis-meta'>
            Profondità: <b>{depth_str}</b> &nbsp;|&nbsp;
            Distanza da Vulcano: <b>{dist_str}</b> &nbsp;|&nbsp;
            Fonte: {source}
          </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def detect_seismic_swarm(events: pd.DataFrame, window_hours: int = 6, min_events: int = 5) -> Optional[str]:
    """Detect a possible seismic swarm in the most recent time window.

    Returns a warning message string if a swarm is detected, else None.
    A swarm is defined (heuristically) as >= min_events in the last window_hours
    with at least 3 events within 20 km of Vulcano, no single dominant event
    (Mmax < 3.5). This is a triage indicator, not an official classification.
    """
    if events.empty or "time" not in events.columns:
        return None
    ev = events.copy()
    ev["time"] = pd.to_datetime(ev["time"], errors="coerce")
    cutoff = ev["time"].max() - pd.Timedelta(hours=window_hours)
    recent = ev[ev["time"] >= cutoff]
    if len(recent) < min_events:
        return None
    near = recent[recent.get("distance_km", pd.Series(dtype=float)).fillna(999) <= 20] if "distance_km" in recent.columns else recent
    if len(near) < 3:
        return None
    mmax = recent["magnitude"].max() if "magnitude" in recent.columns else np.nan
    if np.isfinite(mmax) and mmax >= 4.0:
        return None  # mainshock-aftershock pattern, not flagged as swarm
    n = len(recent)
    n_near = len(near)
    mmax_str = f"Mmax={mmax:.1f}" if np.isfinite(mmax) else "Mmax=?"
    return (
        f"⚠️ Possibile sciame sismico: {n} eventi nelle ultime {window_hours} h, "
        f"di cui {n_near} entro 20 km da Vulcano ({mmax_str}). "
        "Verifica la migrazione spazio-temporale e confronta con dati magnetici e geochimici."
    )


def plot_seismic_depth_timeline(events: pd.DataFrame) -> go.Figure:
    """Time vs depth scatter plot to visualise fluid/dyke migration patterns.

    In volcanic areas, upward migration of seismicity over days–weeks can
    indicate magma or fluid ascent. Colour encodes magnitude; point size depth.
    """
    ev = events.dropna(subset=["time", "depth_km"]).copy()
    ev["time"] = pd.to_datetime(ev["time"], errors="coerce")
    ev["magnitude"] = pd.to_numeric(ev["magnitude"], errors="coerce").fillna(0.5)
    ev["depth_km"] = pd.to_numeric(ev["depth_km"], errors="coerce")
    size = (4 + 3.5 * (ev["magnitude"].clip(0.1, 7) ** 1.7)).clip(upper=55)
    fig = go.Figure(go.Scatter(
        x=ev["time"], y=ev["depth_km"],
        mode="markers",
        marker=dict(
            size=size,
            color=ev["magnitude"],
            colorscale="RdYlGn_r",
            cmin=0, cmax=5,
            showscale=True,
            colorbar=dict(title="Magnitudo", thickness=14, len=0.55, tickfont=dict(size=10)),
            opacity=0.80,
            line=dict(width=0.5, color="rgba(255,255,255,0.20)"),
        ),
        customdata=np.column_stack([
            ev["magnitude"].round(2),
            ev["depth_km"].round(1),
            ev.get("distance_km", pd.Series(np.nan, index=ev.index)).round(1),
            ev.get("place", pd.Series("—", index=ev.index)).astype(str),
        ]),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
            "Magnitudo: <b>%{customdata[0]}</b><br>"
            "Profondità: <b>%{customdata[1]} km</b><br>"
            "Distanza: %{customdata[2]} km<br>"
            "Località: %{customdata[3]}<extra></extra>"
        ),
    ))
    fig.update_yaxes(autorange="reversed", title="Profondità (km)")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Migrazione temporale della sismicità per profondità",
        height=420,
        xaxis_title="Tempo",
    )
    return fig


def plot_seismic_magnitude_frequency(events: pd.DataFrame) -> go.Figure:
    """Gutenberg-Richter magnitude-frequency plot (cumulative) with b-value estimate.

    The b-value of the Gutenberg-Richter relation (log N = a - b*M) is a useful
    indicator: values significantly below 1 suggest tectonic stress dominance,
    while b > 1.2 can indicate fluid/thermal involvement typical of volcanic areas.
    """
    ev = events.dropna(subset=["magnitude"]).copy()
    ev["magnitude"] = pd.to_numeric(ev["magnitude"], errors="coerce").dropna()
    if len(ev) < 5:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="G-R: dati insufficienti (min. 5 eventi)")
        return fig

    mags = ev["magnitude"].sort_values()
    mag_bins = np.arange(np.floor(mags.min() * 2) / 2, np.ceil(mags.max() * 2) / 2 + 0.5, 0.5)
    cumul = [(mags >= m).sum() for m in mag_bins]

    # b-value estimation (least-squares on log10 N vs M)
    b_val = np.nan
    a_val = np.nan
    try:
        mask = np.array(cumul) > 0
        if mask.sum() >= 3:
            x_fit = mag_bins[mask]
            y_fit = np.log10(np.array(cumul)[mask])
            coeffs = np.polyfit(x_fit, y_fit, 1)
            b_val = -coeffs[0]
            a_val = coeffs[1]
    except Exception:
        pass

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mag_bins, y=cumul,
        mode="markers+lines",
        name="N cumulativo",
        marker=dict(size=7, color="#60a5fa"),
        line=dict(color="#60a5fa", width=1.5),
    ))
    if np.isfinite(b_val):
        fit_y = 10 ** (a_val + b_val * (-1) * mag_bins)  # a - b*M
        fig.add_trace(go.Scatter(
            x=mag_bins, y=fit_y,
            mode="lines",
            name=f"G-R fit (b={b_val:.2f})",
            line=dict(color="#f59e0b", width=1.8, dash="dash"),
        ))
        fig.add_annotation(
            x=float(mag_bins[-1]), y=np.log10(max(1, min(cumul[-1] * 5, cumul[0]))),
            text=f"<b>b = {b_val:.2f}</b>",
            showarrow=False, font=dict(size=14, color="#f59e0b"),
            xanchor="right",
        )
    fig.update_yaxes(type="log", title="N cumulativo (log)")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Relazione Gutenberg-Richter — stima valore b",
        height=380,
        xaxis_title="Magnitudo",
    )
    return fig


def plot_interevent_times(events: pd.DataFrame) -> go.Figure:
    """Histogram of inter-event times (in minutes).

    Short inter-event times (< 5 min) clustered together are a hallmark of
    seismic swarms; a Poissonian/exponential distribution indicates background
    tectonic activity. Both patterns inform the volcanic hazard assessment.
    """
    ev = events.dropna(subset=["time"]).copy()
    ev["time"] = pd.to_datetime(ev["time"], errors="coerce").dropna()
    ev = ev.sort_values("time")
    if len(ev) < 4:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="Tempi inter-evento: dati insufficienti")
        return fig
    dt_min = ev["time"].diff().dt.total_seconds().dropna() / 60.0
    dt_min = dt_min[dt_min > 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dt_min.clip(upper=float(dt_min.quantile(0.97))),
        nbinsx=50,
        name="Inter-evento (min)",
        marker_color="rgba(96,165,250,0.70)",
        marker_line=dict(width=0.5, color="#1e3a5f"),
    ))
    median_dt = float(dt_min.median())
    fig.add_vline(x=median_dt, line_dash="dash", line_color="#fbbf24",
                  annotation_text=f"Mediana {median_dt:.1f} min", annotation_font_size=11)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Distribuzione tempi inter-evento sismico",
        height=340,
        xaxis_title="Tempo inter-evento (min)",
        yaxis_title="Conteggio",
    )
    return fig


st.set_page_config(page_title=APP_TITLE, page_icon="🌋", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("🌋 GeoMagVolcano")
    st.caption("Monitoraggio magnetometrico, sismico, geochimico, deformativo e space-weather")
    st.divider()
    st.subheader("📍 Vulcano / stazione")
    lat = st.number_input("Latitudine", value=LAT_DEFAULT, format="%.5f")
    lon = st.number_input("Longitudine", value=LON_DEFAULT, format="%.5f")
    alt = st.number_input("Quota stazione (m)", value=ALT_DEFAULT, format="%.1f")
    igrf_f = st.number_input("IGRF F locale (nT)", value=IGRF_F_DEFAULT, step=10.0)
    st.caption(f"D={IGRF_D_DEFAULT}°, I={IGRF_I_DEFAULT}°, quota={alt:.0f} m")

    st.divider()
    st.subheader("📅 Periodo analisi")
    default_start = date.today() - timedelta(days=30)
    start_date = st.date_input("Da", value=default_start)
    end_date = st.date_input("A", value=date.today())

    st.divider()
    st.subheader("🔑 API keys")
    nasa_key = st.text_input(
        "NASA DONKI API key",
        value=NASA_DEMO_KEY,
        type="password",
        help="Chiave NASA DONKI preconfigurata; puoi modificarla dalla sidebar se necessario."
    )

    st.divider()
    st.subheader("⚙️ Processing")
    unit_mag = st.selectbox("Unita' magnetometro", ["nT", "mG", "Gauss"], index=0)
    resampling = st.selectbox("Ricampionamento", ["Nessuno", "1min", "5min", "10min", "30min", "1h"], index=2)
    spike_sigma = st.slider("Hampel spike threshold (σ robusta)", 2.0, 8.0, 4.0, 0.5)
    sq_kp_threshold = st.slider("Kp massimo per ore quiete Sq", 0.5, 4.0, 2.0, 0.5)
    rolling_hours = st.slider("Finestra anomalia rolling (ore)", 6, 168, 24, 6)

    st.divider()
    st.subheader("🌐 Sismologia online")
    fdsn_label = st.selectbox("Catalogo FDSN", list(FDSN_ENDPOINTS.keys()), index=1)
    max_radius_km = st.slider("Raggio eventi (km)", 5, 500, 80, 5)
    min_mag = st.slider("Magnitudo minima", -1.0, 5.0, 0.0, 0.1)
    max_events = st.slider("Numero massimo eventi", 50, 5000, 1000, 50)

    st.divider()
    st.subheader("🌙 Filtro orario notturno")
    st.caption("Ora UTC. Le ore notturne hanno meno interferenza ionosferica Sq.")
    night_start_h = st.slider("Inizio notte (ora UTC)", 18, 23, 21, 1, key="night_start")
    night_end_h = st.slider("Fine notte (ora UTC)", 1, 8, 6, 1, key="night_end")
    night_label = f"{night_start_h:02d}:00 – {night_end_h:02d}:00 UTC"
    st.caption(f"Finestra notturna: {night_label}")

    st.divider()
    if st.button("🔄 Pulisci cache e aggiorna"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("⏱️ Auto-refresh")
    auto_refresh = st.checkbox("Attiva auto-refresh pagina", value=False)
    refresh_secs = st.selectbox("Intervallo (s)", [30, 60, 120, 300, 600], index=2) if auto_refresh else None
    if auto_refresh and refresh_secs:
        st.caption(f"La pagina si ricarica ogni {refresh_secs} s.")
        # JavaScript-based page reload
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}}, {refresh_secs * 1000});</script>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🚨 Rilevamento sciame")
    swarm_window_h = st.slider("Finestra rilevamento sciame (ore)", 1, 24, 6, 1)
    swarm_min_ev = st.slider("Soglia sciame (N eventi min.)", 3, 20, 5, 1)


# Fetch live external data early because multiple tabs use it.
with st.spinner("Scaricamento dati NOAA/GFZ e preparazione dashboard..."):
    noaa = parse_noaa_live()

kp_now = latest_numeric(noaa["kp1m"], "Kp", latest_numeric(noaa["kp3h"], "Kp", 0.0))
bz_now = latest_numeric(noaa["mag"], "bz_gsm", np.nan)
bt_now = latest_numeric(noaa["mag"], "bt", np.nan)
vp_now = latest_numeric(noaa["plasma"], "speed", np.nan)
np_now = latest_numeric(noaa["plasma"], "density", np.nan)

dst_now = np.nan
if not noaa["dst"].empty:
    dst_col = "dst" if "dst" in noaa["dst"].columns else noaa["dst"].columns[-1]
    dst_now = latest_numeric(noaa["dst"], dst_col, np.nan)

space_quality, space_flags = compute_space_weather_quality(kp_now, dst_now, bz_now, vp_now)

st.title(APP_TITLE)
st.markdown(
    "<div class='info'><b>Obiettivo:</b> separare il segnale vulcanomagnetico locale "
    "dalla variabilita' esterna del campo geomagnetico e confrontarlo con sismicita', "
    "geochimica, idrologia, deformazione, meteo e attivita' solare.</div>",
    unsafe_allow_html=True,
)
st.write("")

cols = st.columns(7)
summary_values = [
    ("Kp live", f"{kp_now:.1f}" if np.isfinite(kp_now) else "N/A"),
    ("Bz GSM", f"{bz_now:.1f} nT" if np.isfinite(bz_now) else "N/A"),
    ("Bt IMF", f"{bt_now:.1f} nT" if np.isfinite(bt_now) else "N/A"),
    ("Vento solare", f"{vp_now:.0f} km/s" if np.isfinite(vp_now) else "N/A"),
    ("Densita'", f"{np_now:.1f} p/cm3" if np.isfinite(np_now) else "N/A"),
    ("F10.7", f"{noaa.get('f107_value', 'N/A')} sfu"),
    ("Qualita' mag", f"{space_quality * 100:.0f}%"),
]
for col, (label, value) in zip(cols, summary_values):
    with col:
        color = kp_color(kp_now) if label == "Kp live" else "#e5e7eb"
        st.markdown(
            f"<div class='mbox'><div class='mlbl'>{label}</div>"
            f"<div class='mval' style='color:{color}'>{value}</div></div>",
            unsafe_allow_html=True,
        )

if space_flags:
    st.markdown(
        "<div class='yellow'><b>Attenzione space-weather:</b> "
        + "; ".join(space_flags)
        + ". Le anomalie magnetiche locali devono essere interpretate con cautela.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='green'><b>Condizioni geomagnetiche favorevoli:</b> "
        "il confronto vulcanomagnetico e' meno contaminato da sorgenti esterne.</div>",
        unsafe_allow_html=True,
    )


tab_mag, tab_sism, tab_aux, tab_space, tab_nasa, tab_proc, tab_state, tab_tools = st.tabs(
    [
        "🧲 Magnetometro",
        "🌎 Sismicita'",
        "📊 Dati multiparametrici",
        "☀️ Campo terrestre & Sole",
        "🛸 NASA DONKI",
        "🔬 Processing",
        "🚦 Stato vulcanico",
        "🧰 Strumenti e integrazioni",
    ]
)


# -----------------------------------------------------------------------------
# Magnetometer tab
# -----------------------------------------------------------------------------
with tab_mag:
    st.header("🧲 Magnetometria locale - APS 1540 / CSV compatibile")
    st.markdown(
        "<div class='info'>Carica un file con F oppure X/Y/Z. Se sono disponibili X/Y/Z, "
        "F viene calcolato come sqrt(X²+Y²+Z²). Gli spike isolati vengono trattati con filtro Hampel. "
        "La scala Y di ogni pannello è adattata automaticamente al min/max del segnale. "
        "Usa il pannello <b>Dati esterni</b> per sovrapporre qualsiasi CSV su asse Y secondario.</div>",
        unsafe_allow_html=True,
    )

    # Palette per tracce esterne (ciclica)
    _EXT_PALETTE = ["#f59e0b", "#c084fc", "#fb923c", "#34d399", "#f87171", "#38bdf8"]

    mag_file = st.file_uploader("Carica CSV magnetometrico", type=["csv", "txt"], key="mag_file")
    if mag_file is not None:
        df_mag_raw = read_uploaded_table(mag_file)
        try:
            df_mag_local = parse_magnetometer(df_mag_raw, unit_mag, spike_sigma)
            if resampling != "Nessuno":
                df_mag_local = df_mag_local.resample(resampling).mean(numeric_only=True)
            df_mag_local["dF_IGRF"] = df_mag_local["F"] - igrf_f
            st.session_state["mag"] = df_mag_local
            st.success(f"Caricati {len(df_mag_local):,} campioni: {df_mag_local.index.min()} → {df_mag_local.index.max()}")

            view = st.radio(
                "Vista",
                ["F totale", "Componenti", "ΔF IGRF", "Qualita' e gap"],
                horizontal=True,
            )

            # ── Dati esterni per confronto ─────────────────────────────────────
            # Il pannello è unico; i widget interni cambiano per "Componenti"
            # (associazione per colonna) o per le altre viste (selezione multipla).
            with st.expander("📂 Dati esterni per confronto — asse Y secondario", expanded=False):
                st.markdown(
                    "Carica un CSV esterno con colonna temporale + una o più colonne numeriche. "
                    "Ogni segnale esterno viene sovrapposto al grafico selezionato su **asse Y secondario** "
                    "con scala automatica adattata ai limiti del segnale esterno stesso."
                )
                ext_file = st.file_uploader("CSV esterno", type=["csv", "txt"], key=f"ext_mag_{view}")
                ext_df: pd.DataFrame = pd.DataFrame()
                ext_num_cols: List[str] = []
                if ext_file is not None:
                    try:
                        _ext_raw = read_uploaded_table(ext_file)
                        ext_df = prepare_time_index(_ext_raw)
                        ext_num_cols = ext_df.select_dtypes(include=[np.number]).columns.tolist()
                        if ext_num_cols:
                            st.success(
                                f"File esterno: {len(ext_df):,} righe — "
                                f"colonne numeriche: {', '.join(ext_num_cols)}"
                            )
                        else:
                            st.warning("Nessuna colonna numerica nel file esterno.")
                    except Exception as _exc_ext:
                        st.warning(f"File esterno non leggibile: {_exc_ext}")

            # ── Vista: F totale ────────────────────────────────────────────────
            if view == "F totale":
                ext_sel_f: List[str] = []
                if ext_num_cols:
                    ext_sel_f = st.multiselect(
                        "Colonne esterne → F totale",
                        ext_num_cols,
                        default=ext_num_cols[:1],
                        key="ext_f_cols",
                    )
                _has_ext = bool(ext_sel_f) and not ext_df.empty

                fig = make_subplots(specs=[[{"secondary_y": _has_ext}]])
                fig.add_trace(
                    go.Scatter(
                        x=df_mag_local.index, y=df_mag_local["F"],
                        name="F locale", line=dict(width=1.2, color="#60a5fa"),
                    ),
                    secondary_y=False,
                )
                fig.add_hline(y=igrf_f, line_dash="dot", line_color="#93a4b7", annotation_text="IGRF locale")
                _y0, _y1 = finite_min_max([df_mag_local["F"]])
                fig.update_yaxes(range=[_y0, _y1], secondary_y=False, title_text="F (nT)", color="#60a5fa")
                if _has_ext:
                    for _ci, _ec in enumerate(ext_sel_f):
                        fig.add_trace(
                            go.Scatter(
                                x=ext_df.index, y=ext_df[_ec],
                                name=f"Ext: {_ec}",
                                line=dict(width=1.3, dash="dot", color=_EXT_PALETTE[_ci % len(_EXT_PALETTE)]),
                                opacity=0.90,
                            ),
                            secondary_y=True,
                        )
                    _all_ext_f = pd.concat([ext_df[c] for c in ext_sel_f])
                    _ey0, _ey1 = finite_min_max([_all_ext_f])
                    fig.update_yaxes(
                        range=[_ey0, _ey1], secondary_y=True,
                        title_text="Ext", showgrid=False,
                        color=_EXT_PALETTE[0],
                    )
                fig.update_layout(**PLOTLY_LAYOUT, title="Campo totale F", height=430)
                st.plotly_chart(fig, use_container_width=True)

            # ── Vista: Componenti ──────────────────────────────────────────────
            elif view == "Componenti":
                comps = [c for c in ["X", "Y", "Z"] if c in df_mag_local.columns]
                if comps:
                    # Associazione colonna esterna per ogni componente
                    ext_comp_map: Dict[str, str] = {}
                    if ext_num_cols:
                        st.markdown("**Associa colonna esterna a ciascun componente** (opzionale):")
                        _pickers = st.columns(len(comps))
                        for _ci, _comp in enumerate(comps):
                            with _pickers[_ci]:
                                _sel = st.selectbox(
                                    f"Ext → {_comp}", ["—"] + ext_num_cols, key=f"ext_comp_{_comp}"
                                )
                                if _sel != "—":
                                    ext_comp_map[_comp] = _sel

                    COMP_COLORS = {"X": "#f87171", "Y": "#4ade80", "Z": "#60a5fa"}
                    _specs = [[{"secondary_y": _comp in ext_comp_map}] for _comp in comps]

                    fig = make_subplots(
                        rows=len(comps), cols=1, shared_xaxes=True,
                        subplot_titles=comps, specs=_specs,
                        vertical_spacing=0.06,
                    )
                    for _i, _comp in enumerate(comps, start=1):
                        _s = df_mag_local[_comp]
                        _color = COMP_COLORS.get(_comp, "#e2e8f0")
                        fig.add_trace(
                            go.Scatter(
                                x=_s.index, y=_s,
                                name=_comp, line=dict(width=1.1, color=_color),
                            ),
                            row=_i, col=1, secondary_y=False,
                        )
                        # Scala Y automatica per ogni singolo componente
                        _cy0, _cy1 = finite_min_max([_s])
                        fig.update_yaxes(
                            range=[_cy0, _cy1], row=_i, col=1,
                            secondary_y=False,
                            title_text=f"{_comp} (nT)",
                            title_font=dict(color=_color, size=11),
                            tickfont=dict(color=_color),
                        )
                        # Segnale esterno sovrapposto su asse Y secondario
                        if _comp in ext_comp_map:
                            _ecol = ext_comp_map[_comp]
                            _ext_s = ext_df[_ecol]
                            _ext_c = _EXT_PALETTE[(_i - 1) % len(_EXT_PALETTE)]
                            _ey0, _ey1 = finite_min_max([_ext_s])
                            fig.add_trace(
                                go.Scatter(
                                    x=_ext_s.index, y=_ext_s,
                                    name=f"Ext {_ecol} → {_comp}",
                                    line=dict(width=1.2, dash="dot", color=_ext_c),
                                    opacity=0.88,
                                ),
                                row=_i, col=1, secondary_y=True,
                            )
                            fig.update_yaxes(
                                range=[_ey0, _ey1], row=_i, col=1,
                                secondary_y=True, showgrid=False,
                                title_text=_ecol,
                                title_font=dict(color=_ext_c, size=10),
                                tickfont=dict(color=_ext_c),
                            )

                    fig.update_layout(
                        **PLOTLY_LAYOUT,
                        height=280 * len(comps),
                        title="Componenti magnetiche X/Y/Z — scala Y automatica per pannello",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Il file contiene solo F; componenti non disponibili.")

            # ── Vista: ΔF IGRF ─────────────────────────────────────────────────
            elif view == "ΔF IGRF":
                ext_sel_df: List[str] = []
                if ext_num_cols:
                    ext_sel_df = st.multiselect(
                        "Colonne esterne → ΔF IGRF",
                        ext_num_cols,
                        default=ext_num_cols[:1],
                        key="ext_df_cols",
                    )
                _has_ext_df = bool(ext_sel_df) and not ext_df.empty

                fig = make_subplots(specs=[[{"secondary_y": _has_ext_df}]])
                filled_anomaly_traces(fig, df_mag_local["dF_IGRF"], "F–IGRF")
                add_sigma_lines_enhanced(fig, df_mag_local["dF_IGRF"])
                _dy0, _dy1 = finite_min_max([df_mag_local["dF_IGRF"]])
                fig.update_yaxes(range=[_dy0, _dy1], secondary_y=False, title_text="ΔF (nT)")
                if _has_ext_df:
                    _all_ext_df = pd.concat([ext_df[c] for c in ext_sel_df])
                    _dey0, _dey1 = finite_min_max([_all_ext_df])
                    for _ci, _ec in enumerate(ext_sel_df):
                        fig.add_trace(
                            go.Scatter(
                                x=ext_df.index, y=ext_df[_ec],
                                name=f"Ext: {_ec}",
                                line=dict(width=1.3, dash="dot", color=_EXT_PALETTE[_ci % len(_EXT_PALETTE)]),
                                opacity=0.90,
                            ),
                            secondary_y=True,
                        )
                    fig.update_yaxes(
                        range=[_dey0, _dey1], secondary_y=True,
                        title_text="Ext", showgrid=False,
                        color=_EXT_PALETTE[0],
                    )
                fig.update_layout(**PLOTLY_LAYOUT, title="Residuo rispetto a IGRF", height=430)
                st.plotly_chart(fig, use_container_width=True)

            # ── Vista: Qualita' e gap ──────────────────────────────────────────
            else:
                dt_seconds = infer_sampling_seconds(df_mag_local.index)
                full_index = pd.date_range(df_mag_local.index.min(), df_mag_local.index.max(), freq=f"{int(max(dt_seconds, 1))}s")
                missing_pct = 100.0 * (1.0 - len(df_mag_local.index.intersection(full_index)) / max(len(full_index), 1))
                st.metric("Campionamento mediano", f"{dt_seconds:.1f} s")
                st.metric("Gap stimati", f"{missing_pct:.2f}%")
                st.dataframe(df_mag_local.describe().round(3), use_container_width=True)

            st.download_button(
                "📥 Scarica magnetometria processata",
                df_mag_local.to_csv().encode("utf-8"),
                file_name="magnetometer_processed_v3.csv",
                mime="text/csv",
            )

            # ── GFZ overlay sul magnetometro locale ────────────────────────────
            st.divider()
            st.subheader("🏛️ Sovrapposizione GFZ Potsdam sul magnetometro locale")
            st.markdown(
                "<div class='info'>Sovrapponi un indice GFZ (Kp, ap, Hp30 …) al segnale locale su asse Y secondario. "
                "Questo confronto diretto aiuta a verificare se un'anomalia locale segue fedelmente l'indice globale "
                "(→ origine esterna) oppure si discosta da esso (→ possibile contributo vulcanomagnetico).</div>",
                unsafe_allow_html=True,
            )
            gfz_overlay_idx = st.selectbox(
                "Indice GFZ da sovrapporre",
                ["Nessuno", "Kp", "ap", "Hp30", "Hp60", "ap30", "ap60"],
                index=1,
                key="mag_gfz_overlay_idx",
            )
            mag_channel_overlay = st.selectbox(
                "Canale magnetometro locale",
                [c for c in ["F", "dF_IGRF", "X", "Y", "Z"] if c in df_mag_local.columns],
                index=0,
                key="mag_gfz_channel",
            )
            if gfz_overlay_idx != "Nessuno":
                start_gfz_ov = f"{start_date.isoformat()}T00:00:00Z"
                end_gfz_ov = f"{end_date.isoformat()}T23:59:59Z"
                df_gfz_ov = fetch_gfz_index(start_gfz_ov, end_gfz_ov, gfz_overlay_idx)
                if not df_gfz_ov.empty and gfz_overlay_idx in df_gfz_ov.columns:
                    fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
                    # Canale locale
                    fig_ov.add_trace(
                        go.Scatter(
                            x=df_mag_local.index,
                            y=df_mag_local[mag_channel_overlay],
                            name=mag_channel_overlay,
                            line=dict(color="#60a5fa", width=1.4),
                        ),
                        secondary_y=False,
                    )
                    # Highlight ore notturne
                    night_series_ov = filter_nighttime(df_mag_local[mag_channel_overlay], night_start_h, night_end_h)
                    fig_ov.add_trace(
                        go.Scatter(
                            x=night_series_ov.index,
                            y=night_series_ov,
                            mode="markers",
                            marker=dict(color="#c084fc", size=3, opacity=0.7),
                            name=f"Notte ({night_label})",
                        ),
                        secondary_y=False,
                    )
                    # Indice GFZ
                    fig_ov.add_trace(
                        go.Scatter(
                            x=df_gfz_ov["time"],
                            y=df_gfz_ov[gfz_overlay_idx],
                            line=dict(color="#f59e0b", width=1.6, dash="dot"),
                            name=f"GFZ {gfz_overlay_idx}",
                            opacity=0.9,
                        ),
                        secondary_y=True,
                    )
                    # Scale Y automatiche su dati reali
                    _lov_y0, _lov_y1 = finite_min_max([df_mag_local[mag_channel_overlay]])
                    _gov_y0, _gov_y1 = finite_min_max([df_gfz_ov[gfz_overlay_idx]])
                    fig_ov.update_yaxes(
                        range=[_lov_y0, _lov_y1],
                        title_text=f"{mag_channel_overlay} (nT)",
                        secondary_y=False, color="#60a5fa",
                    )
                    fig_ov.update_yaxes(
                        range=[_gov_y0, _gov_y1],
                        title_text=f"GFZ {gfz_overlay_idx}",
                        secondary_y=True, color="#f59e0b",
                        showgrid=False,
                    )
                    fig_ov.update_layout(
                        **PLOTLY_LAYOUT,
                        height=450,
                        title=f"Magnetometro locale {mag_channel_overlay} vs GFZ {gfz_overlay_idx}",
                    )
                    st.plotly_chart(fig_ov, use_container_width=True)
                else:
                    st.info(f"Dati GFZ {gfz_overlay_idx} non disponibili per il periodo selezionato.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Parsing magnetometro fallito: {exc}")
    else:
        st.info("Carica un CSV magnetometrico per attivare processing e stato vulcanico.")


# -----------------------------------------------------------------------------
# Seismic tab
# -----------------------------------------------------------------------------
with tab_sism:
    st.header("🌎 Sismicita' - cataloghi online FDSN e CSV locale")
    st.markdown(
        "<div class='info'>La sismicita' viene convertita in rate giornaliero, magnitudo massima "
        "ed energia radiata approssimata. Per aree vulcaniche, sciami di bassa magnitudo e migrazione "
        "temporale/spaziale sono spesso piu' informativi del singolo evento maggiore.</div>",
        unsafe_allow_html=True,
    )

    start_iso = f"{start_date.isoformat()}T00:00:00"
    end_iso = f"{end_date.isoformat()}T23:59:59"
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🌐 Scarica eventi FDSN", type="primary"):
            endpoint = FDSN_ENDPOINTS[fdsn_label]
            events = fetch_fdsn_events(endpoint, start_iso, end_iso, lat, lon, max_radius_km, min_mag, max_events)
            st.session_state["seismic"] = events
    with c2:
        seis_file = st.file_uploader("Oppure carica CSV sismico", type=["csv", "txt"], key="seis_file")
        if seis_file is not None:
            raw = read_uploaded_table(seis_file)
            st.session_state["seismic"] = parse_seismic_csv(raw, lat, lon)

    events = st.session_state.get("seismic", pd.DataFrame())
    if not events.empty:
        # ── Stat summary row ──────────────────────────────────────────────────
        n_tot = len(events)
        mmax_all = float(events["magnitude"].max()) if "magnitude" in events.columns else np.nan
        n_near = int((events["distance_km"] <= 20).sum()) if "distance_km" in events.columns else 0
        last_t = pd.to_datetime(events["time"].max(), errors="coerce") if "time" in events.columns else pd.NaT
        last_str = last_t.strftime("%d/%m %H:%M") if pd.notna(last_t) else "—"
        avg_depth = float(events["depth_km"].median()) if "depth_km" in events.columns else np.nan
        avg_depth_str = f"{avg_depth:.1f} km" if np.isfinite(avg_depth) else "—"
        mmax_str = f"{mmax_all:.1f}" if np.isfinite(mmax_all) else "—"
        stat_html = (
            "<div class='stat-row'>"
            f"<div class='stat-card'><div class='sl'>Tot. eventi</div><div class='sv' style='color:#60a5fa'>{n_tot:,}</div></div>"
            f"<div class='stat-card'><div class='sl'>Entro 20 km</div><div class='sv' style='color:#34d399'>{n_near:,}</div></div>"
            f"<div class='stat-card'><div class='sl'>Mmax</div><div class='sv' style='color:#f87171'>{mmax_str}</div></div>"
            f"<div class='stat-card'><div class='sl'>Profondità mediana</div><div class='sv' style='color:#c084fc'>{avg_depth_str}</div></div>"
            f"<div class='stat-card'><div class='sl'>Ultimo evento</div><div class='sv' style='color:#fbbf24;font-size:16px'>{last_str}</div></div>"
            "</div>"
        )
        st.markdown(stat_html, unsafe_allow_html=True)

        # ── Swarm detection ────────────────────────────────────────────────────
        swarm_msg = detect_seismic_swarm(events, window_hours=swarm_window_h, min_events=swarm_min_ev)
        if swarm_msg:
            st.markdown(f"<div class='swarm-alert'>🔴 {swarm_msg}</div>", unsafe_allow_html=True)
            st.write("")

        # ── Recent events flash section ────────────────────────────────────────
        st.subheader("⚡ Ultimi 5 eventi più recenti")
        render_recent_events_flash(events, n=5)
        st.write("")

        # ── Map ────────────────────────────────────────────────────────────────
        st.plotly_chart(
            plot_event_map(
                events,
                "Mappa sismica delle Isole Eolie — colore = profondità, dimensione = magnitudo",
                lat, lon,
            ),
            use_container_width=True,
        )
        # Trigger Plotly animation autoplay.
        # Plotly in Streamlit renders inside an iframe; the script targets the
        # innermost .js-plotly-plot element and calls Plotly.animate on it so
        # the pulse-ring frames loop immediately without any user interaction.
        st.markdown("""
<script>
(function startSeismicAnimation() {
    var attempt = 0;
    function tryPlay() {
        attempt++;
        var plots = parent.document.querySelectorAll('.js-plotly-plot');
        // Pick the most recently added plot (the map is the first in the tab)
        if (plots && plots.length > 0) {
            var mapEl = null;
            for (var i = 0; i < plots.length; i++) {
                if (plots[i]._fullLayout && plots[i]._fullLayout.mapbox) {
                    mapEl = plots[i]; break;
                }
            }
            if (mapEl && mapEl.data) {
                try {
                    Plotly.animate(mapEl, null, {
                        frame: {duration: 350, redraw: true},
                        transition: {duration: 0},
                        mode: 'immediate',
                        loop: true
                    });
                    return;
                } catch(e) {}
            }
        }
        if (attempt < 30) setTimeout(tryPlay, 400);
    }
    setTimeout(tryPlay, 800);
})();
</script>
""", unsafe_allow_html=True)

        metrics = compute_daily_seismic_metrics(events)
        st.session_state["seismic_metrics"] = metrics
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=["Numero eventi / giorno", "Energia sismica / giorno", "Magnitudo massima / giorno"])
        fig.add_trace(go.Bar(x=metrics.index, y=metrics["count"], name="N eventi"), row=1, col=1)
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics["energy_j"], name="Energia J", line=dict(width=1.3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=metrics.index, y=metrics["max_mag"], name="Mmax", line=dict(width=1.3)), row=3, col=1)
        y_min_count, y_max_count = y_range_controls("seis_count", "eventi/giorno", [metrics["count"]])
        apply_y_range(fig, y_min_count, y_max_count, row=1)
        energy_positive = metrics["energy_j"].replace(0, np.nan).dropna()
        y_min_en, y_max_en = y_range_controls("seis_energy", "energia sismica/giorno", [energy_positive if not energy_positive.empty else metrics["energy_j"]])
        if y_min_en <= 0:
            y_min_en = max(float(energy_positive.min()) if not energy_positive.empty else 1.0, 1e-6)
        apply_y_range(fig, y_min_en, y_max_en, row=2)
        y_min_mag, y_max_mag = y_range_controls("seis_mmax", "magnitudo massima/giorno", [metrics["max_mag"]])
        apply_y_range(fig, y_min_mag, y_max_mag, row=3)
        fig.update_yaxes(type="log", row=2, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=750, title="Metriche sismiche giornaliere")
        st.plotly_chart(fig, use_container_width=True)

        # ── Advanced analysis ──────────────────────────────────────────────────
        with st.expander("🔬 Analisi sismica avanzata", expanded=False):
            st.markdown(
                "<div class='info'>"
                "Questi grafici aggiuntivi aiutano a distinguere sismicità di tipo sciame vulcanico "
                "(b>1.2, inter-eventi brevi e raggruppati, migrazione verso l'alto) da attività tettonica "
                "di fondo (distribuzione esponenziale dei tempi inter-evento, b≈1)."
                "</div>",
                unsafe_allow_html=True,
            )
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                st.plotly_chart(plot_seismic_magnitude_frequency(events), use_container_width=True)
            with adv_c2:
                st.plotly_chart(plot_interevent_times(events), use_container_width=True)
            st.plotly_chart(plot_seismic_depth_timeline(events), use_container_width=True)

            # Depth histogram
            if "depth_km" in events.columns:
                fig_dh = go.Figure(go.Histogram(
                    x=events["depth_km"].dropna(),
                    nbinsx=30,
                    name="Profondità",
                    marker_color="rgba(192,132,252,0.65)",
                    marker_line=dict(width=0.5, color="#4c1d95"),
                ))
                fig_dh.update_layout(
                    **PLOTLY_LAYOUT, height=300,
                    title="Distribuzione profondità ipocentrale",
                    xaxis_title="Profondità (km)", yaxis_title="N eventi",
                )
                st.plotly_chart(fig_dh, use_container_width=True)

        # ── Distance-magnitude scatter ─────────────────────────────────────────
        if "distance_km" in events.columns and "magnitude" in events.columns:
            with st.expander("📐 Scatter distanza–magnitudo", expanded=False):
                fig_dm = go.Figure(go.Scatter(
                    x=events["distance_km"],
                    y=events["magnitude"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=events.get("depth_km", pd.Series(np.nan, index=events.index)),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Prof. km", thickness=12, len=0.5),
                        opacity=0.75,
                    ),
                    hovertemplate="Distanza: %{x:.1f} km<br>Mag: %{y:.1f}<extra></extra>",
                ))
                fig_dm.add_vline(x=20, line_dash="dash", line_color="#f87171",
                                 annotation_text="20 km (prossimità Vulcano)")
                fig_dm.update_layout(
                    **PLOTLY_LAYOUT, height=360,
                    title="Scatter distanza vs magnitudo (colore = profondità)",
                    xaxis_title="Distanza da Vulcano (km)",
                    yaxis_title="Magnitudo",
                )
                st.plotly_chart(fig_dm, use_container_width=True)

        # ── Catalogue table and download ───────────────────────────────────────
        show_cols = [
            col for col in [
                "time", "magnitude", "depth_km", "distance_km",
                "latitude", "longitude", "place", "source", "event_id"
            ]
            if col in events.columns
        ]
        n_show = st.slider("Ultimi N eventi da mostrare in tabella", 10, 500, 50, 10, key="seis_table_n")
        st.dataframe(events[show_cols].tail(n_show), use_container_width=True)
        st.download_button("📥 Scarica eventi sismici", events.to_csv(index=False).encode("utf-8"), "seismic_events_v3.csv", "text/csv")
    else:
        st.info("Scarica eventi FDSN o carica un catalogo CSV per abilitare il confronto sismico.")


# -----------------------------------------------------------------------------
# Aux multiparametric tab
# -----------------------------------------------------------------------------
with tab_aux:
    st.header("📊 Dati multiparametrici: geochimica, deformazione, meteo, idrologia")
    st.markdown(
        "<div class='info'>Puoi caricare CSV con una colonna temporale e variabili numeriche: CO2, SO2, H2S, "
        "temperatura fumarole, livello pozzo, pressione, tilt, GPS, strain, conduttivita'. Ogni variabile viene "
        "normalizzata con z-score robusto per il confronto temporale.</div>",
        unsafe_allow_html=True,
    )

    aux_file = st.file_uploader("Carica CSV multiparametrico", type=["csv", "txt"], key="aux_file")
    decimal_choice = st.selectbox("Separatore decimale CSV ausiliario", [".", ","], index=0)
    if aux_file is not None:
        raw_aux = read_uploaded_table(aux_file, decimal=decimal_choice)
        try:
            aux = parse_multiparameter_csv(raw_aux)
            st.session_state["aux"] = aux
            st.success(f"Variabili caricate: {', '.join(aux.columns)}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Parsing dati multiparametrici fallito: {exc}")

    aux = st.session_state.get("aux", pd.DataFrame())
    if not aux.empty:
        variables = st.multiselect("Variabili da visualizzare", aux.columns.tolist(), default=aux.columns.tolist()[: min(5, len(aux.columns))])
        if variables:
            fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=variables)
            for i, var in enumerate(variables, start=1):
                fig.add_trace(go.Scatter(x=aux.index, y=aux[var], name=var, line=dict(width=1.3)), row=i, col=1)
                add_sigma_lines(fig, aux[var], row=i, col=1)
            y_min, y_max = y_range_controls("aux_series", "serie multiparametriche", [aux[var] for var in variables])
            for row_i in range(1, len(variables) + 1):
                apply_y_range(fig, y_min, y_max, row=row_i)
            fig.update_layout(**PLOTLY_LAYOUT, height=230 * len(variables), title="Serie multiparametriche")
            st.plotly_chart(fig, use_container_width=True)

            fig_z = go.Figure()
            for var in variables:
                fig_z.add_trace(go.Scatter(x=aux.index, y=robust_zscore(aux[var]), name=f"{var} z", line=dict(width=1.2)))
            z_series = [robust_zscore(aux[var]) for var in variables]
            y_min, y_max = y_range_controls("aux_z", "z-score multiparametrico", z_series)
            apply_y_range(fig_z, y_min, y_max)
            fig_z.update_layout(**PLOTLY_LAYOUT, title="Confronto normalizzato robusto", yaxis_title="robust z")
            st.plotly_chart(fig_z, use_container_width=True)

    st.divider()
    st.subheader("🌤️ Meteo online Open-Meteo")
    meteo = fetch_open_meteo(lat, lon)
    if not meteo.empty:
        st.session_state["meteo"] = meteo
        vars_m = [c for c in meteo.columns if c != "time"]
        selected_m = st.multiselect("Variabili meteo", vars_m, default=vars_m[:3])
        if selected_m:
            st.plotly_chart(make_multi_axis_timeseries(meteo, "time", selected_m, "Meteo locale / forecast"), use_container_width=True)
    else:
        st.info("Meteo non disponibile in questo momento.")


# -----------------------------------------------------------------------------
# Space weather tab
# -----------------------------------------------------------------------------
with tab_space:
    st.header("☀️ Campo magnetico terrestre, indici geomagnetici e attivita' solare")
    st.markdown(
        "<div class='info'>Questa sezione serve a distinguere anomalie locali da sorgenti esterne: "
        "Kp/Hp30/ap descrivono la variabilita' geomagnetica globale/sub-oraria, Dst l'intensita' della corrente ad anello, "
        "Bz GSM e vento solare indicano l'accoppiamento con la magnetosfera.</div>",
        unsafe_allow_html=True,
    )

    if not noaa["kp1m"].empty and "Kp" in noaa["kp1m"].columns:
        kp_df = noaa["kp1m"].copy()
        kp_df["Kp"] = pd.to_numeric(kp_df["Kp"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=kp_df["time_tag"], y=kp_df["Kp"], name="Kp 1-min", marker_color=[kp_color(float(k)) for k in kp_df["Kp"].fillna(0)]))
        fig.add_hline(y=5, line_dash="dash", line_color="#f87171", annotation_text="Kp=5 G1")
        fig.add_hline(y=3, line_dash="dot", line_color="#fbbf24", annotation_text="Kp=3 quiet threshold")
        y_min, y_max = y_range_controls("space_kp_live", "Kp live", [kp_df["Kp"]])
        fig.update_layout(**PLOTLY_LAYOUT, title="NOAA SWPC Kp live", yaxis_title="Kp")
        apply_y_range(fig, min(0.0, y_min), max(9.0, y_max))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if not noaa["mag"].empty and "bz_gsm" in noaa["mag"].columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=noaa["mag"]["time_tag"], y=noaa["mag"]["bz_gsm"], name="Bz GSM", line=dict(width=1.2)))
            if "bt" in noaa["mag"].columns:
                fig.add_trace(go.Scatter(x=noaa["mag"]["time_tag"], y=noaa["mag"]["bt"], name="Bt", line=dict(width=1.2)))
            fig.add_hline(y=-10, line_dash="dash", line_color="#f87171", annotation_text="Bz < -10 nT")
            imf_series = [noaa["mag"]["bz_gsm"]]
            if "bt" in noaa["mag"].columns:
                imf_series.append(noaa["mag"]["bt"])
            y_min, y_max = y_range_controls("space_imf", "IMF Bz/Bt", imf_series)
            apply_y_range(fig, y_min, y_max)
            fig.update_layout(**PLOTLY_LAYOUT, title="IMF DSCOVR/RTSW", yaxis_title="nT")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if not noaa["plasma"].empty and "speed" in noaa["plasma"].columns:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=noaa["plasma"]["time_tag"], y=noaa["plasma"]["speed"], name="Vp", line=dict(width=1.2)), secondary_y=False)
            if "density" in noaa["plasma"].columns:
                fig.add_trace(go.Scatter(x=noaa["plasma"]["time_tag"], y=noaa["plasma"]["density"], name="Np", line=dict(width=1.2)), secondary_y=True)
            y_min, y_max = y_range_controls("space_wind_speed", "velocita vento solare", [noaa["plasma"]["speed"]])
            fig.update_yaxes(range=[y_min, y_max], secondary_y=False)
            if "density" in noaa["plasma"].columns:
                d_min, d_max = finite_min_max([noaa["plasma"]["density"]])
                fig.update_yaxes(range=[d_min, d_max], title_text="p/cm3", secondary_y=True)
            fig.update_layout(**PLOTLY_LAYOUT, title="Vento solare", yaxis_title="km/s")
            st.plotly_chart(fig, use_container_width=True)

    if not noaa["xray"].empty and "flux" in noaa["xray"].columns:
        xray = noaa["xray"].copy()
        xray["flux"] = pd.to_numeric(xray["flux"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xray["time_tag"], y=xray["flux"], name="GOES X-ray", line=dict(width=1.2), fill="tozeroy"))
        for value, label in [(1e-7, "C"), (1e-6, "M"), (1e-5, "X")]:
            fig.add_hline(y=value, line_dash="dot", annotation_text=label)
        xray_positive = xray["flux"].replace(0, np.nan).dropna()
        y_min, y_max = y_range_controls("space_xray", "GOES X-ray", [xray_positive if not xray_positive.empty else xray["flux"]])
        if y_min <= 0:
            y_min = max(float(xray_positive.min()) if not xray_positive.empty else 1e-9, 1e-12)
        apply_y_range(fig, y_min, y_max)
        fig.update_layout(**PLOTLY_LAYOUT, title="GOES X-ray 0.1-0.8 nm", yaxis_type="log", yaxis_title="W/m2")
        st.plotly_chart(fig, use_container_width=True)

    if not noaa["dst"].empty:
        dst_df = noaa["dst"].copy()
        time_col = "time_tag" if "time_tag" in dst_df.columns else next((c for c in dst_df.columns if "time" in str(c).lower()), None)
        dst_col = "dst" if "dst" in dst_df.columns else next((c for c in dst_df.columns if c != time_col), None)
        if time_col is not None and dst_col is not None:
            dst_df[time_col] = pd.to_datetime(dst_df[time_col], errors="coerce")
            dst_df[dst_col] = pd.to_numeric(dst_df[dst_col], errors="coerce")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dst_df[time_col], y=dst_df[dst_col], name="Dst", line=dict(width=1.2), fill="tozeroy"))
            for value, label in [(-30, "debole"), (-50, "moderata"), (-100, "intensa")]:
                fig.add_hline(y=value, line_dash="dot", annotation_text=f"Dst {value} nT {label}")
            y_min, y_max = y_range_controls("space_dst", "Dst", [dst_df[dst_col]])
            apply_y_range(fig, y_min, y_max)
            fig.update_layout(**PLOTLY_LAYOUT, title="NOAA geospace Dst 1-hour", yaxis_title="Dst (nT)")
            st.plotly_chart(fig, use_container_width=True)

    if not noaa["kpfcst"].empty:
        kpf = noaa["kpfcst"].copy()
        time_col = "time_tag" if "time_tag" in kpf.columns else next((c for c in kpf.columns if "time" in str(c).lower()), None)
        kp_col = "kp" if "kp" in kpf.columns else ("Kp" if "Kp" in kpf.columns else None)
        if time_col is not None and kp_col is not None:
            kpf[time_col] = pd.to_datetime(kpf[time_col], errors="coerce")
            kpf[kp_col] = pd.to_numeric(kpf[kp_col], errors="coerce")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=kpf[time_col], y=kpf[kp_col], name="Kp observed/estimated/predicted"))
            fig.add_hline(y=5, line_dash="dash", annotation_text="G1 Kp=5")
            y_min, y_max = y_range_controls("space_kp_forecast", "Kp forecast", [kpf[kp_col]])
            fig.update_layout(**PLOTLY_LAYOUT, title="NOAA Kp observed / estimated / predicted", yaxis_title="Kp")
            apply_y_range(fig, min(0.0, y_min), max(9.0, y_max))
            st.plotly_chart(fig, use_container_width=True)


    st.divider()
    st.subheader("🧲 Magnetometro locale nella sezione campo terrestre")
    local_mag = st.session_state.get("mag", pd.DataFrame())
    if not local_mag.empty:
        local_vars = [c for c in ["X", "Y", "Z", "F", "dF_IGRF"] if c in local_mag.columns]
        selected_local = st.multiselect(
            "Canali magnetometro locale da confrontare con space-weather",
            local_vars,
            default=[c for c in ["X", "Y", "Z", "F"] if c in local_vars],
            key="space_local_mag_channels",
        )
        if selected_local:
            axis_vars = [c for c in selected_local if c in ["X", "Y", "Z"]]
            if axis_vars:
                fig_loc_axes = make_subplots(
                    rows=len(axis_vars), cols=1, shared_xaxes=True,
                    vertical_spacing=0.04, subplot_titles=[f"Asse {c}" for c in axis_vars]
                )
                for i, comp in enumerate(axis_vars, start=1):
                    fig_loc_axes.add_trace(
                        go.Scatter(x=local_mag.index, y=local_mag[comp], name=comp, line=dict(width=1.15)),
                        row=i, col=1,
                    )
                y_min, y_max = y_range_controls("space_local_axes", "assi magnetici X/Y/Z", [local_mag[c] for c in axis_vars])
                for row_i in range(1, len(axis_vars) + 1):
                    apply_y_range(fig_loc_axes, y_min, y_max, row=row_i)
                fig_loc_axes.update_layout(
                    **PLOTLY_LAYOUT,
                    height=max(320, 230 * len(axis_vars)),
                    title="Magnetometro locale - componenti assiali X/Y/Z",
                )
                st.plotly_chart(fig_loc_axes, use_container_width=True)

            module_vars = [c for c in selected_local if c in ["F", "dF_IGRF"]]
            if module_vars:
                fig_loc_mod = go.Figure()
                for comp in module_vars:
                    fig_loc_mod.add_trace(
                        go.Scatter(x=local_mag.index, y=local_mag[comp], name=comp, line=dict(width=1.25))
                    )
                y_min, y_max = y_range_controls("space_local_module", "modulo/residuo magnetico", [local_mag[c] for c in module_vars])
                apply_y_range(fig_loc_mod, y_min, y_max)
                fig_loc_mod.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Magnetometro locale - modulo F e residuo F-IGRF",
                    yaxis_title="nT",
                )
                st.plotly_chart(fig_loc_mod, use_container_width=True)

            st.caption(
                "Questi canali sono gli stessi caricati nella tab Magnetometro. Il confronto diretto con Kp, Dst, IMF e vento solare aiuta a separare variazioni locali da disturbi geomagnetici esterni."
            )
    else:
        st.info("Carica prima un CSV magnetometrico nella tab Magnetometro per visualizzare X/Y/Z/F anche qui.")

    st.subheader("🏛️ GFZ Potsdam storici")
    selected_gfz = st.multiselect("Indici GFZ", ["Kp", "ap", "Ap", "Hp30", "Hp60", "ap30", "ap60", "SN", "Fobs", "Fadj"], default=["Kp", "Hp30", "ap"])
    if selected_gfz:
        start_gfz = f"{start_date.isoformat()}T00:00:00Z"
        end_gfz = f"{end_date.isoformat()}T23:59:59Z"
        fig = make_subplots(rows=len(selected_gfz), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=selected_gfz)
        for i, idx_name in enumerate(selected_gfz, start=1):
            df_idx = fetch_gfz_index(start_gfz, end_gfz, idx_name)
            if not df_idx.empty:
                fig.add_trace(go.Scatter(x=df_idx["time"], y=df_idx[idx_name], name=idx_name, line=dict(width=1.2), line_shape="vh"), row=i, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=230 * len(selected_gfz), title="GFZ geomagnetic indices")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # GFZ storici sovrapposti al magnetometro locale
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🔗 GFZ storici sovrapposti al magnetometro locale")
    st.markdown(
        "<div class='info'>"
        "Confronto diretto su asse doppio: segnale locale (blu) vs indice GFZ (arancio tratteggiato). "
        "Se le variazioni del magnetometro locale seguono fedelmente il Kp o l'Hp30, la sorgente è esterna. "
        "Discrepanze persistenti — specialmente nelle ore notturne (viola) — indicano possibile contributo vulcanomagnetico."
        "</div>",
        unsafe_allow_html=True,
    )
    local_mag_sp = st.session_state.get("mag", pd.DataFrame())
    if not local_mag_sp.empty:
        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            gfz_sp_idx = st.selectbox(
                "Indice GFZ",
                ["Kp", "ap", "Hp30", "Hp60", "ap30", "ap60"],
                index=0,
                key="space_gfz_sp_idx",
            )
        with col_sp2:
            mag_sp_ch = st.selectbox(
                "Canale magnetometro",
                [c for c in ["F", "dF_IGRF", "X", "Y", "Z"] if c in local_mag_sp.columns],
                index=min(1, len([c for c in ["F", "dF_IGRF", "X", "Y", "Z"] if c in local_mag_sp.columns]) - 1),
                key="space_mag_sp_ch",
            )
        start_gfz_sp = f"{start_date.isoformat()}T00:00:00Z"
        end_gfz_sp = f"{end_date.isoformat()}T23:59:59Z"
        df_gfz_sp = fetch_gfz_index(start_gfz_sp, end_gfz_sp, gfz_sp_idx)
        if not df_gfz_sp.empty and gfz_sp_idx in df_gfz_sp.columns:
            fig_sp = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                subplot_titles=[
                    f"{mag_sp_ch} locale vs GFZ {gfz_sp_idx}",
                    "Anomalia notturna isolata (campioni notturni evidenziati)",
                ],
            )
            # Row 1: full signal + GFZ
            fig_sp.add_trace(
                go.Scatter(x=local_mag_sp.index, y=local_mag_sp[mag_sp_ch],
                           name=mag_sp_ch, line=dict(color="#60a5fa", width=1.3)),
                row=1, col=1, secondary_y=False,
            )
            night_sp = filter_nighttime(local_mag_sp[mag_sp_ch], night_start_h, night_end_h)
            fig_sp.add_trace(
                go.Scatter(x=night_sp.index, y=night_sp, mode="markers",
                           marker=dict(color="#c084fc", size=3, opacity=0.65),
                           name=f"Notte ({night_label})"),
                row=1, col=1, secondary_y=False,
            )
            fig_sp.add_trace(
                go.Scatter(x=df_gfz_sp["time"], y=df_gfz_sp[gfz_sp_idx],
                           line=dict(color="#f59e0b", width=1.5, dash="dot"),
                           name=f"GFZ {gfz_sp_idx}", opacity=0.9),
                row=1, col=1, secondary_y=True,
            )
            # Row 2: night-only signal with anomaly fill
            if not night_sp.empty:
                night_demeaned = night_sp - night_sp.median()
                filled_anomaly_traces(fig_sp, night_demeaned, "Notte dem.", row=2, col=1)
                add_sigma_lines_enhanced(fig_sp, night_demeaned, row=2, col=1)
            fig_sp.update_yaxes(title_text=f"{mag_sp_ch} (nT)", row=1, col=1, secondary_y=False, color="#60a5fa")
            fig_sp.update_yaxes(title_text=f"GFZ {gfz_sp_idx}", row=1, col=1, secondary_y=True, color="#f59e0b", showgrid=False)
            fig_sp.update_yaxes(title_text="nT (dem.)", row=2, col=1)
            fig_sp.update_layout(**PLOTLY_LAYOUT, height=700,
                                  title=f"GFZ {gfz_sp_idx} vs {mag_sp_ch} locale — confronto spazio-tempo")
            st.plotly_chart(fig_sp, use_container_width=True)
        else:
            st.info(f"Dati GFZ {gfz_sp_idx} non disponibili per il periodo selezionato.")
    else:
        st.info("Carica un CSV magnetometrico nella tab 🧲 Magnetometro per abilitare il confronto GFZ/locale.")


# -----------------------------------------------------------------------------
# NASA DONKI tab - API access inherited from working app4.py
# -----------------------------------------------------------------------------
with tab_nasa:
    st.header("🛸 NASA DONKI - eventi solari storici")
    st.markdown(
        "<div class='info'>Questa sezione usa lo stesso schema API della versione funzionante app4.py: "
        "endpoint DONKI NASA + startDate/endDate + api_key. Serve per contestualizzare flare, CME e "
        "tempeste geomagnetiche rispetto alle anomalie magnetometriche locali.</div>",
        unsafe_allow_html=True,
    )

    start_nasa = start_date.strftime("%Y-%m-%d")
    end_nasa = end_date.strftime("%Y-%m-%d")
    donki_endpoints = {
        "Solar flares (FLR)": "FLR",
        "Geomagnetic storms (GST)": "GST",
        "Coronal mass ejections (CME)": "CME",
        "Solar energetic particles (SEP)": "SEP",
        "Interplanetary shocks (IPS)": "IPS",
        "High-speed streams (HSS)": "HSS",
    }
    selected_donki = st.multiselect(
        "Eventi DONKI da scaricare",
        list(donki_endpoints.keys()),
        default=["Solar flares (FLR)", "Geomagnetic storms (GST)", "Coronal mass ejections (CME)"],
    )

    if selected_donki:
        for label in selected_donki:
            endpoint = donki_endpoints[label]
            df_donki = fetch_nasa_donki(endpoint, start_nasa, end_nasa, nasa_key)
            with st.expander(f"{label}: {len(df_donki)} eventi", expanded=endpoint in {"FLR", "GST", "CME"}):
                if df_donki.empty:
                    st.info("Nessun evento disponibile nel periodo selezionato oppure limite/API non raggiungibile.")
                    continue
                # Normalize common time columns for plotting.
                time_col = next((c for c in ["beginTime", "startTime", "peakTime", "eventTime"] if c in df_donki.columns), None)
                if time_col is not None:
                    df_donki[time_col] = pd.to_datetime(df_donki[time_col], errors="coerce")
                    fig = go.Figure()
                    y_values = df_donki["classType"] if "classType" in df_donki.columns else [endpoint] * len(df_donki)
                    fig.add_trace(
                        go.Scatter(
                            x=df_donki[time_col],
                            y=y_values,
                            mode="markers",
                            name=endpoint,
                            marker=dict(size=9),
                        )
                    )
                    fig.update_layout(**PLOTLY_LAYOUT, title=f"NASA DONKI - {label}", yaxis_title="Classe/evento")
                    st.plotly_chart(fig, use_container_width=True)
                visible_cols = [c for c in ["beginTime", "peakTime", "endTime", "startTime", "classType", "sourceLocation", "activeRegionNum", "gstID", "note"] if c in df_donki.columns]
                if not visible_cols:
                    visible_cols = list(df_donki.columns[:8])
                st.dataframe(df_donki[visible_cols].head(200), use_container_width=True)
                st.download_button(
                    f"Scarica {endpoint} CSV",
                    df_donki.to_csv(index=False).encode("utf-8"),
                    file_name=f"nasa_donki_{endpoint}_{start_nasa}_{end_nasa}.csv",
                    mime="text/csv",
                    key=f"dl_donki_{endpoint}",
                )


# -----------------------------------------------------------------------------
# Processing tab
# -----------------------------------------------------------------------------
with tab_proc:
    st.header("🔬 Processing vulcanomagnetico e confronto temporale")
    st.markdown(
        "<div class='info'><b>Pipeline:</b> F -> ΔF rispetto a IGRF -> rimozione Sq stimata da ore quiete -> "
        "filtro Butterworth zero-phase -> anomalia rolling. Il filtro zero-phase evita shift temporali artificiali, "
        "fondamentali nel confronto con sismicita' e geochimica.</div>",
        unsafe_allow_html=True,
    )

    mag = st.session_state.get("mag", pd.DataFrame())
    if mag.empty:
        st.warning("Carica prima un CSV magnetometrico.")
    else:
        kp_series = None
        if not noaa["kp1m"].empty and "Kp" in noaa["kp1m"].columns:
            kp_tmp = noaa["kp1m"].dropna(subset=["time_tag"]).copy()
            kp_tmp["Kp"] = pd.to_numeric(kp_tmp["Kp"], errors="coerce")
            kp_series = kp_tmp.set_index("time_tag")["Kp"]

        proc = mag.copy()
        proc["dF_IGRF"] = proc["F"] - igrf_f
        sq = estimate_sq_from_quiet_hours(proc["F"], kp_series, sq_kp_threshold)
        proc["F_noSq"] = proc["F"] - sq
        proc["dF_noSq"] = proc["F_noSq"] - igrf_f

        c1, c2, c3 = st.columns(3)
        with c1:
            lowcut = st.number_input("High-pass period (ore, 0=off)", 0.0, 2000.0, 0.0, 1.0)
        with c2:
            highcut = st.number_input("Low-pass period (ore, 0=off)", 0.0, 2000.0, 72.0, 1.0)
        with c3:
            filter_order = st.slider("Ordine Butterworth", 2, 8, 4, 1)

        proc["dF_filtered"] = butterworth_filter(proc["dF_noSq"], lowcut if lowcut > 0 else None, highcut if highcut > 0 else None, filter_order)
        samples_per_hour = max(1, int(round(3600.0 / infer_sampling_seconds(proc.index))))
        window_samples = max(5, rolling_hours * samples_per_hour)
        proc["mag_anomaly"] = rolling_anomaly(proc["dF_filtered"], window_samples)
        proc["mag_z"] = robust_zscore(proc["mag_anomaly"], window_samples)
        st.session_state["processed"] = proc

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                            subplot_titles=["F osservato e Sq stimata", "ΔF no Sq", "ΔF filtrato", "Anomalia rolling / z robusto"])
        fig.add_trace(go.Scatter(x=proc.index, y=proc["F"], name="F", line=dict(width=1.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=proc.index, y=sq, name="Sq", line=dict(width=1.2, dash="dot")), row=1, col=1)
        # Night samples highlighted in the Sq panel
        night_f = filter_nighttime(proc["F"], night_start_h, night_end_h)
        fig.add_trace(go.Scatter(x=night_f.index, y=night_f, mode="markers",
                                 marker=dict(color="#c084fc", size=2.5, opacity=0.6),
                                 name=f"Notte ({night_label})", showlegend=True), row=1, col=1)
        # ΔF no Sq — filled anomaly
        filled_anomaly_traces(fig, proc["dF_noSq"], "dF noSq", row=2, col=1)
        add_anomaly_background_bands(fig, proc["dF_noSq"], row=2, col=1)
        # ΔF filtered — filled anomaly
        filled_anomaly_traces(fig, proc["dF_filtered"], "dF filt.", row=3, col=1)
        add_anomaly_background_bands(fig, proc["dF_filtered"], row=3, col=1)
        # Anomalia rolling — filled anomaly + sigma lines
        filled_anomaly_traces(fig, proc["mag_anomaly"], "anomalia", row=4, col=1,
                               color_pos="rgba(74,222,128,0.25)", color_neg="rgba(248,113,113,0.25)")
        add_sigma_lines_enhanced(fig, proc["mag_anomaly"], row=4, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=1050, title="Pipeline vulcanomagnetica")
        st.plotly_chart(fig, use_container_width=True)

        if periodogram is not None:
            with st.expander("📈 Spettro / periodogramma ΔF filtrato"):
                s = proc["dF_filtered"].dropna()
                if len(s) > 32:
                    fs = 1.0 / infer_sampling_seconds(s.index)
                    f, pxx = periodogram(s.to_numpy(), fs=fs, scaling="density")
                    period_h = np.where(f > 0, 1.0 / f / 3600.0, np.nan)
                    spec = pd.DataFrame({"period_h": period_h, "power": pxx}).replace([np.inf, -np.inf], np.nan).dropna()
                    spec = spec[(spec["period_h"] >= 0.01) & (spec["period_h"] <= 1000)]
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=spec["period_h"], y=spec["power"], mode="lines", name="PSD"))
                    fig_p.update_layout(**PLOTLY_LAYOUT, title="Periodogramma", xaxis_type="log", yaxis_type="log", xaxis_title="Periodo (h)", yaxis_title="PSD")
                    st.plotly_chart(fig_p, use_container_width=True)

        # =====================================================================
        # NIGHTTIME ANALYSIS & VOLCANIC ANOMALY DISCRIMINATION
        # =====================================================================
        st.divider()
        st.header("🌙 Analisi notturna e discriminazione anomalie vulcaniche")
        st.markdown(
            "<div class='info'>"
            "<b>Principio fisico:</b> durante le ore notturne la corrioneto ionosferica Sq si azzera, "
            "il contributo del vento solare è prevalentemente magnetosferico e difficilmente produce variazioni "
            "locali coerenti per giorni. Un'anomalia <b>persistente</b> nei dati notturni, non correlata con Kp/Hp, "
            "è un indicatore primario di sorgente vulcanomagnetica crostale (variazione di temperatura, "
            "redistribuzione fluidi, stress meccanico).<br><br>"
            f"<b>Finestra notturna attiva:</b> {night_label} UTC — modificabile nella sidebar."
            "</div>",
            unsafe_allow_html=True,
        )

        # --- Night vs Day comparison ---
        st.subheader("1. Confronto diurno vs notturno")
        night_anom = filter_nighttime(proc["mag_anomaly"], night_start_h, night_end_h)
        day_anom = proc["mag_anomaly"].loc[~proc["mag_anomaly"].index.isin(night_anom.index)]

        fig_nd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               subplot_titles=["Anomalia NOTTURNA", "Anomalia DIURNA"])
        filled_anomaly_traces(fig_nd, night_anom, "Notte", row=1, col=1,
                               color_pos="rgba(192,132,252,0.25)", color_neg="rgba(248,113,113,0.25)",
                               line_color="#c084fc")
        add_sigma_lines_enhanced(fig_nd, night_anom, row=1, col=1)
        filled_anomaly_traces(fig_nd, day_anom, "Giorno", row=2, col=1,
                               color_pos="rgba(96,165,250,0.20)", color_neg="rgba(251,146,60,0.20)",
                               line_color="#60a5fa")
        add_sigma_lines_enhanced(fig_nd, day_anom, row=2, col=1)
        fig_nd.update_yaxes(title_text="nT (notte)", row=1, col=1)
        fig_nd.update_yaxes(title_text="nT (giorno)", row=2, col=1)
        fig_nd.update_layout(**PLOTLY_LAYOUT, height=560,
                              title="Anomalia rolling: distribuzione diurna vs notturna")
        st.plotly_chart(fig_nd, use_container_width=True)

        col_nd1, col_nd2, col_nd3 = st.columns(3)
        night_std = float(night_anom.std()) if not night_anom.empty else 0.0
        day_std = float(day_anom.std()) if not day_anom.empty else 0.0
        night_p95 = float(night_anom.abs().quantile(0.95)) if not night_anom.empty else 0.0
        col_nd1.metric("σ notturno", f"{night_std:.2f} nT")
        col_nd2.metric("σ diurno", f"{day_std:.2f} nT")
        ratio_label = f"{night_std/day_std:.2f}" if day_std > 0 else "N/A"
        col_nd3.metric("Rapporto σ notte/giorno", ratio_label,
                       help="Valori < 0.7 indicano che il disturbo è prevalentemente diurno (ionosferico). "
                            "Valori > 1.0 suggeriscono una sorgente attiva di notte.")

        # --- Night daily stats and persistence ---
        st.subheader("2. Statistiche notturne giornaliere e persistenza anomalia")
        night_stats = night_daily_stats(proc["mag_anomaly"], night_start_h, night_end_h)
        night_z = filter_nighttime(proc["mag_z"], night_start_h, night_end_h)
        persist_df = night_persistence_score(night_z, threshold=2.0, min_days=2)

        fig_persist = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    subplot_titles=["MAD notturno giornaliero (ampiezza anomalia)",
                                                    "Streak anomalia notturna persistente (giorni consecutivi ≥2σ)"])
        fig_persist.add_trace(
            go.Bar(x=night_stats.index, y=night_stats["night_mad"],
                   marker_color=np.where(night_stats["night_mad"] > night_stats["night_mad"].median() * 2,
                                         "#f87171", "#60a5fa"),
                   name="MAD notturno"),
            row=1, col=1,
        )
        streak_colors = ["#4ade80" if s == 0 else "#fbbf24" if s < 3 else "#f87171"
                         for s in persist_df["streak"].fillna(0)]
        fig_persist.add_trace(
            go.Bar(x=persist_df.index, y=persist_df["streak"],
                   marker_color=streak_colors, name="Giorni consecutivi"),
            row=2, col=1,
        )
        fig_persist.update_yaxes(title_text="MAD (nT)", row=1, col=1)
        fig_persist.update_yaxes(title_text="Giorni", row=2, col=1)
        fig_persist.update_layout(**PLOTLY_LAYOUT, height=500, title="Persistenza anomalia notturna")
        st.plotly_chart(fig_persist, use_container_width=True)

        if not persist_df.empty:
            max_streak = int(persist_df["streak"].max())
            n_flagged = int(persist_df["flagged"].sum())
            c1p, c2p = st.columns(2)
            c1p.metric("Giorni con anomalia notturna ≥2σ", f"{n_flagged}")
            c2p.metric("Streak massima consecutiva", f"{max_streak} giorni",
                       delta="⚠ attenzione" if max_streak >= 3 else None,
                       delta_color="inverse" if max_streak >= 3 else "normal")

        # --- Cross-correlation with GFZ Kp ---
        st.subheader("3. Cross-correlazione: anomalia notturna vs GFZ Kp")
        st.markdown(
            "Una correlazione positiva a lag 0 con Kp indica sorgente <b>esterna</b>. "
            "Correlazione bassa o picco a lag ≠ 0 suggerisce dinamiche <b>locali</b>.",
            unsafe_allow_html=True,
        )
        ccf_lag_max = st.slider("Lag massimo cross-correlazione (ore)", 12, 120, 48, 6, key="ccf_lag_max")
        ccf_resample = st.selectbox("Ricampionamento CCF", ["60", "30", "120"], index=0, key="ccf_resample")
        start_ccf = f"{start_date.isoformat()}T00:00:00Z"
        end_ccf = f"{end_date.isoformat()}T23:59:59Z"
        df_kp_ccf = fetch_gfz_index(start_ccf, end_ccf, "Kp")
        if not df_kp_ccf.empty and "Kp" in df_kp_ccf.columns:
            kp_s_ccf = df_kp_ccf.set_index("time")["Kp"]
            if not night_anom.empty:
                ccf_df = compute_crosscorr(night_anom, kp_s_ccf,
                                           max_lag_hours=ccf_lag_max,
                                           resample_minutes=int(ccf_resample))
                if not ccf_df.empty:
                    peak_row = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
                    fig_ccf = go.Figure()
                    fig_ccf.add_trace(go.Scatter(x=ccf_df["lag_h"], y=ccf_df["correlation"],
                                                  mode="lines", name="CCF",
                                                  line=dict(color="#60a5fa", width=1.5)))
                    fig_ccf.add_vline(x=0, line_dash="dot", line_color="#93a4b7", annotation_text="lag 0")
                    fig_ccf.add_vline(x=float(peak_row["lag_h"]), line_dash="dash",
                                      line_color="#f87171",
                                      annotation_text=f"picco {peak_row['lag_h']:.1f}h r={peak_row['correlation']:.2f}")
                    for thr in [0.3, -0.3]:
                        fig_ccf.add_hline(y=thr, line_dash="dot", line_color="#fbbf24", annotation_text=f"r={thr}")
                    fig_ccf.update_layout(**PLOTLY_LAYOUT, height=350,
                                          title="CCF anomalia notturna vs GFZ Kp",
                                          xaxis_title="Lag (ore)", yaxis_title="r")
                    st.plotly_chart(fig_ccf, use_container_width=True)
                    c1c, c2c, c3c = st.columns(3)
                    c1c.metric("Correlazione a lag 0", f"{ccf_df.loc[ccf_df['lag_h'].abs().idxmin(), 'correlation']:.3f}")
                    c2c.metric("Picco correlazione", f"{float(peak_row['correlation']):.3f}")
                    c3c.metric("Lag del picco", f"{float(peak_row['lag_h']):.1f} h")
        else:
            st.info("Dati GFZ Kp non disponibili; la cross-correlazione richiede una connessione a GFZ Potsdam.")

        # --- Cross-correlation and scatter with auxiliary parameters ---
        aux_for_ccf = st.session_state.get("aux", pd.DataFrame())
        if not aux_for_ccf.empty:
            st.subheader("4. Correlazione anomalia notturna vs parametri ausiliari")
            st.markdown(
                "Scatter e cross-correlazione tra anomalia magnetica notturna e variabili geochimiche, "
                "deformative o termiche. Correlazioni significative con parametri vulcanici — senza "
                "corrispondente aumento di Kp — rafforzano l'ipotesi di sorgente crostale.",
                unsafe_allow_html=True,
            )
            aux_ccf_vars = st.multiselect(
                "Parametri ausiliari da confrontare",
                aux_for_ccf.columns.tolist(),
                default=aux_for_ccf.columns.tolist()[: min(4, len(aux_for_ccf.columns))],
                key="proc_aux_ccf_vars",
            )
            if aux_ccf_vars and not night_anom.empty:
                # Scatter matrix (anomalia notturna daily median vs aux daily median)
                night_daily_med = night_anom.resample("1D").median()
                scatter_rows = 1
                scatter_cols = len(aux_ccf_vars)
                fig_sc = make_subplots(
                    rows=scatter_rows, cols=scatter_cols,
                    subplot_titles=[f"Mag notte vs {v}" for v in aux_ccf_vars],
                )
                for j, var in enumerate(aux_ccf_vars, start=1):
                    aux_daily = aux_for_ccf[var].resample("1D").median()
                    combined_sc = pd.concat([night_daily_med.rename("mag_night"), aux_daily.rename(var)], axis=1).dropna()
                    if not combined_sc.empty:
                        corr_val = float(combined_sc.corr().iloc[0, 1]) if len(combined_sc) > 2 else np.nan
                        fig_sc.add_trace(
                            go.Scatter(
                                x=combined_sc[var], y=combined_sc["mag_night"],
                                mode="markers",
                                marker=dict(
                                    color=combined_sc["mag_night"],
                                    colorscale="RdBu_r",
                                    size=7, opacity=0.8,
                                    showscale=(j == 1),
                                ),
                                name=var,
                                text=combined_sc.index.strftime("%Y-%m-%d"),
                                hovertemplate=f"<b>{var}</b>: %{{x:.3f}}<br>Mag notte: %{{y:.3f}} nT<br>%{{text}}<extra></extra>",
                            ),
                            row=1, col=j,
                        )
                        fig_sc.update_xaxes(title_text=var, row=1, col=j)
                        if j == 1:
                            fig_sc.update_yaxes(title_text="Anomalia notturna (nT/giorno)", row=1, col=j)
                        if np.isfinite(corr_val):
                            fig_sc.update_xaxes(title_text=f"{var} (r={corr_val:.2f})", row=1, col=j)
                fig_sc.update_layout(**PLOTLY_LAYOUT, height=380,
                                      title="Scatter: anomalia notturna vs parametri ausiliari (mediati al giorno)")
                st.plotly_chart(fig_sc, use_container_width=True)

                # CCF with each auxiliary variable
                st.markdown("**Cross-correlazioni temporali (ora-su-ora) anomalia notturna vs ausiliari**")
                fig_mccf = make_subplots(
                    rows=len(aux_ccf_vars), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=[f"CCF mag notte vs {v}" for v in aux_ccf_vars],
                )
                for k, var in enumerate(aux_ccf_vars, start=1):
                    ccf_aux = compute_crosscorr(night_anom, aux_for_ccf[var], max_lag_hours=ccf_lag_max, resample_minutes=60)
                    if not ccf_aux.empty:
                        fig_mccf.add_trace(
                            go.Scatter(x=ccf_aux["lag_h"], y=ccf_aux["correlation"],
                                       mode="lines", name=var,
                                       line=dict(width=1.4)),
                            row=k, col=1,
                        )
                        fig_mccf.add_vline(x=0, line_dash="dot", line_color="#93a4b7")
                        for thr in [0.3, -0.3]:
                            fig_mccf.add_hline(y=thr, line_dash="dot", line_color="#fbbf24", row=k, col=1)
                fig_mccf.update_layout(**PLOTLY_LAYOUT, height=280 * len(aux_ccf_vars),
                                       title="Cross-correlazione anomalia notturna vs parametri ausiliari")
                st.plotly_chart(fig_mccf, use_container_width=True)

        # --- Volcanic discrimination summary ---
        st.subheader("5. Indice discriminazione vulcanomagnetica notturna")
        if not persist_df.empty and not night_anom.empty:
            max_streak_v = int(persist_df["streak"].max())
            n_flagged_v = int(persist_df["flagged"].sum())
            night_rms = float(np.sqrt(np.mean(night_anom.dropna()**2))) if not night_anom.empty else 0.0
            day_rms = float(np.sqrt(np.mean(day_anom.dropna()**2))) if not day_anom.empty else 0.0
            ratio_rms = night_rms / max(day_rms, 1e-12)

            # Simple composite score 0-100
            score_persist = min(max_streak_v / 5.0, 1.0) * 35.0
            score_ratio = min(max(ratio_rms - 0.5, 0) / 1.0, 1.0) * 35.0
            score_nflagged = min(n_flagged_v / 7.0, 1.0) * 30.0
            disc_score = score_persist + score_ratio + score_nflagged
            disc_score_adj = disc_score * space_quality  # penalise during geomagnetic storms

            if disc_score_adj < 20:
                disc_css, disc_level = "green", "BASSO — nessuna evidenza notturna significativa"
            elif disc_score_adj < 45:
                disc_css, disc_level = "yellow", "MODERATO — anomalia notturna presente, monitorare"
            elif disc_score_adj < 70:
                disc_css, disc_level = "orange", "ELEVATO — anomalia notturna persistente, possibile sorgente crostale"
            else:
                disc_css, disc_level = "red", "ALTO — anomalia notturna intensa e persistente, escalation consigliata"

            st.markdown(
                f"<div class='{disc_css}'>"
                f"<h3>Discriminazione notturna: {disc_level}</h3>"
                f"<p><b>Score:</b> {disc_score_adj:.1f}/100 (penalizzato space-weather {space_quality*100:.0f}%) &nbsp;|&nbsp;"
                f"Streak max: {max_streak_v}gg &nbsp;|&nbsp; Notti flaggate: {n_flagged_v} &nbsp;|&nbsp; "
                f"RMS notte/giorno: {ratio_rms:.2f}</p>"
                f"<p><b>Nota:</b> Questo indice è puramente indicativo. "
                f"Un valore elevato richiede convalida con stazione di riferimento regionale (INTERMAGNET/INGV) "
                f"e revisione di un esperto prima di qualsiasi comunicazione ufficiale.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Detail table
            detail_df = pd.DataFrame([
                {"Contributo": "Streak persistenza (×35)", "Valore": f"{max_streak_v} giorni", "Score parziale": f"{score_persist:.1f}"},
                {"Contributo": "Rapporto RMS notte/giorno (×35)", "Valore": f"{ratio_rms:.2f}", "Score parziale": f"{score_ratio:.1f}"},
                {"Contributo": "Notti flaggate ≥2σ (×30)", "Valore": str(n_flagged_v), "Score parziale": f"{score_nflagged:.1f}"},
                {"Contributo": "Penalizzazione space-weather", "Valore": f"{space_quality*100:.0f}%", "Score parziale": f"{disc_score_adj:.1f} finale"},
            ])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

            # Download night data
            night_export = pd.concat([night_anom.rename("mag_anomaly_night"), night_z.rename("mag_z_night")], axis=1)
            st.download_button(
                "📥 Scarica dati notturni processati",
                night_export.to_csv().encode("utf-8"),
                file_name="mag_anomaly_nighttime.csv",
                mime="text/csv",
            )

        # =====================================================================
        # Integrated temporal comparison
        # =====================================================================
        st.divider()
        st.subheader("🔗 Confronto temporale integrato con sismicita' e CSV multiparametrici")
        rows = [("Anomalia magnetica", proc.index, proc["mag_anomaly"])]
        seis_metrics = st.session_state.get("seismic_metrics", pd.DataFrame())
        if not seis_metrics.empty:
            rows.append(("N eventi/giorno", seis_metrics.index, seis_metrics["count"]))
            rows.append(("Energia sismica", seis_metrics.index, seis_metrics["energy_j"]))
        aux = st.session_state.get("aux", pd.DataFrame())
        if not aux.empty:
            chosen_aux = st.multiselect("Aggiungi variabili ausiliarie al confronto", aux.columns.tolist(), default=aux.columns.tolist()[: min(3, len(aux.columns))])
            for var in chosen_aux:
                rows.append((var, aux.index, aux[var]))
        fig_cmp = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=[r[0] for r in rows])
        for i, (name, x, y) in enumerate(rows, start=1):
            fig_cmp.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(width=1.25)), row=i, col=1)
            if name == "Energia sismica":
                fig_cmp.update_yaxes(type="log", row=i, col=1)
        fig_cmp.update_layout(**PLOTLY_LAYOUT, height=220 * len(rows), title="Dashboard temporale integrata")
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.download_button("📥 Scarica processing vulcanomagnetico", proc.to_csv().encode("utf-8"), "vulcanomagnetic_processing_v3.csv", "text/csv")


# -----------------------------------------------------------------------------
# Volcano state tab
# -----------------------------------------------------------------------------
with tab_state:
    st.header("🚦 Valutazione stato vulcanico multiparametrico")
    st.markdown(
        "<div class='info'>L'indice e' uno strumento di triage scientifico, non un sistema ufficiale di allerta. "
        "Integra anomalie robuste di magnetometria, sismicita', geochimica/deformazione e penalizza la confidenza "
        "durante condizioni geomagnetiche disturbate.</div>",
        unsafe_allow_html=True,
    )

    proc = st.session_state.get("processed", pd.DataFrame())
    metrics = st.session_state.get("seismic_metrics", pd.DataFrame())
    aux = st.session_state.get("aux", pd.DataFrame())

    mag_z = 0.0
    if not proc.empty and "mag_z" in proc.columns:
        recent_mag_z = proc["mag_z"].dropna().tail(max(5, rolling_hours))
        mag_z = float(recent_mag_z.abs().median()) if not recent_mag_z.empty else 0.0

    seismic_rate_z = 0.0
    seismic_energy_z = 0.0
    if not metrics.empty:
        seismic_rate_z = float(robust_zscore(metrics["count"]).dropna().tail(3).abs().max() or 0.0)
        energy_log = np.log10(metrics["energy_j"].replace(0, np.nan))
        seismic_energy_z = float(robust_zscore(energy_log).dropna().tail(3).abs().max() or 0.0)

    st.subheader("Mappatura variabili ausiliarie")
    geochem_z = deformation_z = temp_z = 0.0
    if not aux.empty:
        aux_cols = aux.columns.tolist()
        geochem_vars = st.multiselect("Variabili geochimiche/gas", aux_cols, default=[c for c in aux_cols if c.lower() in {"co2", "so2", "h2s", "co2_flux"}])
        deformation_vars = st.multiselect("Variabili deformative/idrologiche", aux_cols, default=[c for c in aux_cols if c.lower() in {"tilt", "gps", "lev", "level", "strain"}])
        temp_vars = st.multiselect("Variabili termiche", aux_cols, default=[c for c in aux_cols if "temp" in c.lower() or "twater" in c.lower()])

        def group_z(columns: Iterable[str]) -> float:
            values = []
            for col_name in columns:
                z = robust_zscore(aux[col_name]).dropna().tail(5).abs()
                if not z.empty:
                    values.append(float(z.max()))
            return float(max(values)) if values else 0.0

        geochem_z = group_z(geochem_vars)
        deformation_z = group_z(deformation_vars)
        temp_z = group_z(temp_vars)
    else:
        st.info("Carica CSV ausiliari per includere geochimica/deformazione/termica nello stato.")

    state = compute_volcano_state(
        mag_z=mag_z,
        seismic_rate_z=seismic_rate_z,
        seismic_energy_z=seismic_energy_z,
        geochem_z=geochem_z,
        deformation_z=deformation_z,
        temp_z=temp_z,
        space_quality=space_quality,
        space_flags=space_flags,
    )

    st.markdown(
        f"<div class='{state.css_class}'><h3>{state.level}</h3>"
        f"<p><b>Score multiparametrico:</b> {state.score:.1f}/100 &nbsp; | &nbsp; "
        f"<b>Confidenza vulcanomagnetica:</b> {state.confidence:.0f}%</p>"
        f"<p>{state.explanation}</p>"
        f"<p><b>Azione suggerita:</b> {state.recommended_action}</p></div>",
        unsafe_allow_html=True,
    )

    df_state = pd.DataFrame(
        [
            {"Parametro": "Magnetometria", "z/score": mag_z},
            {"Parametro": "Rate sismico", "z/score": seismic_rate_z},
            {"Parametro": "Energia sismica", "z/score": seismic_energy_z},
            {"Parametro": "Geochimica/gas", "z/score": geochem_z},
            {"Parametro": "Deformazione/idrologia", "z/score": deformation_z},
            {"Parametro": "Termico", "z/score": temp_z},
            {"Parametro": "Qualita' space-weather", "z/score": space_quality},
        ]
    )
    fig = go.Figure(go.Bar(x=df_state["Parametro"], y=df_state["z/score"], name="contributi"))
    fig.update_layout(**PLOTLY_LAYOUT, title="Contributi alla valutazione", yaxis_title="z robusto / qualita'")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_state.round(3), use_container_width=True)


# -----------------------------------------------------------------------------
# Tools/integrations tab
# -----------------------------------------------------------------------------
with tab_tools:
    st.header("🧰 Strumenti funzionali e integrazioni consigliate")
    st.markdown(
        """
### Funzioni gia' implementate in questa versione
- Upload CSV magnetometrico con F o X/Y/Z, filtro Hampel, ricampionamento, ΔF rispetto a IGRF.
- Sezione dedicata a campo magnetico terrestre e attivita' solare: Kp, Dst, IMF Bz/Bt, vento solare, GOES X-ray, GFZ Kp/Hp/ap.
- Download sismicita' da servizi FDSN configurabili e upload CSV sismico locale.
- Upload multiparametrico per geochimica, idrologia, deformazione e temperatura.
- Stato vulcanico multiparametrico VERDE/GIALLO/ARANCIO/ROSSO con confidenza penalizzata da space-weather.
- Confronto temporale integrato e download dei dataset elaborati.

### Integrazioni fortemente consigliate
1. **INTERMAGNET/INGV AQU-DUR**: scarico automatico minuto/secondo per sottrarre una reference station regionale.
2. **ObsPy/EIDA**: download waveform per RSAM, tremor amplitude, spectral centroid, VLP/LP event rate.
3. **GNSS/Tilt**: ingestione CSV o API per radial/tangential tilt, baseline GPS, strain e velocita'.
4. **Geochimica automatica**: CO2 diffuso, SO2 plume, rapporto CO2/SO2, temperatura fumarole, livello/temperatura pozzi.
5. **Database locale SQLite/PostgreSQL**: persistenza dei dati IoT, quality flags, audit trail degli eventi e versioning del processing.
6. **Alerting**: soglie con isteresi temporale, invio email/Telegram solo dopo persistenza multi-campione e verifica Kp/Dst.
7. **Modello fisico**: inversione semplice di dipolo magnetico o sorgente termomagnetica per stimare profondita'/intensita' della sorgente.

### Colonne CSV consigliate
- Magnetometro: `Datetime, X, Y, Z, F, T_sensor`
- Sismica: `Datetime, latitude, longitude, depth_km, magnitude`
- Geochimica: `Datetime, CO2, SO2, H2S, fumarole_temp, soil_temp, well_level, conductivity`
- Deformazione: `Datetime, tilt_x, tilt_y, gps_e, gps_n, gps_u, strain`
        """
    )

    st.subheader("🔌 Registro API recuperate / integrate")
    api_rows = []
    for family, values in API_DATABASES.items():
        for key, value in values.items():
            if key == "descrizione":
                continue
            api_rows.append(
                {
                    "Database": family,
                    "Codice/API": key,
                    "Endpoint o parametri": str(value),
                    "Descrizione": values.get("descrizione", ""),
                }
            )
    st.dataframe(pd.DataFrame(api_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Scarica registro API CSV",
        pd.DataFrame(api_rows).to_csv(index=False).encode("utf-8"),
        file_name="geomagvolcano_api_registry.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "GeoMagVolcano Monitor v4.0 | Analisi notturna vulcanomagnetica + GFZ overlay + cross-correlazione + discriminazione anomalie | "
    "NASA DONKI + NOAA SWPC + GFZ Potsdam + FDSN + Open-Meteo | "
    "Uso scientifico preliminare: validare sempre con reti ufficiali e analisi esperta."
)


# =============================================================================
# Entry point — rende lo script eseguibile direttamente con:
#   python geomag_app_v6.py          (oppure ./geomag_app_v6.py)
# Streamlit viene avviato in un sottoprocesso, esattamente come farebbe:
#   streamlit run geomag_app_v6.py [--server.port N] [--server.headless true]
# =============================================================================

if __name__ == "__main__" and os.environ.get("_GEOMAG_LAUNCHED_VIA_STREAMLIT") != "1":
    # Streamlit executes this same file with __name__ set to "__main__" too,
    # so without the env-var guard above every relaunch would spawn another
    # "streamlit run" subprocess that opens yet another browser tab, forever.
    import subprocess
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="GeoMagVolcano Monitor — avvio diretto",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Porta su cui esporre l'app Streamlit",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Indirizzo di binding (usa 0.0.0.0 per accesso di rete)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Non aprire automaticamente il browser",
    )
    args = parser.parse_args()

    # Verifica dipendenze prima di lanciare
    _missing = []
    for _pkg in ("streamlit", "pandas", "numpy", "requests", "plotly"):
        try:
            __import__(_pkg)
        except ImportError:
            _missing.append(_pkg)
    if _missing:
        print(
            f"[GeoMagVolcano] Dipendenze mancanti: {', '.join(_missing)}\n"
            f"Installa con:  pip install {' '.join(_missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, "-m", "streamlit", "run", script_path,
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.no_browser else "false",
        "--browser.gatherUsageStats", "false",
    ]

    print(
        f"[GeoMagVolcano] Avvio su http://{args.host}:{args.port}\n"
        f"Premi Ctrl+C per fermare."
    )
    _child_env = os.environ.copy()
    _child_env["_GEOMAG_LAUNCHED_VIA_STREAMLIT"] = "1"
    try:
        subprocess.run(cmd, check=True, env=_child_env)
    except KeyboardInterrupt:
        print("\n[GeoMagVolcano] Fermato.")
    except subprocess.CalledProcessError as exc:
        print(f"[GeoMagVolcano] Errore avvio Streamlit: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)
