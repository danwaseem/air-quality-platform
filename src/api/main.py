"""
Air Quality Prediction API

Loads trained GradientBoosting (and optionally RandomForest) models at startup
and serves next-24-hour predictions for PM2.5, NO2, and ozone.

Startup behaviour
-----------------
- Scans MODELS_DIR (default: ./models) for gbm_<poll>.joblib and rf_<poll>.joblib.
- RF models larger than MAX_MODEL_SIZE_MB (default: 200 MB) are skipped with a
  warning — RF files are ~900 MB each and exceed Cloud Run's memory budget.
- Metrics are read from MODELS_DIR/metrics.json.

Feature vector (20 features)
-----------------------------
17 AQS features: lag_1h/24h/168h, roll_24h/168h mean+std, cyclical time,
  season, latitude, longitude, drift_flag.
3 weather features: temperature_2m (°C), wind_speed_10m (m/s),
  relative_humidity_2m (%) — live from OpenWeather / Open-Meteo in
  /predict/live; passed as NaN in /predict (HistGBR handles natively).

Environment variables
---------------------
MODELS_DIR          Path to model artifacts directory (default: models)
MAX_MODEL_SIZE_MB   Max model file size to load in MB (default: 200)
PORT                Uvicorn listen port (default: 8080, used by Cloud Run)
AIRNOW_KEY          EPA AirNow API key (required for /predict/live)
OPENWEATHER_KEY     OpenWeatherMap key (optional; falls back to Open-Meteo if absent)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from src.features.build_features import FEATURE_COLS, POLLUTANTS
from src.api.schemas import (
    HealthResponse,
    LivePredictRequest,
    LivePredictResponse,
    ModelInfo,
    ModelMetrics,
    PredictRequest,
    PredictResponse,
    RootResponse,
)

log = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────

UNITS: dict[str, str] = {
    "pm25":  "µg/m³",
    "no2":   "ppb",
    "ozone": "ppm",
}

# Clip predictions to these bounds (same as cleaning pipeline)
PHYS_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25":  (0.0, 2000.0),
    "no2":   (0.0,  500.0),
    "ozone": (0.0,    0.6),
}

# ── Runtime configuration ─────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MAX_MODEL_BYTES = int(os.getenv("MAX_MODEL_SIZE_MB", "200")) * 1_000_000

# ── Application state (populated at startup) ──────────────────────────────────

# {pollutant: {model_name: fitted_model}}
_MODELS: dict[str, dict[str, Any]] = {p: {} for p in POLLUTANTS}
# {pollutant: {model_name: {mae, rmse, r2, n_test}}}
_METRICS: dict[str, Any] = {}

# Live-data clients — initialized lazily on first /predict/live request
_airnow_client:  Any = None
_weather_client: Any = None


def _get_airnow_client() -> Any:
    """Return a cached AirNowClient, initializing it on first call.

    Reads three optional env vars so the bounding box and timeout can be
    tuned without a code change:
      AIRNOW_TIMEOUT   read timeout in seconds          (default 60)
      AIRNOW_BBOX_LAT  bbox half-height in degrees      (default 0.15 ≈ 10 mi)
      AIRNOW_BBOX_LON  bbox half-width  in degrees      (default 0.18 ≈ 10 mi)

    Smaller bbox → fewer observations → faster response, but more risk of
    gaps if nearby stations have outages.  The gap-warning behavior in
    LiveObservations surfaces this transparently in every /predict/live response.
    Widen for rural areas: AIRNOW_BBOX_LAT=0.36 AIRNOW_BBOX_LON=0.45
    """
    global _airnow_client
    if _airnow_client is None:
        from src.ingestion.airnow_client import AirNowClient
        key = os.getenv("AIRNOW_KEY", "")
        if not key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "AIRNOW_KEY environment variable is not set. "
                    "Register at https://docs.airnowapi.org/account/request/ "
                    "and add AIRNOW_KEY to your .env file."
                ),
            )
        _airnow_client = AirNowClient(
            api_key=key,
            timeout=      int(os.getenv("AIRNOW_TIMEOUT",   "60")),
            bbox_deg_lat=float(os.getenv("AIRNOW_BBOX_LAT", "0.15")),
            bbox_deg_lon=float(os.getenv("AIRNOW_BBOX_LON", "0.18")),
        )
        log.info(
            "AirNow client: timeout=%ss  bbox=±%.2f°lat ±%.2f°lon",
            _airnow_client.timeout,
            _airnow_client.bbox_deg_lat,
            _airnow_client.bbox_deg_lon,
        )
    return _airnow_client


def _get_weather_client() -> Any | None:
    """Return a cached OpenWeatherClient, or None if key not configured."""
    global _weather_client
    if _weather_client is None:
        from src.ingestion.weather_client import OpenWeatherClient
        key = os.getenv("OPENWEATHER_KEY", "")
        if key:
            _weather_client = OpenWeatherClient(api_key=key)
        else:
            log.info("OPENWEATHER_KEY not set — /predict/live will use Open-Meteo as weather source")
            _weather_client = False  # sentinel: tried but not configured
    return _weather_client if _weather_client else None


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> None:
    """
    Discover and load model files from MODELS_DIR.
    Files larger than MAX_MODEL_BYTES are skipped to stay within container memory.
    """
    for pollutant in POLLUTANTS:
        _MODELS[pollutant] = {}
        for model_name, filename in [
            ("GradientBoosting", f"gbm_{pollutant}.joblib"),
            ("RandomForest",     f"rf_{pollutant}.joblib"),
        ]:
            path = MODELS_DIR / filename
            if not path.exists():
                log.debug("Not found, skipping: %s", path)
                continue

            size_mb = path.stat().st_size / 1e6
            if path.stat().st_size > MAX_MODEL_BYTES:
                log.warning(
                    "Skipping %s/%s (%.0f MB > MAX_MODEL_SIZE_MB=%s MB)",
                    pollutant, model_name, size_mb,
                    os.getenv("MAX_MODEL_SIZE_MB", "200"),
                )
                continue

            log.info("Loading %s / %s (%.1f MB) …", pollutant, model_name, size_mb)
            _MODELS[pollutant][model_name] = joblib.load(path)
            log.info("  loaded %s / %s", pollutant, model_name)

    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        _METRICS.update(json.loads(metrics_path.read_text()))
        log.info("Metrics loaded from %s", metrics_path)
    else:
        log.warning("metrics.json not found at %s — /models will return empty metrics", metrics_path)

    n_loaded = sum(len(v) for v in _MODELS.values())
    log.info("Startup complete: %d model(s) loaded", n_loaded)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_models()
    yield
    # Nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Air Quality Prediction API",
    description=(
        "Next-24-hour PM2.5, NO2, and ozone predictions from EPA AQS station features.\n\n"
        "**Units:** PM2.5 in µg/m³ · NO2 in ppb · Ozone in ppm\n\n"
        "**Default model:** GradientBoosting (HistGBR). "
        "Set `model=RandomForest` if that variant is loaded."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Feature helpers ───────────────────────────────────────────────────────────

def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _assemble_features(req: PredictRequest, ts: datetime) -> dict[str, float | None]:
    """
    Build the 20-feature dict in FEATURE_COLS order.
    Time features are derived from `ts`; optional readings become None (→ NaN for GBM).
    Weather features are absent from the manual request schema so they pass as None;
    HistGBR handles NaN natively — the model was trained with them.
    """
    hour_sin, hour_cos = _cyclical(ts.hour, 24)
    dow_sin,  dow_cos  = _cyclical(ts.weekday(), 7)
    month_sin, month_cos = _cyclical(ts.month, 12)

    r = req.readings
    return {
        "lag_1h":         r.lag_1h,
        "lag_24h":        r.lag_24h,
        "lag_168h":       r.lag_168h,
        "roll_24h_mean":  r.roll_24h_mean,
        "roll_24h_std":   r.roll_24h_std,
        "roll_168h_mean": r.roll_168h_mean,
        "roll_168h_std":  r.roll_168h_std,
        "hour_sin":       hour_sin,
        "hour_cos":       hour_cos,
        "dow_sin":        dow_sin,
        "dow_cos":        dow_cos,
        "month_sin":      month_sin,
        "month_cos":      month_cos,
        "season":         float((ts.month - 1) // 3),
        "latitude":       req.station.latitude,
        "longitude":      req.station.longitude,
        "drift_flag":     float(r.drift_flag),
        # Weather features not available through manual endpoint
        "temperature_2m":       None,
        "wind_speed_10m":       None,
        "relative_humidity_2m": None,
    }


def _to_frame(features: dict[str, float | None]) -> pd.DataFrame:
    """
    Convert feature dict to a single-row DataFrame in FEATURE_COLS order.
    Preserves column names so sklearn doesn't warn about missing feature names.
    Missing keys (e.g. weather absent from manual /predict) are treated as NaN.
    """
    row = []
    for col in FEATURE_COLS:
        v = features.get(col)
        row.append(float(v) if v is not None else float("nan"))
    return pd.DataFrame([row], columns=FEATURE_COLS)


# ── Live weather helpers ──────────────────────────────────────────────────────

@dataclasses.dataclass
class _LiveWeather:
    """Result of a live weather fetch — maps directly to the three model features."""
    temperature_2m:       float | None = None   # °C
    wind_speed_10m:       float | None = None   # m/s
    relative_humidity_2m: float | None = None   # %
    source:               str          = "unknown"
    error:                str | None   = None


def _fetch_live_weather(lat: float, lon: float) -> _LiveWeather:
    """
    Fetch current weather for (lat, lon) for model feature assembly.

    Priority:
    1. OpenWeatherMap — if OPENWEATHER_KEY is set and the call succeeds.
    2. Open-Meteo current forecast — free, no key, same variable names and
       units as the historical archive used during training.

    Never raises; sets .error on total failure so callers can degrade to NaN.
    """
    from src.ingestion.weather_client import fetch_openmeteo_current

    # ── Try OpenWeather first ─────────────────────────────────────────────────
    ow_client = _get_weather_client()
    if ow_client:
        wx = ow_client.fetch_current(lat, lon)
        if not wx.error:
            return _LiveWeather(
                temperature_2m=       wx.temp_c,
                wind_speed_10m=       wx.wind_speed_ms,
                relative_humidity_2m= float(wx.humidity_pct) if wx.humidity_pct is not None else None,
                source="OpenWeatherMap",
            )
        log.warning("OpenWeather fetch failed (%s); falling back to Open-Meteo", wx.error)

    # ── Fall back to Open-Meteo current forecast (free, no key) ──────────────
    wx = fetch_openmeteo_current(lat, lon)
    if not wx.error:
        return _LiveWeather(
            temperature_2m=       wx.temp_c,
            wind_speed_10m=       wx.wind_speed_ms,
            relative_humidity_2m= float(wx.humidity_pct) if wx.humidity_pct is not None else None,
            source="Open-Meteo (current forecast)",
        )

    return _LiveWeather(
        error=f"OpenWeatherMap and Open-Meteo both failed. Last error: {wx.error}",
        source="all weather sources failed",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=RootResponse, tags=["meta"])
def root() -> RootResponse:
    """Basic API info and available endpoints."""
    return RootResponse(
        name="Air Quality Prediction API",
        version="0.1.0",
        description=(
            "Next-24-hour PM2.5, NO2, and ozone predictions "
            "from EPA AQS monitoring station features."
        ),
        docs_url="/docs",
        endpoints=["GET /", "GET /health", "GET /models", "POST /predict", "POST /predict/live"],
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """
    Liveness / readiness probe.

    Returns `status: ok` when at least one model is loaded,
    `status: degraded` when no models are available.
    """
    loaded = {p: list(m.keys()) for p, m in _MODELS.items() if m}
    total = sum(len(v) for v in loaded.values())
    return HealthResponse(
        status="ok" if total > 0 else "degraded",
        models_loaded=loaded,
        total_models=total,
    )


@app.get("/models", response_model=list[ModelInfo], tags=["models"])
def list_models() -> list[ModelInfo]:
    """
    List all pollutants, which model variants are loaded for each,
    and their test-set performance metrics.
    """
    return [
        ModelInfo(
            pollutant=p,
            available_models=list(_MODELS.get(p, {}).keys()),
            metrics=_METRICS.get(p, {}),
        )
        for p in POLLUTANTS
    ]


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict the pollutant concentration 24 hours from `prediction_time`.

    **Required readings:** `lag_1h`, `lag_24h`

    **Optional readings:** `lag_168h`, `roll_*` — omitting them passes NaN to the
    model. GradientBoosting (HistGBR) handles this natively. RandomForest does not
    and will return HTTP 422 if any optional field is absent.

    **Units:** PM2.5 µg/m³ · NO2 ppb · Ozone ppm
    """
    pollutant = req.pollutant

    # ── Resolve model ─────────────────────────────────────────────────────────
    poll_models = _MODELS.get(pollutant, {})
    if not poll_models:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No models loaded for '{pollutant}'. "
                "Check /health for currently loaded models."
            ),
        )

    model_name = req.model or "GradientBoosting"
    if model_name not in poll_models:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' is not loaded for '{pollutant}'. "
                f"Loaded models: {list(poll_models)}."
            ),
        )

    model = poll_models[model_name]

    # ── Timestamps ────────────────────────────────────────────────────────────
    ts = (req.prediction_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target_ts = ts + timedelta(hours=24)

    # ── Feature assembly ──────────────────────────────────────────────────────
    features = _assemble_features(req, ts)

    # RandomForest cannot handle NaN — validate completeness
    if model_name == "RandomForest":
        missing = [
            k for k, v in features.items()
            if v is None or (isinstance(v, float) and math.isnan(v))
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "RandomForest requires all feature values. "
                        "The following optional fields were not provided:"
                    ),
                    "missing_features": missing,
                    "hint": (
                        "Supply all readings fields, "
                        "or omit 'model' to use GradientBoosting (handles missing values)."
                    ),
                },
            )

    X = _to_frame(features)
    prediction = float(model.predict(X)[0])

    # Clip to physical plausibility bounds
    lo, hi = PHYS_BOUNDS[pollutant]
    prediction = max(lo, min(hi, prediction))

    # ── Build response ────────────────────────────────────────────────────────
    raw_m = _METRICS.get(pollutant, {}).get(model_name, {})
    return PredictResponse(
        pollutant=pollutant,
        unit=UNITS[pollutant],
        prediction_24h_ahead=round(prediction, 4),
        model_used=model_name,
        model_metrics=ModelMetrics(
            mae=raw_m.get("mae"),
            rmse=raw_m.get("rmse"),
            r2=raw_m.get("r2"),
            n_test=raw_m.get("n_test"),
        ),
        prediction_time=ts.isoformat(),
        target_time=target_ts.isoformat(),
        features_used=features,
    )


# ── /predict/live ─────────────────────────────────────────────────────────────

@app.post("/predict/live", response_model=LivePredictResponse, tags=["prediction"])
def predict_live(req: LivePredictRequest) -> LivePredictResponse:
    """
    Predict the next-24h pollutant concentration using **live** data sources.

    Fetches:
    - Past ~170 h of hourly AirNow concentrations near the requested lat/lon
      (lag/rolling features — AQS pollutant observations).
    - Current weather from OpenWeatherMap (if OPENWEATHER_KEY set) or
      Open-Meteo current forecast (free, no key, same units as training).

    Assembles the full 20-feature vector the weather-augmented GBM was trained
    on and returns the prediction with a **feature_provenance** map.

    Provenance labels:
    - **live** — directly observed from AirNow or live weather API
    - **derived** — computed from the live time series (rolling stats, time encodings)
    - **input** — taken from the request body (lat/lon)
    - **unavailable/backfilled** — not obtainable; NaN passed to model
      (GradientBoosting handles this natively)
    - **derived/assumed** — inferred with an assumption (drift_flag → 0)

    Returns HTTP 422 if essential lags (lag_1h or lag_24h) are unavailable.
    Weather failure is non-fatal — the model runs with NaN weather features
    and a warning is surfaced in the response.
    """
    pollutant = req.pollutant
    lat, lon  = req.latitude, req.longitude

    # ── Resolve model ─────────────────────────────────────────────────────────
    poll_models = _MODELS.get(pollutant, {})
    if not poll_models:
        raise HTTPException(
            status_code=503,
            detail=f"No models loaded for '{pollutant}'. Check /health.",
        )
    model = poll_models.get("GradientBoosting")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"GradientBoosting model not loaded for '{pollutant}'.",
        )

    # ── Fetch live AirNow observations ────────────────────────────────────────
    airnow = _get_airnow_client()
    try:
        obs = airnow.get_observations(lat, lon, pollutant)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"AirNow API error: {exc}") from exc

    # Use the most recent non-null observation hour as t=0
    valid_hours = [(h, v) for h, v in obs.hourly if v is not None]
    if not valid_hours:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"No {pollutant} observations found near ({lat}, {lon}) "
                    f"in the past {airnow.history_hours} hours."
                ),
                "hint": (
                    "Try a larger search area or check whether AirNow monitors "
                    "this pollutant in this region."
                ),
                "airnow_warnings": obs.warnings,
            },
        )

    now_h = valid_hours[-1][0]   # most recent observed hour (UTC, on the hour)
    target_ts = now_h + timedelta(hours=24)

    # ── Build AQS feature vector + provenance ─────────────────────────────────
    features:   dict[str, float | None] = {}
    provenance: dict[str, str]          = {}
    warnings                            = list(obs.warnings)

    def _prov_lag(name: str, offset_h: int) -> None:
        val = obs.at(now_h - timedelta(hours=offset_h))
        features[name]   = val
        provenance[name] = "live" if val is not None else "unavailable/backfilled"
        if val is None:
            warnings.append(
                f"{name}: no AirNow observation at "
                f"{(now_h - timedelta(hours=offset_h)).isoformat()} UTC."
            )

    _prov_lag("lag_1h",   1)
    _prov_lag("lag_24h",  24)
    _prov_lag("lag_168h", 168)

    # Guard: lag_1h and lag_24h are required — refuse to predict without them
    missing_required = [k for k in ("lag_1h", "lag_24h") if features[k] is None]
    if missing_required:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"Cannot build a reliable prediction: essential features "
                    f"{missing_required} are unavailable from AirNow near "
                    f"({lat}, {lon})."
                ),
                "context": (
                    "These lags require AirNow observations from 1h and 24h ago. "
                    "The station may be offline, reporting delayed, or outside "
                    "AirNow's monitoring network for this pollutant."
                ),
                "latest_observation_utc": now_h.isoformat(),
                "coverage_hours": obs.coverage_hours(),
                "airnow_warnings": obs.warnings,
            },
        )

    # Rolling 24h stats
    r24_mean = obs.rolling_mean(now_h, 24)
    r24_std  = obs.rolling_std(now_h, 24)
    features["roll_24h_mean"]   = r24_mean
    features["roll_24h_std"]    = r24_std
    provenance["roll_24h_mean"] = "derived" if r24_mean is not None else "unavailable/backfilled"
    provenance["roll_24h_std"]  = "derived" if r24_std  is not None else "unavailable/backfilled"

    # Rolling 168h stats
    r168_mean = obs.rolling_mean(now_h, 168)
    r168_std  = obs.rolling_std(now_h, 168)
    features["roll_168h_mean"]   = r168_mean
    features["roll_168h_std"]    = r168_std
    provenance["roll_168h_mean"] = "derived" if r168_mean is not None else "unavailable/backfilled"
    provenance["roll_168h_std"]  = "derived" if r168_std  is not None else "unavailable/backfilled"

    if r168_mean is None:
        n_168_obs = sum(1 for h, v in obs.hourly if v is not None
                        and now_h - timedelta(hours=168) <= h < now_h)
        warnings.append(
            f"roll_168h features unavailable: only {n_168_obs} non-null observations "
            "in the 168h window (need ≥2). NaN passed to GBM (handled natively)."
        )

    # Cyclical time features
    hour_sin, hour_cos   = _cyclical(now_h.hour, 24)
    dow_sin,  dow_cos    = _cyclical(now_h.weekday(), 7)
    month_sin, month_cos = _cyclical(now_h.month, 12)

    for k in ("hour_sin", "hour_cos", "dow_sin", "dow_cos",
              "month_sin", "month_cos", "season"):
        provenance[k] = "derived"

    features.update({
        "hour_sin":   hour_sin,
        "hour_cos":   hour_cos,
        "dow_sin":    dow_sin,
        "dow_cos":    dow_cos,
        "month_sin":  month_sin,
        "month_cos":  month_cos,
        "season":     float((now_h.month - 1) // 3),
    })

    # Geography — from request
    features["latitude"]   = lat
    features["longitude"]  = lon
    provenance["latitude"]  = "input"
    provenance["longitude"] = "input"

    # Drift flag — cannot compute rolling z-score from single-point live data
    features["drift_flag"]   = 0.0
    provenance["drift_flag"] = "derived/assumed"
    warnings.append(
        "drift_flag set to 0 (assumed): rolling z-score drift detection "
        "requires a long historical baseline and cannot be computed in real time."
    )

    # ── Live weather features (model inputs) ──────────────────────────────────
    wx_live = _fetch_live_weather(lat, lon)

    if wx_live.error:
        # Surface the failure — do NOT silently zero-fill
        warnings.append(
            f"Live weather unavailable ({wx_live.error}). "
            "temperature_2m, wind_speed_10m, relative_humidity_2m passed as NaN — "
            "GradientBoosting handles missing values natively; expect slightly "
            "lower accuracy than the weather-augmented training baseline."
        )
        wx_prov = "unavailable/backfilled"
    else:
        wx_prov = "live"

    features["temperature_2m"]       = wx_live.temperature_2m
    features["wind_speed_10m"]       = wx_live.wind_speed_10m
    features["relative_humidity_2m"] = wx_live.relative_humidity_2m

    provenance["temperature_2m"]       = wx_prov
    provenance["wind_speed_10m"]       = wx_prov
    provenance["relative_humidity_2m"] = wx_prov

    # ── Predict ───────────────────────────────────────────────────────────────
    X = _to_frame(features)
    prediction = float(model.predict(X)[0])
    lo, hi = PHYS_BOUNDS[pollutant]
    prediction = max(lo, min(hi, prediction))

    # ── Build response ────────────────────────────────────────────────────────
    raw_m = _METRICS.get(pollutant, {}).get("GradientBoosting", {})

    live_count    = sum(1 for v in provenance.values() if v == "live")
    derived_count = sum(1 for v in provenance.values() if v == "derived")
    unavail_count = sum(1 for v in provenance.values() if v == "unavailable/backfilled")
    assumed_count = sum(1 for v in provenance.values() if v == "derived/assumed")

    return LivePredictResponse(
        pollutant=pollutant,
        unit=UNITS[pollutant],
        prediction_24h_ahead=round(prediction, 4),
        model_used="GradientBoosting",
        model_metrics=ModelMetrics(
            mae=raw_m.get("mae"),
            rmse=raw_m.get("rmse"),
            r2=raw_m.get("r2"),
            n_test=raw_m.get("n_test"),
        ),
        prediction_time=now_h.isoformat(),
        target_time=target_ts.isoformat(),
        features_used=features,
        feature_provenance=provenance,
        data_sources={
            "airnow": {
                "observations_fetched":   obs.n_raw_obs,
                "coverage_hours":         obs.coverage_hours(),
                "history_window_hours":   airnow.history_hours,
                "reporting_areas":        obs.sources,
                "latest_observation_utc": now_h.isoformat(),
            },
            "weather": {
                "source":             wx_live.source,
                "temperature_2m_c":   wx_live.temperature_2m,
                "wind_speed_10m_ms":  wx_live.wind_speed_10m,
                "humidity_pct":       wx_live.relative_humidity_2m,
                **({"error": wx_live.error} if wx_live.error else {}),
                "model_note": (
                    "These are live model inputs — GBM was retrained with "
                    "weather features from Open-Meteo historical archive."
                ),
            },
            "feature_summary": {
                "live":                   live_count,
                "derived":                derived_count,
                "unavailable/backfilled": unavail_count,
                "derived/assumed":        assumed_count,
                "total":                  len(provenance),
            },
        },
        warnings=warnings,
    )
