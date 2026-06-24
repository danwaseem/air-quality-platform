"""
EPA AirNow API client for real-time air quality observations.

Uses the /aq/data/ BBOX endpoint which returns raw hourly concentrations
(not AQI) — required so live readings are in the same units the GBM model
was trained on (PM2.5 µg/m³, NO2 ppb, ozone ppm).

Endpoint:
  GET https://www.airnowapi.org/aq/data/
  ?startDate=YYYY-MM-DDTHH&endDate=YYYY-MM-DDTHH
  &parameters=PM25,OZONE,NO2
  &BBOX=minLon,minLat,maxLon,maxLat
  &dataType=C&format=application/json&verbose=1
  &includerawconcentrations=1&API_KEY=...

One API call fetches a full time window for all three pollutants.

Bounding box size tradeoff
--------------------------
A smaller BBOX returns fewer station-hours per API call, which reduces
response time substantially (Phoenix ozone: ~5,500 obs with the old ±25 mi
box vs ~900 obs with ±10 mi).  The downside is that with fewer stations
available, any single-station outage is more likely to leave a gap in the
170-hour time series, causing lag_1h/24h or rolling-window features to be
unavailable — which the endpoint surfaces honestly in `warnings`.

Defaults (±10 miles / ~16 km) are tuned for dense urban areas.  Widen for
rural sites that may have only one nearby monitor:
  AIRNOW_BBOX_LAT=0.36  AIRNOW_BBOX_LON=0.45  (restores the original ±25 mi)

Configurable env vars
---------------------
  AIRNOW_TIMEOUT   Read timeout in seconds (default 60)
  AIRNOW_BBOX_LAT  Half-height of bounding box in decimal degrees (default 0.15 ≈ 10 mi)
  AIRNOW_BBOX_LON  Half-width  of bounding box in decimal degrees (default 0.18 ≈ 10 mi)

Set AIRNOW_KEY in .env.  Register at https://docs.airnowapi.org/account/request/
"""
from __future__ import annotations

import logging
import math
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_URL = "https://www.airnowapi.org/aq/data/"

# API parameter name → internal pollutant name
_AIRNOW_PARAM = {"pm25": "PM25", "no2": "NO2", "ozone": "OZONE"}

# Response "Parameter" values AirNow uses (PM2.5 has a dot in responses)
_RESPONSE_PARAM = {"pm25": "PM2.5", "no2": "NO2", "ozone": "OZONE"}

# Default bounding box: ±10 miles around the requested point.
# Tighter than the old ±25 mi to reduce observation count and API latency.
# Override with AIRNOW_BBOX_LAT / AIRNOW_BBOX_LON env vars for rural areas.
_BBOX_DEG_LAT = 0.15   # 1° lat ≈ 111 km; 0.15° ≈ 17 km ≈ 10 mi
_BBOX_DEG_LON = 0.18   # 1° lon at 37°N ≈ 89 km; 0.18° ≈ 16 km ≈ 10 mi

_DEFAULT_TIMEOUT = 60  # seconds; override with AIRNOW_TIMEOUT env var

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0   # seconds; doubles on each retry


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class LiveObservations:
    """
    Hourly concentration series and derived lag features for one pollutant.

    All times are UTC.  Missing hours have NaN values.
    """
    pollutant:    str
    # Ordered list of (utc_hour: datetime, concentration: float|None)
    hourly:       list[tuple[datetime, float | None]] = field(default_factory=list)
    # Reporting area names that contributed data
    sources:      list[str] = field(default_factory=list)
    # Number of distinct station-hour obs fetched (before averaging)
    n_raw_obs:    int = 0
    warnings:     list[str] = field(default_factory=list)

    # ── Feature extraction ────────────────────────────────────────────────────

    def _series(self) -> dict[datetime, float | None]:
        return {h: v for h, v in self.hourly}

    def at(self, utc_hour: datetime) -> float | None:
        """Return the observed concentration for an exact UTC hour, or None."""
        key = utc_hour.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        return self._series().get(key)

    def rolling_mean(self, end_hour: datetime, window_hours: int) -> float | None:
        """Mean of the `window_hours` hours ending at (but not including) end_hour."""
        vals = [
            v for h, v in self.hourly
            if v is not None
            and end_hour - timedelta(hours=window_hours) <= h < end_hour
        ]
        return round(sum(vals) / len(vals), 4) if len(vals) >= 2 else None

    def rolling_std(self, end_hour: datetime, window_hours: int) -> float | None:
        """Std dev of the `window_hours` hours ending at (but not including) end_hour."""
        vals = [
            v for h, v in self.hourly
            if v is not None
            and end_hour - timedelta(hours=window_hours) <= h < end_hour
        ]
        if len(vals) < 2:
            return None
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return round(math.sqrt(variance), 4)

    def coverage_hours(self) -> int:
        """How many hours in the series have a non-null value."""
        return sum(1 for _, v in self.hourly if v is not None)


# ── Client ────────────────────────────────────────────────────────────────────

class AirNowClient:
    """
    Thin wrapper around the AirNow /aq/data/ BBOX endpoint.

    Parameters
    ----------
    api_key       : AIRNOW_KEY from .env
    history_hours : Hours of history to fetch (default 170 → covers all lag
                    and 168h rolling features with a 2h buffer).
                    More hours = one larger API response (not extra calls).
    bbox_deg_lat  : Half-height of bounding box in degrees (default ≈10 mi).
                    Widen for rural areas with sparse monitors.
    bbox_deg_lon  : Half-width  of bounding box in degrees (default ≈10 mi).
    timeout       : HTTP read timeout in seconds (default 60).
                    Dense-metro ozone requests (~5,500 obs) need >30 s.
    """

    def __init__(
        self,
        api_key: str,
        history_hours: int = 170,
        bbox_deg_lat: float = _BBOX_DEG_LAT,
        bbox_deg_lon: float = _BBOX_DEG_LON,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key       = api_key
        self.history_hours = history_hours
        self.bbox_deg_lat  = bbox_deg_lat
        self.bbox_deg_lon  = bbox_deg_lon
        self.timeout       = timeout
        self._session      = requests.Session()

    # ── Low-level fetch ───────────────────────────────────────────────────────

    def _bbox(self, lat: float, lon: float) -> str:
        min_lon = round(lon - self.bbox_deg_lon, 4)
        max_lon = round(lon + self.bbox_deg_lon, 4)
        min_lat = round(lat - self.bbox_deg_lat, 4)
        max_lat = round(lat + self.bbox_deg_lat, 4)
        return f"{min_lon},{min_lat},{max_lon},{max_lat}"

    def _get(self, params: dict[str, Any]) -> list[dict]:
        """HTTP GET with retry / rate-limit handling."""
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(
                    _BASE_URL, params=params, timeout=self.timeout
                )
                if resp.status_code == 429:
                    wait = _RETRY_DELAY * (2 ** attempt)
                    log.warning("AirNow rate-limited; waiting %.1fs", wait)
                    _time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                # AirNow returns a plain list or {"message": "..."} on error
                if isinstance(data, dict):
                    msg = data.get("message", data.get("Message", str(data)))
                    raise RuntimeError(f"AirNow API error: {msg}")
                return data
            except requests.RequestException as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise RuntimeError(f"AirNow request failed after {_MAX_RETRIES} attempts: {exc}") from exc
                _time.sleep(_RETRY_DELAY * (2 ** attempt))
        return []

    # ── Public: fetch raw records ─────────────────────────────────────────────

    def fetch_raw(
        self,
        lat: float,
        lon: float,
        end_utc: datetime | None = None,
    ) -> list[dict]:
        """
        Fetch all PM2.5, NO2, and ozone records within the BBOX for the past
        `history_hours` hours.  Returns the raw AirNow JSON list.

        Each record has keys: Latitude, Longitude, UTC, Parameter,
        Unit, Value (NowCast), RawConcentration, AQI, Category,
        SiteName, AgencyName, FullAQSCode (verbose=1).
        """
        if end_utc is None:
            end_utc = datetime.now(timezone.utc)

        # Align to the hour boundary (AirNow returns whole-hour obs)
        end_h   = end_utc.replace(minute=0, second=0, microsecond=0)
        start_h = end_h - timedelta(hours=self.history_hours)

        params = {
            "startDate":               start_h.strftime("%Y-%m-%dT%H"),
            "endDate":                 end_h.strftime("%Y-%m-%dT%H"),
            "parameters":              "PM25,OZONE,NO2",
            "BBOX":                    self._bbox(lat, lon),
            "dataType":                "C",
            "format":                  "application/json",
            "verbose":                 1,
            "includerawconcentrations": 1,
            "API_KEY":                 self.api_key,
        }
        log.info(
            "AirNow fetch: %.4f,%.4f  %s → %s",
            lat, lon,
            start_h.strftime("%Y-%m-%dT%H"),
            end_h.strftime("%Y-%m-%dT%H"),
        )
        return self._get(params)

    # ── Public: build observations object ────────────────────────────────────

    def get_observations(
        self,
        lat: float,
        lon: float,
        pollutant: str,
        end_utc: datetime | None = None,
    ) -> LiveObservations:
        """
        Fetch raw records and aggregate them into an `LiveObservations` object.

        Multiple nearby stations are averaged per hour (mirrors the training
        pipeline's POC-averaging strategy).  Hours with no data are represented
        as None in the hourly list.
        """
        if pollutant not in _AIRNOW_PARAM:
            raise ValueError(f"Unknown pollutant '{pollutant}'. Choose from {list(_AIRNOW_PARAM)}")

        if end_utc is None:
            end_utc = datetime.now(timezone.utc)
        end_h = end_utc.replace(minute=0, second=0, microsecond=0)

        obs = LiveObservations(pollutant=pollutant)
        raw = self.fetch_raw(lat, lon, end_utc)

        if not raw:
            obs.warnings.append(
                f"AirNow returned no data for {pollutant} near "
                f"({lat:.4f}, {lon:.4f}) in the past {self.history_hours} h."
            )
            return obs

        obs.n_raw_obs = len(raw)

        # Filter to the requested pollutant
        target_param = _RESPONSE_PARAM[pollutant]
        records = [r for r in raw if r.get("Parameter") == target_param]

        if not records:
            obs.warnings.append(
                f"No '{target_param}' observations in response "
                f"(only: {sorted({r.get('Parameter') for r in raw})})."
            )
            return obs

        # Collect site names
        obs.sources = sorted({r.get("SiteName", "unknown") for r in records if r.get("SiteName")})

        # ── Ozone unit normalisation (done once, before the per-record loop) ──
        # AirNow reports ozone in ppb; AQS (and the trained GBM models) use ppm.
        # 60 ppb → 0.060 ppm.  PM2.5 (µg/m³) and NO2 (ppb) already match
        # training units and must NOT be converted.
        #
        # We sample the Unit field from the first record to detect API changes.
        # If AirNow ever switches ozone to ppm, the Unit field will say "PPM"
        # and we skip the division.  Any other unexpected unit triggers a warning.
        _convert_ozone: bool = False
        if pollutant == "ozone":
            sample_unit = next(
                (r.get("Unit", "").strip().upper() for r in records if r.get("Unit")),
                "PPB",  # assume ppb when Unit field is absent (known AirNow default)
            )
            if sample_unit == "PPB":
                _convert_ozone = True
                log.debug(
                    "Ozone: AirNow Unit=%r — applying ÷1000 (ppb→ppm) "
                    "to match AQS training units",
                    sample_unit,
                )
            elif sample_unit == "PPM":
                log.info(
                    "Ozone: AirNow Unit=%r — no conversion needed (already ppm)",
                    sample_unit,
                )
            else:
                log.warning(
                    "Ozone: unexpected AirNow Unit=%r (expected 'PPB'). "
                    "Skipping ppb→ppm conversion — verify AirNow has not "
                    "changed reporting units before deploying.",
                    sample_unit,
                )

        # Parse UTC timestamps and group by hour
        hour_buckets: dict[datetime, list[float]] = {}
        for rec in records:
            utc_str = rec.get("UTC", "")
            try:
                # AirNow format: "2023-11-15T14:00"
                dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            # Prefer RawConcentration over NowCast Value (more faithful to training data)
            raw_conc = rec.get("RawConcentration")
            nowcast  = rec.get("Value")
            val      = raw_conc if raw_conc is not None else nowcast
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval < 0:
                continue  # physically impossible; skip

            if _convert_ozone:
                fval = fval / 1000.0  # ppb → ppm (AirNow ppb, model trained on ppm)

            hour_buckets.setdefault(dt, []).append(fval)

        # Average across stations per hour; build complete hourly list
        hourly_avgs: dict[datetime, float] = {
            h: round(sum(vals) / len(vals), 4)
            for h, vals in hour_buckets.items()
        }

        # Fill the complete expected time grid (start → end, hourly)
        start_h = end_h - timedelta(hours=self.history_hours)
        current = start_h
        while current <= end_h:
            obs.hourly.append((current, hourly_avgs.get(current)))
            current += timedelta(hours=1)

        missing = sum(1 for _, v in obs.hourly if v is None)
        if missing > 0:
            obs.warnings.append(
                f"{missing} of {len(obs.hourly)} hours have no observation "
                f"(station gaps or AirNow reporting delays)."
            )

        log.info(
            "AirNow %s: %d hours fetched, %d non-null, %d stations: %s",
            pollutant, len(obs.hourly), obs.coverage_hours(),
            len(obs.sources), obs.sources,
        )
        return obs
