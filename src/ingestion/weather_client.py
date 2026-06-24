"""
Weather clients for live model feature assembly.

Two sources are available — the endpoint tries them in order:
  1. OpenWeatherMap  (OPENWEATHER_KEY in .env)  — named stations, rich response
  2. Open-Meteo current forecast (free, no key) — same provider and variable names
     as the historical archive used for training, so units are guaranteed identical.

Both clients return `CurrentWeather` with:
  temp_c        → temperature_2m  (°C)
  wind_speed_ms → wind_speed_10m  (m/s)
  humidity_pct  → relative_humidity_2m (%)

These map directly to the three weather features added in the second training round.
Failures set `.error` and never raise so the caller can degrade to NaN gracefully.
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
_MAX_RETRIES  = 2
_RETRY_DELAY  = 1.5


@dataclass
class CurrentWeather:
    temp_c:        float | None = None
    wind_speed_ms: float | None = None
    humidity_pct:  float | None = None
    conditions:    str          = "unknown"
    station_name:  str          = "unknown"
    source:        str          = "OpenWeatherMap"
    error:         str | None   = None   # set if fetch failed

    def as_dict(self) -> dict:
        return {
            "temp_c":        self.temp_c,
            "wind_speed_ms": self.wind_speed_ms,
            "humidity_pct":  self.humidity_pct,
            "conditions":    self.conditions,
            "station_name":  self.station_name,
            "source":        self.source,
            **({"error": self.error} if self.error else {}),
        }


class OpenWeatherClient:
    """
    Minimal wrapper around the OpenWeatherMap current-weather endpoint.

    Failures degrade gracefully — the caller receives a `CurrentWeather`
    object with `error` set rather than an exception, so weather
    unavailability never blocks a live prediction.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key  = api_key
        self._session = requests.Session()

    def fetch_current(self, lat: float, lon: float) -> CurrentWeather:
        """
        Return current weather conditions for the given coordinates.

        Always returns a `CurrentWeather`; sets `.error` on failure
        so the caller can degrade gracefully without try/except.
        """
        params = {
            "lat":   lat,
            "lon":   lon,
            "appid": self.api_key,
            "units": "metric",   # temperature in °C, wind in m/s
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(_CURRENT_URL, params=params, timeout=10)
                if resp.status_code == 401:
                    return CurrentWeather(
                        error="OpenWeather API key invalid or not yet activated (can take a few hours)."
                    )
                if resp.status_code == 429:
                    _time.sleep(_RETRY_DELAY * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return _parse(data)
            except requests.RequestException as exc:
                if attempt == _MAX_RETRIES - 1:
                    log.warning("OpenWeather fetch failed: %s", exc)
                    return CurrentWeather(error=str(exc))
                _time.sleep(_RETRY_DELAY)

        return CurrentWeather(error="Max retries exceeded")


def _parse(data: dict) -> CurrentWeather:
    """Parse an OpenWeather current-weather JSON response."""
    try:
        main    = data.get("main", {})
        wind    = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        name    = data.get("name", "unknown")

        return CurrentWeather(
            temp_c        = round(float(main["temp"]),  1) if "temp"  in main else None,
            wind_speed_ms = round(float(wind.get("speed", 0)), 1),
            humidity_pct  = int(main["humidity"]) if "humidity" in main else None,
            conditions    = weather.get("description", "unknown"),
            station_name  = name,
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Could not parse OpenWeather response: %s", exc)
        return CurrentWeather(error=f"Parse error: {exc}")


# ── Open-Meteo current forecast (free, no API key) ────────────────────────────

_OPENMETEO_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_openmeteo_current(lat: float, lon: float) -> CurrentWeather:
    """
    Fetch current conditions from the Open-Meteo forecast API.

    Free, no API key required.  Variable names and units match the Open-Meteo
    archive used for training (temperature_2m in °C, wind_speed_10m in m/s,
    relative_humidity_2m in %).

    Always returns CurrentWeather; sets .error on failure.
    """
    import requests  # local import — weather_client may be used outside web context

    try:
        resp = requests.get(
            _OPENMETEO_CURRENT_URL,
            params={
                "latitude":        lat,
                "longitude":       lon,
                "current":         "temperature_2m,wind_speed_10m,relative_humidity_2m",
                "wind_speed_unit": "ms",
                "forecast_days":   1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        return CurrentWeather(
            temp_c=        float(cur["temperature_2m"])      if "temperature_2m"      in cur else None,
            wind_speed_ms= float(cur["wind_speed_10m"])      if "wind_speed_10m"      in cur else None,
            humidity_pct=  float(cur["relative_humidity_2m"]) if "relative_humidity_2m" in cur else None,
            conditions=    "current (Open-Meteo)",
            station_name=  "Open-Meteo grid point",
            source=        "Open-Meteo (current forecast)",
        )
    except Exception as exc:
        log.warning("Open-Meteo current fetch failed: %s", exc)
        return CurrentWeather(error=str(exc), source="Open-Meteo (current forecast)")
