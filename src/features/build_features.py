"""
Feature engineering for next-24-hour air quality prediction.

For each (station, pollutant) time series the following features are built:

  Lag features      : lag_1h, lag_24h, lag_168h (1 h, 24 h, 1 week)
  Rolling stats     : 24 h and 168 h mean & std (anchored on past values — no leakage)
  Cyclical time     : hour-of-day, day-of-week, month (sin/cos encoding)
  Season            : 0=winter, 1=spring, 2=summer, 3=fall
  Geography         : station latitude, longitude
  Quality flag      : drift_flag (int) for the target pollutant
  Weather (optional): temperature_2m (°C), wind_speed_10m (m/s),
                      relative_humidity_2m (%) — current-hour values.
                      Taken from clean_hourly_weather.parquet if present;
                      set to NaN if the column is absent in the input
                      (HistGBR handles NaN natively — no leakage: weather at
                      time t is a valid input for predicting pollutant at t+24h).

Target: pollutant value 24 hours ahead (shift -24).

NaN handling
------------
- Essential lags (lag_1h, lag_24h) missing → row dropped.
- lag_168h missing (first week) → imputed from roll_168h_mean if available, else dropped.
- Target missing → row dropped.
- Weather values missing (~0.5% of rows from Open-Meteo gaps) → row dropped;
  count is logged at INFO level per station and summarised in build_all_features.
- Stations with fewer than `min_readings` valid measurements → skipped entirely.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POLLUTANTS: list[str] = ["pm25", "no2", "ozone"]

# Weather features added in the second training round (Open-Meteo historical).
# Placed at the end so models trained WITHOUT weather remain compatible via
# NaN-passthrough when these columns are absent from the input DataFrame.
WEATHER_COLS: list[str] = [
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
]

# Canonical ordered feature column list used by all downstream models.
# 17 base features + 3 weather = 20 total.
FEATURE_COLS: list[str] = [
    "lag_1h", "lag_24h", "lag_168h",
    "roll_24h_mean", "roll_24h_std",
    "roll_168h_mean", "roll_168h_std",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "season",
    "latitude", "longitude",
    "drift_flag",
    # weather (NaN when not available; HistGBR handles gracefully)
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
]

# Columns required to be non-null after engineering; rows missing any are dropped
_REQUIRED_COLS: list[str] = ["target", "lag_1h", "lag_24h"]


def _cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a periodic feature as (sin, cos) pair."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def engineer_station_features(
    df_station: pd.DataFrame,
    pollutant: str,
) -> pd.DataFrame:
    """
    Build a feature-matrix row for every hour in one station's time series.

    Parameters
    ----------
    df_station : DataFrame for a single station, sorted by timestamp_utc,
                 with a contiguous hourly index (as produced by the cleaning pipeline).
    pollutant  : one of "pm25", "no2", "ozone"

    Returns
    -------
    DataFrame with FEATURE_COLS + [station_id, timestamp_utc, target].
    Rows with missing essential features are already dropped.
    """
    df = df_station.sort_values("timestamp_utc").reset_index(drop=True)
    series = df[pollutant]
    ts = df["timestamp_utc"]

    out = pd.DataFrame(index=df.index)
    out["station_id"] = df["station_id"].values
    out["timestamp_utc"] = ts  # Series assignment preserves datetime64[us, UTC] dtype

    # ── Lag features ─────────────────────────────────────────────────────────
    out["lag_1h"] = series.shift(1)
    out["lag_24h"] = series.shift(24)
    out["lag_168h"] = series.shift(168)

    # ── Rolling statistics (shift(1) anchors window on PAST values) ───────────
    s_past = series.shift(1)
    for w, label in ((24, "24h"), (168, "168h")):
        min_p = w // 2  # require at least half the window to compute
        out[f"roll_{label}_mean"] = s_past.rolling(w, min_periods=min_p).mean()
        out[f"roll_{label}_std"] = s_past.rolling(w, min_periods=min_p).std()

    # ── Cyclical time features ────────────────────────────────────────────────
    out["hour_sin"], out["hour_cos"] = _cyclical(ts.dt.hour, 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(ts.dt.dayofweek, 7)
    out["month_sin"], out["month_cos"] = _cyclical(ts.dt.month, 12)
    out["season"] = ((ts.dt.month - 1) // 3).astype(int)

    # ── Station geography ─────────────────────────────────────────────────────
    out["latitude"] = df["latitude"].values
    out["longitude"] = df["longitude"].values

    # ── Quality flag ─────────────────────────────────────────────────────────
    out["drift_flag"] = df[f"{pollutant}_drift_flag"].astype(int).values

    # ── Weather features (current-hour; NaN when column absent in input) ──────
    # Using weather at time t to predict t+24h is valid — no temporal leakage.
    for wx_col in WEATHER_COLS:
        out[wx_col] = df[wx_col].values if wx_col in df.columns else float("nan")

    # ── Target: value 24 h ahead ──────────────────────────────────────────────
    out["target"] = series.shift(-24)

    # ── Impute lag_168h from rolling mean where it's NaN (first ~7 days) ─────
    na_168 = out["lag_168h"].isna()
    out.loc[na_168, "lag_168h"] = out.loc[na_168, "roll_168h_mean"]

    # ── Drop rows missing any required or feature column ─────────────────────
    # Log weather-specific drops separately so callers can see the weather impact.
    drop_cols = list(dict.fromkeys(_REQUIRED_COLS + FEATURE_COLS))  # deduped, ordered
    n_before = len(out)

    # Count rows that would survive without weather (for logging purposes)
    non_wx_drop = list(dict.fromkeys(_REQUIRED_COLS + [c for c in FEATURE_COLS if c not in WEATHER_COLS]))
    n_without_wx_drop = len(out.dropna(subset=non_wx_drop))

    out = out.dropna(subset=drop_cols).reset_index(drop=True)
    n_dropped       = n_before - len(out)
    n_wx_dropped    = n_without_wx_drop - len(out)   # extra rows lost due to weather NaN

    if n_dropped:
        log.debug(
            "  station %s / %s: dropped %d / %d rows "
            "(%d weather-null, %d other NaN)",
            df["station_id"].iloc[0], pollutant,
            n_dropped, n_before, max(n_wx_dropped, 0),
            n_dropped - max(n_wx_dropped, 0),
        )

    return out


def build_all_features(
    df: pd.DataFrame,
    pollutant: str,
    min_readings: int = 200,
) -> pd.DataFrame:
    """
    Build features for all stations that actively measure `pollutant`.

    Parameters
    ----------
    df           : full cleaned DataFrame (all stations, all hours)
    pollutant    : one of "pm25", "no2", "ozone"
    min_readings : stations with fewer valid readings are skipped

    Returns
    -------
    DataFrame sorted by timestamp_utc with FEATURE_COLS + [station_id, timestamp_utc, target].
    Logs usable row count and station count.
    """
    parts: list[pd.DataFrame] = []
    n_skipped = 0
    has_weather = all(c in df.columns for c in WEATHER_COLS)

    for sid, grp in df.groupby("station_id", sort=False):
        valid = grp[pollutant].notna().sum()
        if valid < min_readings:
            n_skipped += 1
            continue
        feat_df = engineer_station_features(grp, pollutant)
        if not feat_df.empty:
            parts.append(feat_df)

    if not parts:
        log.warning("%-6s: no usable stations found (min_readings=%d)", pollutant, min_readings)
        return pd.DataFrame(columns=FEATURE_COLS + ["station_id", "timestamp_utc", "target"])

    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    # Log weather-null row impact when weather columns were present in input
    if has_weather:
        wx_null = result[WEATHER_COLS].isna().any(axis=1).sum()
        if wx_null > 0:
            log.info(
                "%-6s: %d rows have null weather features "
                "(already dropped — Open-Meteo gaps)",
                pollutant, wx_null,
            )
        log.info(
            "%-6s: %8d usable rows  |  %3d stations  |  %d skipped (<200 readings)"
            "  |  weather: present (%.2f%% null dropped)",
            pollutant, len(result), len(parts), n_skipped,
            result[WEATHER_COLS].isna().any(axis=1).mean() * 100,
        )
    else:
        log.info(
            "%-6s: %8d usable rows  |  %3d stations  |  %d skipped (<200 readings)"
            "  |  weather: absent (NaN passthrough)",
            pollutant, len(result), len(parts), n_skipped,
        )

    return result
