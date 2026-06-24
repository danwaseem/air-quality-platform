"""
Cleaning pipeline for EPA AQS hourly air quality data.

Pipeline steps
--------------
1. normalize_schema  — stable station IDs, UTC timestamps, wide pivot, complete hourly grid
2. fill_gaps         — interpolate short gaps (≤ SHORT_GAP_HOURS), flag/leave longer ones
3. flag_outliers     — physically impossible values + rolling z-score drift detection
4. save              — data/processed/clean_hourly.parquet + cleaning_report.json

Units in the cleaned output
---------------------------
  pm25  : µg/m³   (micrograms per cubic metre, LC)
  no2   : ppb      (parts per billion)
  ozone : ppm      (parts per million — AQS native; 0.070 ppm = 70 ppb = EPA 8-hr standard)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

POLLUTANTS = ["pm25", "no2", "ozone"]

PARAM_CODE_MAP: dict[str, str] = {
    "88101": "pm25",
    "42602": "no2",
    "44201": "ozone",
}

# Physically plausible bounds (inclusive) — values outside → NaN + flagged
# PM2.5 capped at 2000 to preserve extreme wildfire events (2023 CA fires hit ~1000)
PHYS_BOUNDS: dict[str, tuple[float, float]] = {
    "pm25":  (0.0, 2000.0),  # µg/m³
    "no2":   (0.0,  500.0),  # ppb
    "ozone": (0.0,    0.6),  # ppm  (0.604 ppm = EPA AQI max breakpoint)
}

SHORT_GAP_HOURS: int = 3       # interpolate gaps of this length or shorter
DRIFT_Z_THRESHOLD: float = 3.5 # rolling z-score threshold for drift detection
DRIFT_WINDOW_HOURS: int = 168  # 7-day window for drift baseline (centred)
DRIFT_MIN_PERIODS: int = 24    # minimum valid readings to compute a z-score


# ── Step 1: Schema normalization ──────────────────────────────────────────────

def normalize_schema(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Produce a wide, hourly-complete DataFrame from raw long-format AQS data.

    Output columns:
        station_id, timestamp_utc, pm25, no2, ozone,
        latitude, longitude, state, county
    """
    n_raw = len(df_raw)
    log.info("Step 1 — Schema normalization  (%d raw rows)", n_raw)

    df = df_raw.copy()

    # -- Station ID: AQS canonical SSCCCNNNN ----------------------------------
    df["station_id"] = (
        df["state_code"].astype(str).str.zfill(2)
        + df["county_code"].astype(str).str.zfill(3)
        + df["site_number"].astype(str).str.zfill(4)
    )

    # -- UTC timestamp --------------------------------------------------------
    df["timestamp_utc"] = pd.to_datetime(
        df["date_gmt"] + " " + df["time_gmt"],
        format="%Y-%m-%d %H:%M",
        utc=True,
        errors="coerce",
    )
    n_bad_ts = int(df["timestamp_utc"].isna().sum())
    if n_bad_ts:
        log.warning("  Dropped %d rows with unparseable GMT timestamp", n_bad_ts)
    df = df.dropna(subset=["timestamp_utc"])

    # -- Parameter filter + mapping -------------------------------------------
    df["parameter_code"] = df["parameter_code"].astype(str).str.strip()
    df = df[df["parameter_code"].isin(PARAM_CODE_MAP)].copy()
    df["pollutant"] = df["parameter_code"].map(PARAM_CODE_MAP)
    n_filtered = len(df)
    log.info(
        "  After parameter filter: %d rows kept, %d dropped (non-target params)",
        n_filtered, n_raw - n_bad_ts - n_filtered,
    )

    # -- Station metadata (first non-null value per station) ------------------
    meta_cols = [c for c in ["latitude", "longitude", "state", "county"] if c in df.columns]
    station_meta = df.groupby("station_id")[meta_cols].first().reset_index()

    # -- Aggregate multiple POC (monitors) at same station+time+pollutant -----
    # Some sites run parallel sensors; take the mean.
    df_agg = (
        df.groupby(["station_id", "timestamp_utc", "pollutant"])["sample_measurement"]
        .mean()
        .reset_index()
    )

    # -- Pivot: long → wide ---------------------------------------------------
    df_wide = df_agg.pivot_table(
        index=["station_id", "timestamp_utc"],
        columns="pollutant",
        values="sample_measurement",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None
    for p in POLLUTANTS:
        if p not in df_wide.columns:
            df_wide[p] = np.nan

    # -- Build complete hourly grid per station --------------------------------
    # AQS can skip hours; we make gaps explicit so interpolation & missingness
    # analysis operate on a fully regular time axis.
    log.info("  Building complete hourly index per station …")
    station_ranges = df_wide.groupby("station_id")["timestamp_utc"].agg(["min", "max"])
    idx_rows: list[tuple[str, pd.Timestamp]] = []
    for sid, row in station_ranges.iterrows():
        for ts in pd.date_range(row["min"], row["max"], freq="h", tz="UTC"):
            idx_rows.append((sid, ts))

    idx_df = pd.DataFrame(idx_rows, columns=["station_id", "timestamp_utc"])
    df_complete = idx_df.merge(df_wide, on=["station_id", "timestamp_utc"], how="left")
    df_complete = df_complete.merge(station_meta, on="station_id", how="left")
    df_complete = df_complete.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    # -- Metrics --------------------------------------------------------------
    n_stations = int(df_complete["station_id"].nunique())
    stations_by_pollutant = {
        p: int(df_complete.groupby("station_id")[p].apply(lambda s: s.notna().any()).sum())
        for p in POLLUTANTS
    }

    metrics: dict[str, Any] = {
        "raw_rows": n_raw,
        "rows_after_ts_parse": n_raw - n_bad_ts,
        "rows_after_param_filter": n_filtered,
        "pivoted_rows": len(df_complete),
        "pivoted_cols": len(df_complete.columns),
        "station_count": n_stations,
        "stations_by_pollutant": stations_by_pollutant,
    }
    log.info(
        "  Pivoted shape: %d rows × %d cols | %d distinct stations",
        len(df_complete), len(df_complete.columns), n_stations,
    )
    for p, cnt in stations_by_pollutant.items():
        log.info("    %-6s  %d stations with ≥1 reading", p, cnt)

    return df_complete, metrics


# ── Missingness helpers ───────────────────────────────────────────────────────

def _missingness_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Per-pollutant missingness, counting only rows that belong to stations
    which actively measure that pollutant (i.e. have at least one non-null).
    """
    stats: dict[str, dict[str, Any]] = {}
    for p in POLLUTANTS:
        active_mask = df.groupby("station_id")[p].transform(lambda s: s.notna().any())
        col = df.loc[active_mask, p]
        total = len(col)
        missing = int(col.isna().sum())
        stats[p] = {
            "total_expected": total,
            "missing_count": missing,
            "missing_pct": round(missing / total * 100, 2) if total else 0.0,
        }
    return stats


# ── Step 2: Gap filling ───────────────────────────────────────────────────────

def fill_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Linear interpolation for gaps ≤ SHORT_GAP_HOURS per station per pollutant.
    Longer gaps stay NaN.  Adds <pollutant>_interp_flag columns (True = interpolated).
    """
    log.info("Step 2 — Gap filling  (short gap ≤ %d h)", SHORT_GAP_HOURS)
    df = df.copy()

    before = _missingness_stats(df)
    log.info("  Missingness BEFORE gap fill:")
    for p, s in before.items():
        log.info("    %-6s  %7d / %7d missing  (%.1f%%)",
                 p, s["missing_count"], s["total_expected"], s["missing_pct"])

    interp_counts: dict[str, int] = {}
    long_gap_counts: dict[str, int] = {}

    station_groups = df.groupby("station_id", sort=False).groups  # {sid: Index}

    for p in POLLUTANTS:
        flag_col = f"{p}_interp_flag"
        df[flag_col] = False
        n_interp = 0
        n_long = 0

        for sid, grp_idx in station_groups.items():
            series = df.loc[grp_idx, p]
            if series.isna().all():
                continue  # station doesn't measure this pollutant

            was_null = series.isna()
            if not was_null.any():
                continue

            # Linear interpolation is equivalent to time-based on a uniform hourly grid
            filled = series.interpolate(method="linear", limit=SHORT_GAP_HOURS)

            newly_filled = was_null & ~filled.isna()
            n_interp += int(newly_filled.sum())

            # Long gap = was null and STILL null after limited interpolation
            still_null = was_null & filled.isna()
            n_long += int(still_null.sum())

            df.loc[grp_idx, p] = filled
            df.loc[grp_idx, flag_col] = newly_filled

        interp_counts[p] = n_interp
        long_gap_counts[p] = n_long
        log.info("  %-6s  interpolated %7d values | %7d remain in long gaps (>%dh)",
                 p, n_interp, n_long, SHORT_GAP_HOURS)

    after = _missingness_stats(df)
    log.info("  Missingness AFTER gap fill:")
    for p, s in after.items():
        log.info("    %-6s  %7d / %7d missing  (%.1f%%)",
                 p, s["missing_count"], s["total_expected"], s["missing_pct"])

    metrics: dict[str, Any] = {
        "before": before,
        "after": after,
        "interpolated": interp_counts,
        "long_gaps_remaining": long_gap_counts,
    }
    return df, metrics


# ── Step 3: Outlier and drift detection ───────────────────────────────────────

def _rolling_zscore_flag(series: pd.Series) -> pd.Series:
    roll = series.rolling(DRIFT_WINDOW_HOURS, center=True, min_periods=DRIFT_MIN_PERIODS)
    mu = roll.mean()
    sigma = roll.std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mu) / sigma
    return (z.abs() > DRIFT_Z_THRESHOLD).fillna(False)


def flag_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    1. Negative values → NaN + <pollutant>_neg_flag
    2. Values outside PHYS_BOUNDS → NaN + <pollutant>_bounds_flag
    3. Rolling 7-day z-score > DRIFT_Z_THRESHOLD → <pollutant>_drift_flag
       (value is preserved — drift is possible, not definitively wrong)
    """
    log.info("Step 3 — Outlier / drift detection")
    df = df.copy()

    neg_counts: dict[str, int] = {}
    bounds_counts: dict[str, int] = {}
    drift_counts: dict[str, int] = {}

    for p in POLLUTANTS:
        lo, hi = PHYS_BOUNDS[p]
        col = df[p]

        # Negative readings
        neg_mask = col.notna() & (col < 0)
        neg_counts[p] = int(neg_mask.sum())
        df[f"{p}_neg_flag"] = neg_mask

        # Beyond physical bounds (includes negatives — reported separately for clarity)
        bounds_mask = col.notna() & ((col < lo) | (col > hi))
        bounds_counts[p] = int(bounds_mask.sum())
        df[f"{p}_bounds_flag"] = bounds_mask
        df.loc[bounds_mask, p] = np.nan  # remove impossible / extreme values

        log.info(
            "  %-6s  %5d negative | %5d beyond bounds [%.4g – %.4g] → set to NaN",
            p, neg_counts[p], bounds_counts[p], lo, hi,
        )

        # Per-station rolling z-score drift (computed on cleaned values)
        drift_flags = (
            df.groupby("station_id", sort=False)[p]
            .transform(_rolling_zscore_flag)
        )
        df[f"{p}_drift_flag"] = drift_flags.fillna(False)
        drift_counts[p] = int(drift_flags.sum())
        log.info("  %-6s  %5d readings flagged as possible drift (|z| > %.1f, 7-day window)",
                 p, drift_counts[p], DRIFT_Z_THRESHOLD)

    metrics: dict[str, Any] = {
        "negative": neg_counts,
        "beyond_bounds": bounds_counts,
        "drift_flagged": drift_counts,
    }
    return df, metrics


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_pipeline(
    raw_path: Path,
    out_data: Path = Path("data/processed/clean_hourly.parquet"),
    out_report: Path = Path("data/processed/cleaning_report.json"),
) -> dict[str, Any]:
    """
    Run all cleaning steps end-to-end.

    Parameters
    ----------
    raw_path  : path to raw Parquet (merged or single chunk)
    out_data  : where to write the cleaned Parquet
    out_report: where to write the JSON metrics report

    Returns the full report dict.
    """
    log.info("=" * 60)
    log.info("Cleaning pipeline start")
    log.info("Input : %s", raw_path)
    log.info("Output: %s", out_data)
    log.info("=" * 60)

    df_raw = pd.read_parquet(raw_path)
    log.info("Loaded raw data: %d rows × %d cols", *df_raw.shape)

    df, schema_metrics = normalize_schema(df_raw)
    df, gap_metrics = fill_gaps(df)
    df, outlier_metrics = flag_outliers(df)

    log.info("-" * 60)
    log.info("Final dataset: %d rows × %d cols", *df.shape)

    out_data.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_data, index=False, compression="snappy")
    log.info("Saved data   → %s  (%.1f MB)", out_data, out_data.stat().st_size / 1e6)

    report: dict[str, Any] = {
        "schema": schema_metrics,
        "missingness": gap_metrics,
        "outliers": outlier_metrics,
        "output": {
            "rows": len(df),
            "columns": list(df.columns),
            "path": str(out_data),
        },
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("Saved report → %s", out_report)
    log.info("=" * 60)
    log.info("Cleaning pipeline complete")

    return report
