#!/usr/bin/env python3
"""
Export pipeline artifacts to BI-tool-friendly CSVs.

Outputs (all written to dashboards/data/):
  air_quality_daily.csv       hourly → daily mean per station (trends dashboard)
  monthly_state_summary.csv   monthly mean per state + active station count
  model_metrics.csv           flat metrics.json (one row per pollutant × model)
  bias_by_state.csv           flat bias_audit.json per-state breakdown
  bias_by_region.csv          flat bias_audit.json per-region breakdown
  station_aqi.csv             per-station 2023 annual average + EPA AQI category
                              for each pollutant (missing stations → "No Data")

Date handling: timestamps are UTC throughout; daily aggregation uses UTC dates.
Seasons: 0 = Winter (DJF), 1 = Spring (MAM), 2 = Summer (JJA), 3 = Fall (SON).

Run from project root:
  python scripts/export_for_dashboards.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
CLEAN_PQ      = ROOT / "data"    / "processed" / "clean_hourly.parquet"
METRICS_JSON  = ROOT / "models"  / "metrics.json"
AUDIT_JSON    = ROOT / "audits"  / "bias_audit.json"
OUT_DIR       = ROOT / "dashboards" / "data"

POLLUTANTS    = ["pm25", "no2", "ozone"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _season(month: pd.Series) -> pd.Series:
    """0=Winter(DJF) 1=Spring(MAM) 2=Summer(JJA) 3=Fall(SON)."""
    return pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=[0, 1, 2, 3, 0],   # DJF wraps: Jan/Feb=0, Dec=0
        ordered=False,
    ).astype(int)


def _write(df: pd.DataFrame, name: str) -> None:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    size_kb = path.stat().st_size / 1024
    print(f"  {name:<35}  {len(df):>8,} rows  {size_kb:>7.1f} KB  → {path}")


# ── AQI breakpoints ───────────────────────────────────────────────────────────
# (threshold, label) pairs in ascending order; the first threshold the value
# does not exceed wins.  NaN/None → "No Data".  Exceeds all thresholds → "Hazardous".

_AQI_BREAKS: dict[str, list[tuple[float, str]]] = {
    "pm25":  [(12.0,  "Good"), (35.4,  "Moderate"), (55.4,   "USG"),
              (150.4, "Unhealthy"), (250.4, "Very Unhealthy")],
    "no2":   [(53.0,  "Good"), (100.0, "Moderate"), (360.0,  "USG"),
              (649.0, "Unhealthy"), (1249.0, "Very Unhealthy")],
    "ozone": [(0.054, "Good"), (0.070, "Moderate"), (0.085,  "USG"),
              (0.105, "Unhealthy"), (0.200, "Very Unhealthy")],
}

_AQI_ORDER = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous", "No Data"]


def _aqi_category(value: float, pollutant: str) -> str:
    if pd.isna(value):
        return "No Data"
    for threshold, label in _AQI_BREAKS[pollutant]:
        if value <= threshold:
            return label
    return "Hazardous"


# ── 1. air_quality_daily.csv ──────────────────────────────────────────────────

def export_daily(df_hourly: pd.DataFrame) -> None:
    """
    Aggregate hourly readings to daily per station.
    Only includes station-days where at least one pollutant has a reading.
    """
    df = df_hourly.copy()

    # UTC date (BI tools want a plain date, not a timezone-aware timestamp)
    df["date"] = df["timestamp_utc"].dt.date.astype(str)
    df["month"]  = df["timestamp_utc"].dt.month
    df["season"] = _season(df["month"])

    # Daily mean of pollutants (NaN rows are ignored by mean())
    agg = (
        df.groupby(["station_id", "date"], sort=True)
        .agg(
            state      = ("state",     "first"),
            county     = ("county",    "first"),
            latitude   = ("latitude",  "first"),
            longitude  = ("longitude", "first"),
            month      = ("month",     "first"),
            season     = ("season",    "first"),
            pm25_mean  = ("pm25",      "mean"),
            no2_mean   = ("no2",       "mean"),
            ozone_mean = ("ozone",     "mean"),
        )
        .reset_index()
    )

    # Drop station-days where all three pollutants are NaN (station not active)
    any_reading = agg[["pm25_mean", "no2_mean", "ozone_mean"]].notna().any(axis=1)
    agg = agg[any_reading].reset_index(drop=True)

    # Round pollutant values to 4 decimal places for compact CSVs
    for col in ["pm25_mean", "no2_mean", "ozone_mean"]:
        agg[col] = agg[col].round(4)

    col_order = [
        "station_id", "date", "state", "county",
        "latitude", "longitude", "month", "season",
        "pm25_mean", "no2_mean", "ozone_mean",
    ]
    _write(agg[col_order], "air_quality_daily.csv")


# ── 2. monthly_state_summary.csv ──────────────────────────────────────────────

def export_monthly_state(df_hourly: pd.DataFrame) -> None:
    """
    Monthly mean pollutant concentrations per state, plus count of
    active stations for each pollutant that month.
    """
    df = df_hourly.copy()
    df["year_month"] = df["timestamp_utc"].dt.tz_localize(None).dt.to_period("M").astype(str)
    df["month"]  = df["timestamp_utc"].dt.month
    df["season"] = _season(df["month"])

    # Monthly mean concentrations per state
    means = (
        df.groupby(["state", "year_month"], sort=True)
        .agg(
            month       = ("month",  "first"),
            season      = ("season", "first"),
            pm25_mean   = ("pm25",   "mean"),
            no2_mean    = ("no2",    "mean"),
            ozone_mean  = ("ozone",  "mean"),
        )
        .reset_index()
    )

    # Active station counts: stations with ≥1 non-null reading that month
    def _active_stations(df_grp: pd.DataFrame, col: str) -> int:
        return int(df_grp.loc[df_grp[col].notna(), "station_id"].nunique())

    station_counts = (
        df.groupby(["state", "year_month"])
        .apply(lambda g: pd.Series({
            "stations_pm25":  _active_stations(g, "pm25"),
            "stations_no2":   _active_stations(g, "no2"),
            "stations_ozone": _active_stations(g, "ozone"),
        }), include_groups=False)
        .reset_index()
    )

    summary = means.merge(station_counts, on=["state", "year_month"], how="left")
    for col in ["pm25_mean", "no2_mean", "ozone_mean"]:
        summary[col] = summary[col].round(4)

    col_order = [
        "state", "year_month", "month", "season",
        "pm25_mean", "no2_mean", "ozone_mean",
        "stations_pm25", "stations_no2", "stations_ozone",
    ]
    _write(summary[col_order], "monthly_state_summary.csv")


# ── 3. model_metrics.csv ─────────────────────────────────────────────────────

def export_model_metrics(metrics: dict) -> None:
    """
    Flatten models/metrics.json to one row per (pollutant, model).
    Includes all three models (GradientBoosting, RandomForest, ARIMA)
    even if some values are null.
    """
    rows = []
    for pollutant, models in metrics.items():
        for model_name, m in models.items():
            rows.append({
                "pollutant": pollutant,
                "model":     model_name,
                "mae":       m.get("mae"),
                "rmse":      m.get("rmse"),
                "r2":        m.get("r2"),
                "n_test":    m.get("n_test"),
            })

    df = pd.DataFrame(rows)
    _write(df, "model_metrics.csv")


# ── 4 & 5. bias_by_state.csv / bias_by_region.csv ────────────────────────────

def export_bias(audit: dict) -> None:
    """
    Flatten audits/bias_audit.json into per-state and per-region CSVs.
    """
    state_rows  = []
    region_rows = []

    for pollutant in POLLUTANTS:
        poll_data = audit.get(pollutant, {})
        if not poll_data:
            continue

        # Per-state
        for state, m in poll_data.get("by_state", {}).items():
            state_rows.append({
                "pollutant":  pollutant,
                "state":      state,
                "region":     m.get("region"),
                "mae":        m.get("mae"),
                "rmse":       m.get("rmse"),
                "r2":         m.get("r2"),
                "n_test":     m.get("n_test"),
                "n_stations": m.get("n_stations"),
                "flagged":    m.get("flagged", False),
            })

        # Per-region
        for region, m in poll_data.get("by_region", {}).items():
            region_rows.append({
                "pollutant":  pollutant,
                "region":     region,
                "states":     ", ".join(m.get("states", [])),
                "mae":        m.get("mae"),
                "rmse":       m.get("rmse"),
                "r2":         m.get("r2"),
                "n_test":     m.get("n_test"),
                "n_stations": m.get("n_stations"),
            })

    _write(pd.DataFrame(state_rows),  "bias_by_state.csv")
    _write(pd.DataFrame(region_rows), "bias_by_region.csv")


# ── 6. station_aqi.csv ───────────────────────────────────────────────────────

def export_station_aqi(df_hourly: pd.DataFrame) -> None:
    """
    One row per station: full-year average concentration + EPA AQI category
    for each pollutant.  Stations with no readings for a pollutant get
    avg = NaN and category = "No Data" so missing data is never miscategorised.
    """
    agg = (
        df_hourly.groupby("station_id", sort=True)
        .agg(
            state     = ("state",     "first"),
            county    = ("county",    "first"),
            latitude  = ("latitude",  "first"),
            longitude = ("longitude", "first"),
            avg_pm25  = ("pm25",      "mean"),
            avg_no2   = ("no2",       "mean"),
            avg_ozone = ("ozone",     "mean"),
        )
        .reset_index()
    )

    for poll, avg_col, cat_col in [
        ("pm25",  "avg_pm25",  "aqi_cat_pm25"),
        ("no2",   "avg_no2",   "aqi_cat_no2"),
        ("ozone", "avg_ozone", "aqi_cat_ozone"),
    ]:
        agg[cat_col] = agg[avg_col].apply(lambda v, p=poll: _aqi_category(v, p))

    for col in ["avg_pm25", "avg_no2", "avg_ozone"]:
        agg[col] = agg[col].round(4)

    col_order = [
        "station_id", "state", "county", "latitude", "longitude",
        "avg_pm25", "avg_no2", "avg_ozone",
        "aqi_cat_pm25", "aqi_cat_no2", "aqi_cat_ozone",
    ]
    _write(agg[col_order], "station_aqi.csv")

    # Print category distribution per pollutant
    for poll, cat_col, unit in [
        ("pm25",  "aqi_cat_pm25",  "µg/m³"),
        ("no2",   "aqi_cat_no2",   "ppb"),
        ("ozone", "aqi_cat_ozone", "ppm"),
    ]:
        counts = agg[cat_col].value_counts()
        print(f"\n  {poll} ({unit}) AQI categories ({counts.sum()} stations):")
        for cat in _AQI_ORDER:
            n = counts.get(cat, 0)
            if n:
                print(f"    {cat:<18} {n:>4}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Pre-flight
    missing = [p for p in [CLEAN_PQ, METRICS_JSON, AUDIT_JSON] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: missing input: {p}")
        print("Run pull_data → clean_data → train_models → bias_audit first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading clean_hourly.parquet …", flush=True)
    df_hourly = pd.read_parquet(CLEAN_PQ)
    print(f"  {len(df_hourly):,} rows · {df_hourly['station_id'].nunique()} stations"
          f" · {df_hourly['state'].nunique()} states\n")

    metrics = json.loads(METRICS_JSON.read_text())
    audit   = json.loads(AUDIT_JSON.read_text())

    print(f"Writing CSVs to {OUT_DIR}/")
    print(f"  {'File':<35}  {'Rows':>8}  {'Size':>8}  Path")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*20}")

    export_daily(df_hourly)
    export_monthly_state(df_hourly)
    export_model_metrics(metrics)
    export_bias(audit)
    export_station_aqi(df_hourly)

    print(f"\nDone. 6 files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
