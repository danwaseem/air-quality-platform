#!/usr/bin/env python3
"""
Geographic bias audit for trained GBM air quality models.

Uses the same time-based test split as training (no leakage).  Measures
MAE / RMSE / R² on the test set broken down by:
  - State  (10 states in the dataset)
  - Census region  (West / South / Northeast / Midwest)

Surfaces:
  - Best and worst region per pollutant
  - MAE spread ratio (worst / best)
  - States flagged as substantially worse than overall average (≥ 1.5× overall MAE)
  - Spearman correlation between per-state sample size (n_test) and MAE
    (tests whether data sparsity predicts model underperformance)

Outputs:
  audits/bias_audit.json   — full per-region metrics + disparity summary
  audits/bias_audit.md     — plain-language audit summary

Run from project root:
  python scripts/bias_audit.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.WARNING)

from src.features.build_features import FEATURE_COLS, POLLUTANTS, build_all_features
from src.models.train import TEST_FRACTION, time_split

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
# Use the weather-augmented parquet so the feature engineering (20 features)
# and test split exactly match the deployed weather-augmented GBM models.
CLEAN_PARQUET = ROOT / "data" / "processed" / "clean_hourly_weather.parquet"
MODELS_DIR    = ROOT / "models"
AUDIT_JSON    = ROOT / "audits" / "bias_audit.json"
AUDIT_MD      = ROOT / "audits" / "bias_audit.md"

# ── Constants ─────────────────────────────────────────────────────────────────

UNITS = {"pm25": "µg/m³", "no2": "ppb", "ozone": "ppm"}

# US Census Bureau regions (only states present in the dataset are needed,
# but the full mapping is included for correctness / future data pulls)
CENSUS_REGIONS: dict[str, str] = {
    # Northeast
    "Connecticut": "Northeast", "Maine": "Northeast", "Massachusetts": "Northeast",
    "New Hampshire": "Northeast", "New Jersey": "Northeast", "New York": "Northeast",
    "Pennsylvania": "Northeast", "Rhode Island": "Northeast", "Vermont": "Northeast",
    # Midwest
    "Illinois": "Midwest", "Indiana": "Midwest", "Iowa": "Midwest",
    "Kansas": "Midwest", "Michigan": "Midwest", "Minnesota": "Midwest",
    "Missouri": "Midwest", "Nebraska": "Midwest", "North Dakota": "Midwest",
    "Ohio": "Midwest", "South Dakota": "Midwest", "Wisconsin": "Midwest",
    # South
    "Alabama": "South", "Arkansas": "South", "Delaware": "South",
    "Florida": "South", "Georgia": "South", "Kentucky": "South",
    "Louisiana": "South", "Maryland": "South", "Mississippi": "South",
    "North Carolina": "South", "Oklahoma": "South", "South Carolina": "South",
    "Tennessee": "South", "Texas": "South", "Virginia": "South",
    "West Virginia": "South",
    # West
    "Alaska": "West", "Arizona": "West", "California": "West", "Colorado": "West",
    "Hawaii": "West", "Idaho": "West", "Montana": "West", "Nevada": "West",
    "New Mexico": "West", "Oregon": "West", "Utah": "West",
    "Washington": "West", "Wyoming": "West",
}

# Flag states whose MAE exceeds this multiple of the overall test MAE
FLAG_THRESHOLD_RATIO: float = 1.5


# ── Metrics helper ────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = int(len(yt))
    if n < 2:
        return {"mae": None, "rmse": None, "r2": None, "n_test": n}
    return {
        "mae":    round(float(mean_absolute_error(yt, yp)), 4),
        "rmse":   round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "r2":     round(float(r2_score(yt, yp)), 4),
        "n_test": n,
    }


# ── Per-pollutant audit ───────────────────────────────────────────────────────

def audit_pollutant(
    pollutant: str,
    df_clean: pd.DataFrame,
    station_meta: pd.DataFrame,
    model: Any,
) -> dict[str, Any]:
    """
    Run the full bias audit for one pollutant.

    Returns a dict with overall metrics, per-state breakdown,
    per-region breakdown, and disparity summary.
    """
    print(f"\n  {pollutant.upper()}  ({UNITS[pollutant]})")

    # ── Features + test split (identical to training) ─────────────────────────
    df_feat = build_all_features(df_clean, pollutant)
    if df_feat.empty:
        print(f"    No features — skipping")
        return {}

    _, df_test, _ = time_split(df_feat, TEST_FRACTION)

    if df_test.empty:
        print(f"    No test rows — skipping")
        return {}

    # ── Predictions ───────────────────────────────────────────────────────────
    X_test = df_test[FEATURE_COLS]
    y_true = df_test["target"].values
    y_pred = model.predict(X_test)

    # ── Attach geography via station_id ───────────────────────────────────────
    test_df = df_test[["station_id", "timestamp_utc"]].copy()
    test_df["y_true"] = y_true
    test_df["y_pred"] = y_pred
    test_df = test_df.merge(station_meta[["station_id", "state", "region"]], on="station_id", how="left")

    n_unmapped = test_df["state"].isna().sum()
    if n_unmapped:
        print(f"    WARNING: {n_unmapped} test rows have no state mapping — dropped")
    test_df = test_df.dropna(subset=["state"])

    # ── Overall metrics ───────────────────────────────────────────────────────
    overall = _metrics(test_df["y_true"].values, test_df["y_pred"].values)
    overall_mae = overall["mae"]
    print(f"    overall → MAE={overall_mae:.4f}  RMSE={overall['rmse']:.4f}"
          f"  R²={overall['r2']:.4f}  n={overall['n_test']:,}")

    # ── Per-state metrics ─────────────────────────────────────────────────────
    state_metrics: dict[str, Any] = {}
    for state, grp in test_df.groupby("state"):
        m = _metrics(grp["y_true"].values, grp["y_pred"].values)
        m["region"] = grp["region"].iloc[0] if "region" in grp.columns else "Unknown"
        m["n_stations"] = int(grp["station_id"].nunique())
        m["flagged"] = (
            m["mae"] is not None
            and overall_mae is not None
            and m["mae"] > overall_mae * FLAG_THRESHOLD_RATIO
        )
        state_metrics[state] = m

    # ── Per-region metrics ────────────────────────────────────────────────────
    region_metrics: dict[str, Any] = {}
    for region, grp in test_df.groupby("region"):
        m = _metrics(grp["y_true"].values, grp["y_pred"].values)
        m["n_stations"] = int(grp["station_id"].nunique())
        m["states"] = sorted(grp["state"].unique().tolist())
        region_metrics[region] = m

    # ── Disparity summary ─────────────────────────────────────────────────────
    valid_states = {s: v for s, v in state_metrics.items() if v["mae"] is not None}
    valid_regions = {r: v for r, v in region_metrics.items() if v["mae"] is not None}

    best_state  = min(valid_states, key=lambda s: valid_states[s]["mae"])
    worst_state = max(valid_states, key=lambda s: valid_states[s]["mae"])
    best_region  = min(valid_regions, key=lambda r: valid_regions[r]["mae"])
    worst_region = max(valid_regions, key=lambda r: valid_regions[r]["mae"])

    state_mae_ratio  = round(valid_states[worst_state]["mae"]  / valid_states[best_state]["mae"],  2)
    region_mae_ratio = round(valid_regions[worst_region]["mae"] / valid_regions[best_region]["mae"], 2)

    flagged_states = [s for s, v in valid_states.items() if v["flagged"]]

    # Spearman correlation: n_test vs MAE across states (sparsity → error?)
    ns    = [valid_states[s]["n_test"] for s in valid_states]
    maes  = [valid_states[s]["mae"]    for s in valid_states]
    rho, p_val = (spearmanr(ns, maes) if len(ns) >= 4
                  else (float("nan"), float("nan")))

    disparity = {
        "best_state":           best_state,
        "best_state_mae":       valid_states[best_state]["mae"],
        "worst_state":          worst_state,
        "worst_state_mae":      valid_states[worst_state]["mae"],
        "state_mae_ratio":      state_mae_ratio,
        "best_region":          best_region,
        "best_region_mae":      valid_regions[best_region]["mae"],
        "worst_region":         worst_region,
        "worst_region_mae":     valid_regions[worst_region]["mae"],
        "region_mae_ratio":     region_mae_ratio,
        "flagged_states":       flagged_states,
        "flag_threshold_ratio": FLAG_THRESHOLD_RATIO,
        "spearman_n_vs_mae": {
            "rho":   round(float(rho), 3) if not np.isnan(rho) else None,
            "p_val": round(float(p_val), 4) if not np.isnan(p_val) else None,
            "interpretation": _spearman_interp(rho, p_val),
        },
    }

    print(f"    best  state:  {best_state} (MAE={valid_states[best_state]['mae']:.4f})")
    print(f"    worst state:  {worst_state} (MAE={valid_states[worst_state]['mae']:.4f})")
    print(f"    state MAE ratio (worst/best): {state_mae_ratio:.2f}×")
    if flagged_states:
        print(f"    FLAGGED (>{FLAG_THRESHOLD_RATIO:.0%} above overall): {', '.join(flagged_states)}")
    print(f"    spearman ρ(n_test, MAE): {rho:.3f}  p={p_val:.4f}  "
          f"→ {_spearman_interp(rho, p_val)}")

    return {
        "overall":    overall,
        "by_state":   state_metrics,
        "by_region":  region_metrics,
        "disparity":  disparity,
    }


def _spearman_interp(rho: float, p_val: float) -> str:
    if np.isnan(rho):
        return "insufficient data"
    sig = p_val < 0.05
    direction = "negative" if rho < 0 else "positive"
    strength = "strong" if abs(rho) >= 0.6 else "moderate" if abs(rho) >= 0.3 else "weak"
    sig_str = "significant" if sig else "not significant"
    return f"{strength} {direction} correlation ({sig_str} at p<0.05)"


# ── Table printer ─────────────────────────────────────────────────────────────

def _print_region_table(pollutant: str, result: dict[str, Any]) -> None:
    if not result:
        return
    overall_mae = result["overall"]["mae"]
    print(f"\n  {'─'*72}")
    print(f"  {pollutant.upper()}  ({UNITS[pollutant]})  —  overall MAE={overall_mae:.4f}")
    print(f"  {'─'*72}")

    # State table
    print(f"\n  {'State':<20} {'Region':<12} {'MAE':>8} {'RMSE':>8} {'R²':>7} "
          f"{'n_test':>8} {'n_sta':>6}  Flag")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6}  {'-'*4}")

    by_state = result["by_state"]
    for state in sorted(by_state, key=lambda s: (by_state[s].get("mae") or 999)):
        m = by_state[state]
        mae_s  = f"{m['mae']:.4f}"  if m["mae"]  is not None else "  n/a"
        rmse_s = f"{m['rmse']:.4f}" if m["rmse"] is not None else "  n/a"
        r2_s   = f"{m['r2']:.4f}"   if m["r2"]   is not None else "  n/a"
        flag   = "⚑" if m.get("flagged") else ""
        print(f"  {state:<20} {m['region']:<12} {mae_s:>8} {rmse_s:>8} {r2_s:>7} "
              f"{m['n_test']:>8,} {m['n_stations']:>6}  {flag}")

    # Region summary
    print(f"\n  {'Census Region':<14} {'MAE':>8} {'RMSE':>8} {'R²':>7} "
          f"{'n_test':>8} {'n_sta':>6}  States")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6}  {'-'*30}")
    by_region = result["by_region"]
    for region in sorted(by_region, key=lambda r: (by_region[r].get("mae") or 999)):
        m = by_region[region]
        mae_s  = f"{m['mae']:.4f}"  if m["mae"]  is not None else "  n/a"
        rmse_s = f"{m['rmse']:.4f}" if m["rmse"] is not None else "  n/a"
        r2_s   = f"{m['r2']:.4f}"   if m["r2"]   is not None else "  n/a"
        states = ", ".join(m["states"])
        print(f"  {region:<14} {mae_s:>8} {rmse_s:>8} {r2_s:>7} "
              f"{m['n_test']:>8,} {m['n_stations']:>6}  {states}")

    d = result["disparity"]
    print(f"\n  State MAE spread:   {d['worst_state']} ({d['worst_state_mae']:.4f})"
          f" / {d['best_state']} ({d['best_state_mae']:.4f})"
          f"  = {d['state_mae_ratio']:.2f}×")
    print(f"  Region MAE spread:  {d['worst_region']} ({d['worst_region_mae']:.4f})"
          f" / {d['best_region']} ({d['best_region_mae']:.4f})"
          f"  = {d['region_mae_ratio']:.2f}×")
    sp = d["spearman_n_vs_mae"]
    rho_s = f"{sp['rho']:.3f}" if sp["rho"] is not None else "n/a"
    print(f"  ρ(n_test, MAE):     {rho_s}  →  {sp['interpretation']}")
    if d["flagged_states"]:
        print(f"  ⚑ Flagged states:  {', '.join(d['flagged_states'])}"
              f"  (MAE > {FLAG_THRESHOLD_RATIO:.0%} × overall)")


# ── Markdown writer ───────────────────────────────────────────────────────────

def _write_markdown(audit: dict[str, Any]) -> None:
    lines: list[str] = []

    lines += [
        "# Air Quality GBM Model — Geographic Bias Audit",
        "",
        "> **Scope:** GBM (HistGradientBoostingRegressor) predictions on the held-out test set,",
        "> using the same time-based 80/20 split as training (no temporal leakage).",
        "> 10 US states, 2023, hourly cadence.",
        "> This document measures and documents disparities; it does not claim they have been eliminated.",
        "",
        "## Summary of Findings",
        "",
    ]

    # Quick summary table
    lines += [
        "| Pollutant | Overall MAE | State MAE range | Ratio | Worst state | Flagged |",
        "|-----------|-------------|-----------------|-------|-------------|---------|",
    ]
    for poll in POLLUTANTS:
        r = audit.get(poll, {})
        if not r:
            lines.append(f"| {poll} | — | — | — | — | — |")
            continue
        d  = r["disparity"]
        o  = r["overall"]
        unit = UNITS[poll]
        lines.append(
            f"| {poll} ({unit}) | {o['mae']:.4f} | "
            f"{d['best_state_mae']:.4f} – {d['worst_state_mae']:.4f} | "
            f"{d['state_mae_ratio']:.2f}× | {d['worst_state']} | "
            f"{', '.join(d['flagged_states']) or 'none'} |"
        )

    lines += ["", "---", ""]

    for poll in POLLUTANTS:
        r = audit.get(poll, {})
        if not r:
            continue
        d   = r["disparity"]
        o   = r["overall"]
        sp  = d["spearman_n_vs_mae"]
        unit = UNITS[poll]

        lines += [
            f"## {poll.upper()} ({unit})",
            "",
            f"**Overall test MAE:** {o['mae']:.4f} · RMSE: {o['rmse']:.4f} · R²: {o['r2']:.4f} · n={o['n_test']:,}",
            "",
            "### By Census Region",
            "",
            "| Region | MAE | RMSE | R² | n_test | n_stations | States |",
            "|--------|-----|------|----|--------|------------|--------|",
        ]
        by_region = r["by_region"]
        for region in sorted(by_region, key=lambda rg: (by_region[rg].get("mae") or 999)):
            m = by_region[region]
            mae_s = f"{m['mae']:.4f}" if m["mae"] is not None else "n/a"
            rmse_s = f"{m['rmse']:.4f}" if m["rmse"] is not None else "n/a"
            r2_s = f"{m['r2']:.4f}" if m["r2"] is not None else "n/a"
            lines.append(
                f"| {region} | {mae_s} | {rmse_s} | {r2_s} | "
                f"{m['n_test']:,} | {m['n_stations']} | {', '.join(m['states'])} |"
            )

        lines += [
            "",
            "### By State",
            "",
            "| State | Region | MAE | RMSE | R² | n_test | n_stations | Flagged |",
            "|-------|--------|-----|------|----|--------|------------|---------|",
        ]
        by_state = r["by_state"]
        for state in sorted(by_state, key=lambda s: (by_state[s].get("mae") or 999)):
            m = by_state[state]
            mae_s = f"{m['mae']:.4f}" if m["mae"] is not None else "n/a"
            rmse_s = f"{m['rmse']:.4f}" if m["rmse"] is not None else "n/a"
            r2_s = f"{m['r2']:.4f}" if m["r2"] is not None else "n/a"
            flag_s = "⚑ yes" if m.get("flagged") else "no"
            lines.append(
                f"| {state} | {m['region']} | {mae_s} | {rmse_s} | {r2_s} | "
                f"{m['n_test']:,} | {m['n_stations']} | {flag_s} |"
            )

        # Disparity narrative
        rho_s = f"{sp['rho']:.3f}" if sp["rho"] is not None else "n/a"
        flagged_str = (
            f"**{', '.join(d['flagged_states'])}** (MAE ≥ {FLAG_THRESHOLD_RATIO:.0%}× overall average)"
            if d["flagged_states"] else "none"
        )
        lines += [
            "",
            "### Disparity Analysis",
            "",
            f"- **Best state:** {d['best_state']} (MAE = {d['best_state_mae']:.4f})",
            f"- **Worst state:** {d['worst_state']} (MAE = {d['worst_state_mae']:.4f})",
            f"- **State MAE spread ratio (worst/best):** {d['state_mae_ratio']:.2f}×",
            f"- **Best region:** {d['best_region']} (MAE = {d['best_region_mae']:.4f})",
            f"- **Worst region:** {d['worst_region']} (MAE = {d['worst_region_mae']:.4f})",
            f"- **Region MAE spread ratio:** {d['region_mae_ratio']:.2f}×",
            f"- **Flagged states:** {flagged_str}",
            f"- **Spearman ρ(n\\_test, MAE) across states:** {rho_s} — {sp['interpretation']}",
            "",
        ]

    # Equity interpretation section
    lines += [
        "---",
        "",
        "## Equity Interpretation",
        "",
        "### What this audit measures",
        "Geographic disparity in model accuracy is a real equity concern: communities in",
        "regions where the model performs poorly receive less reliable air quality forecasts,",
        "which can affect health-protective decisions (outdoor activity, ventilation, alerts).",
        "",
        "### Data sparsity and its role",
        "The Spearman ρ values above test whether states with more test data tend to have",
        "lower error.  A significant negative correlation (ρ < 0, p < 0.05) would confirm",
        "that data-sparse states are systematically underserved.  The values above should",
        "be interpreted in this light.",
        "",
        "### Likely causes of regional disparity",
        "- **Pollution regime differences:** Ozone in the West (CA, AZ) peaks in summer",
        "  and is driven by photochemistry + topography — patterns that may generalise",
        "  poorly to Southern or Midwestern ozone dynamics.",
        "- **PM2.5 episodic events:** California wildfire smoke creates extreme PM2.5",
        "  spikes that a 24h-ahead model trained mostly on background conditions will",
        "  underpredict.  This inflates CA error.",
        "- **NO₂ urban vs suburban mix:** NO₂ is highly local (traffic/industrial point",
        "  sources).  Dense urban networks (NY, IL) provide tighter spatial coverage;",
        "  sparser states have higher station-to-station variance that the model cannot",
        "  resolve with lat/lon alone.",
        "- **Sample size imbalance:** States like California (many stations, many rows)",
        "  may dominate model training, leading to better fit in CA and relatively worse",
        "  fit in states with fewer training examples.",
        "",
        "### Limitations of this audit",
        "- Only 10 states are covered; findings do not generalise to unrepresented regions.",
        "- Census regions group states with very different pollution climates",
        "  (e.g. West includes urban CA and rural AZ).",
        "- Station-level disparity (urban vs rural within a state) is not captured here.",
        "- A 1-year dataset (2023) may not represent long-run regional patterns.",
        "",
        "### What was not done",
        "- No demographic overlay (EJ communities, income, race) — this would be a",
        "  necessary next step for a full environmental justice audit.",
        "- No temporal breakdown (does accuracy drop in wildfire season?  winter inversions?).",
        "",
        f"*Generated by scripts/bias_audit.py · model: GBM weather-augmented (20 features) · "
        f"data: clean_hourly_weather.parquet · "
        f"test fraction: {TEST_FRACTION} · flag threshold: {FLAG_THRESHOLD_RATIO:.0%}× overall MAE*",
    ]

    AUDIT_MD.write_text("\n".join(lines))
    print(f"\nMarkdown → {AUDIT_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Pre-flight
    if not CLEAN_PARQUET.exists():
        sys.exit(f"ERROR: {CLEAN_PARQUET} not found — run src/ingestion/historical_weather.py first.")
    missing_models = [p for p in POLLUTANTS
                      if not (MODELS_DIR / f"gbm_{p}.joblib").exists()]
    if missing_models:
        sys.exit(f"ERROR: GBM models missing for {missing_models} — run scripts/train_models.py first.")

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data …", flush=True)
    df_clean = pd.read_parquet(CLEAN_PARQUET)

    # Build station → state + region lookup (one row per station_id)
    station_meta = (
        df_clean.groupby("station_id")[["state"]]
        .first()
        .reset_index()
    )
    station_meta["region"] = station_meta["state"].map(CENSUS_REGIONS).fillna("Unknown")

    unmapped = station_meta[station_meta["region"] == "Unknown"]["state"].unique()
    if len(unmapped):
        print(f"WARNING: states not mapped to Census region: {list(unmapped)}")

    print(
        f"Dataset: {len(df_clean):,} rows · "
        f"{df_clean['station_id'].nunique()} stations · "
        f"{df_clean['state'].nunique()} states\n"
    )
    print("States → regions:")
    for _, row in station_meta.drop_duplicates("state").sort_values("state").iterrows():
        print(f"  {row['state']:<20} → {row['region']}")

    print("\nLoading GBM models …")
    models = {p: joblib.load(MODELS_DIR / f"gbm_{p}.joblib") for p in POLLUTANTS}

    print("\nRunning per-pollutant audit (test split: last"
          f" {TEST_FRACTION:.0%} of timeline) …")

    audit: dict[str, Any] = {}
    for pollutant in POLLUTANTS:
        audit[pollutant] = audit_pollutant(
            pollutant, df_clean, station_meta, models[pollutant]
        )

    # Print full tables
    print("\n\n" + "=" * 74)
    print("  GEOGRAPHIC BIAS AUDIT — FULL TABLES")
    print("=" * 74)
    for poll in POLLUTANTS:
        _print_region_table(poll, audit[poll])

    # Save JSON
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nJSON   → {AUDIT_JSON}")

    # Write markdown
    _write_markdown(audit)


if __name__ == "__main__":
    main()
