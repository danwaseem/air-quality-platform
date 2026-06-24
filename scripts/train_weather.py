#!/usr/bin/env python3
"""
Retrain GBM models with weather features and produce an ablation comparison.

What this script does
---------------------
1. Reads the BASELINE metrics from models/metrics.json (previous run, 17 features,
   no weather) — saved before any files are overwritten.
2. Trains GBM on data/processed/clean_hourly_weather.parquet (20 features:
   the original 17 + temperature_2m, wind_speed_10m, relative_humidity_2m).
   Rows with null weather (~0.55%) are dropped; the count is logged.
3. Saves new model artifacts over the originals:
     models/gbm_{pollutant}.joblib
     models/metrics.json
4. Writes models/weather_ablation.json — side-by-side baseline vs. augmented.
5. Prints a comparison table and an honest verdict per pollutant.

Fairness guarantees
-------------------
- Same seed=42 → identical random initialisation.
- Same TEST_FRACTION=0.2 and time-based split logic → identical train/test cutoff.
- Only GBM is retrained (RF is too large to store and wasn't changed; ARIMA is
  univariate and weather features don't apply to it).

Run from project root:
  python scripts/train_weather.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.features.build_features import (
    FEATURE_COLS, POLLUTANTS, WEATHER_COLS, build_all_features,
)
from src.models.train import (
    GBM_PARAMS, TEST_FRACTION, _metrics, time_split, train_gbm,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
WEATHER_PQ  = ROOT / "data" / "processed" / "clean_hourly_weather.parquet"
MODELS_DIR  = ROOT / "models"
METRICS_JSON  = MODELS_DIR / "metrics.json"
ABLATION_JSON = MODELS_DIR / "weather_ablation.json"

SEED = 42
IMPROVE_THRESHOLD = 0.005   # >0.5% MAE change counts as meaningful


# ── Ablation helpers ──────────────────────────────────────────────────────────

def _verdict(base_mae: float | None, aug_mae: float | None) -> str:
    if base_mae is None or aug_mae is None:
        return "unknown"
    delta_pct = (aug_mae - base_mae) / base_mae
    if delta_pct < -IMPROVE_THRESHOLD:
        return "improved"
    if delta_pct > IMPROVE_THRESHOLD:
        return "degraded"
    return "no change"


def _delta_str(base: float | None, aug: float | None, higher_better: bool = False) -> str:
    """Return a signed delta string with direction arrow."""
    if base is None or aug is None:
        return "  n/a"
    d = aug - base
    if higher_better:
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
    else:
        arrow = "↓" if d < 0 else ("↑" if d > 0 else "→")
    return f"{d:+.4f} {arrow}"


# ── Print table ───────────────────────────────────────────────────────────────

def _print_table(ablation: dict) -> None:
    W = 76
    print("\n" + "=" * W)
    print("  Weather Ablation — Baseline (17 feat) vs. Augmented (20 feat, +weather)")
    print("=" * W)

    header = (
        f"  {'Pollutant':<8}  {'Metric':<6}  "
        f"{'Baseline':>10}  {'Augmented':>10}  {'Delta':>12}  Verdict"
    )
    print(header)
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

    verdicts = {}
    for poll in POLLUTANTS:
        data  = ablation["pollutants"][poll]
        base  = data["baseline"]
        aug   = data["augmented"]
        v     = data["verdict"]
        verdicts[poll] = v

        v_str = {"improved": "✓ improved", "degraded": "✗ degraded",
                 "no change": "~ no change"}.get(v, v)

        for metric, higher_better in [("mae", False), ("rmse", False), ("r2", True)]:
            b_val = base.get(metric)
            a_val = aug.get(metric)
            b_s   = f"{b_val:.4f}" if b_val is not None else "  n/a"
            a_s   = f"{a_val:.4f}" if a_val is not None else "  n/a"
            d_s   = _delta_str(b_val, a_val, higher_better)
            verdict_col = v_str if metric == "mae" else ""
            print(f"  {poll:<8}  {metric:<6}  {b_s:>10}  {a_s:>10}  {d_s:>12}  {verdict_col}")

        n_b = base.get("n_test", 0)
        n_a = aug.get("n_test", 0)
        print(f"  {poll:<8}  {'n_test':<6}  {n_b:>10,}  {n_a:>10,}  "
              f"  {n_a - n_b:>+7,}     (rows dropped for wx nulls)")
        print()

    print("=" * W)
    print(f"  Summary  (threshold: >{IMPROVE_THRESHOLD*100:.1f}% MAE change to flag)")
    for poll, v in verdicts.items():
        symbol = {"improved": "✓", "degraded": "✗", "no change": "~"}.get(v, "?")
        print(f"    {symbol}  {poll:<6}  {v}")
    print("=" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not WEATHER_PQ.exists():
        sys.exit(
            f"ERROR: {WEATHER_PQ} not found.\n"
            "Run: python src/ingestion/historical_weather.py"
        )
    if not METRICS_JSON.exists():
        sys.exit(
            f"ERROR: {METRICS_JSON} not found.\n"
            "Run: python scripts/train_models.py first to establish a baseline."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save baseline metrics before anything is overwritten ─────────────────
    baseline_metrics = json.loads(METRICS_JSON.read_text())
    log.info("Baseline metrics loaded from %s", METRICS_JSON)

    # ── Load weather-augmented data ───────────────────────────────────────────
    log.info("Loading %s …", WEATHER_PQ.name)
    df = pd.read_parquet(WEATHER_PQ)
    log.info("  shape: %s  |  weather cols present: %s",
             df.shape, [c for c in WEATHER_COLS if c in df.columns])

    wx_null_pct = df[WEATHER_COLS].isna().any(axis=1).mean() * 100
    log.info("  rows with any null weather: %.2f%%", wx_null_pct)

    # ── Train per pollutant ───────────────────────────────────────────────────
    augmented_metrics: dict[str, dict] = {}
    np.random.seed(SEED)

    for pollutant in POLLUTANTS:
        log.info("=" * 60)
        log.info("Training  %s  (weather-augmented, %d features)",
                 pollutant.upper(), len(FEATURE_COLS))
        log.info("=" * 60)

        df_feat = build_all_features(df, pollutant)
        if df_feat.empty:
            log.error("No features for %s — skipping", pollutant)
            augmented_metrics[pollutant] = {}
            continue

        df_train, df_test, cutoff_ts = time_split(df_feat, TEST_FRACTION)
        X_train = df_train[FEATURE_COLS]
        y_train = df_train["target"]
        X_test  = df_test[FEATURE_COLS]
        y_test  = df_test["target"]

        log.info("  Training GBM (%s) …", pollutant)
        gbm   = train_gbm(X_train, y_train, seed=SEED)
        preds = gbm.predict(X_test)
        m     = _metrics(y_test.values, preds)
        augmented_metrics[pollutant] = {"GradientBoosting": m}

        log.info("  GBM %s  MAE=%.4f  RMSE=%.4f  R²=%.4f  n=%d",
                 pollutant, m["mae"], m["rmse"], m["r2"], m["n_test"])

        # Save model (overwrites previous)
        out_path = MODELS_DIR / f"gbm_{pollutant}.joblib"
        joblib.dump(gbm, out_path)
        log.info("  Saved → %s  (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)

    # ── Merge metrics.json: keep RF/ARIMA from baseline, update GBM ──────────
    merged_metrics = {}
    for poll in POLLUTANTS:
        merged_metrics[poll] = {
            **baseline_metrics.get(poll, {}),          # keep RF, ARIMA
            **augmented_metrics.get(poll, {}),          # overwrite GBM
        }

    METRICS_JSON.write_text(json.dumps(merged_metrics, indent=2))
    log.info("Updated metrics → %s", METRICS_JSON)

    # ── Ablation comparison ───────────────────────────────────────────────────
    ablation: dict = {
        "config": {
            "baseline_features":   17,
            "augmented_features":  len(FEATURE_COLS),
            "added_features":      WEATHER_COLS,
            "baseline_data":       "data/processed/clean_hourly.parquet",
            "augmented_data":      str(WEATHER_PQ.relative_to(ROOT)),
            "seed":                SEED,
            "test_fraction":       TEST_FRACTION,
            "improve_threshold_pct": IMPROVE_THRESHOLD * 100,
            "note": (
                "Rows with null weather (~0.55%) are dropped before training. "
                "n_test difference reflects this smaller effective dataset. "
                "Same time-based cutoff logic ensures temporal comparability."
            ),
        },
        "pollutants": {},
    }

    for poll in POLLUTANTS:
        base = baseline_metrics.get(poll, {}).get("GradientBoosting", {})
        aug  = augmented_metrics.get(poll, {}).get("GradientBoosting", {})

        delta = {}
        for k in ("mae", "rmse", "r2"):
            if base.get(k) is not None and aug.get(k) is not None:
                delta[k] = round(aug[k] - base[k], 4)

        ablation["pollutants"][poll] = {
            "baseline":  base,
            "augmented": aug,
            "delta":     delta,
            "verdict":   _verdict(base.get("mae"), aug.get("mae")),
        }

    ABLATION_JSON.write_text(json.dumps(ablation, indent=2))
    log.info("Ablation saved → %s", ABLATION_JSON)

    _print_table(ablation)

    print(f"\nArtifacts updated:")
    print(f"  models/gbm_{{pm25,no2,ozone}}.joblib  — weather-augmented GBM models")
    print(f"  {METRICS_JSON.name}              — GBM metrics updated; RF/ARIMA unchanged")
    print(f"  {ABLATION_JSON.name}     — full ablation comparison")


if __name__ == "__main__":
    main()
