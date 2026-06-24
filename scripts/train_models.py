#!/usr/bin/env python3
"""
Train and compare GradientBoosting, RandomForest, and ARIMA models
for next-24-hour PM2.5, NO2, and ozone prediction.

Reads data/processed/clean_hourly.parquet.
Saves model artifacts + metrics.json + model_card.md to models/.

Usage
-----
    python scripts/train_models.py
    python scripts/train_models.py --input data/processed/clean_hourly.parquet
    python scripts/train_models.py --pollutant pm25          # single pollutant
    python scripts/train_models.py --arima-station 482011039
    python scripts/train_models.py --seed 42 --output models/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import FEATURE_COLS, POLLUTANTS
from src.models.train import (
    ARIMA_ORDER,
    GBM_PARAMS,
    RF_PARAMS,
    TEST_FRACTION,
    run_training,
    train_pollutant,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path,
                   default=Path("data/processed/clean_hourly.parquet"))
    p.add_argument("--output", type=Path, default=Path("models"))
    p.add_argument("--pollutant", choices=POLLUTANTS + ["all"], default="all",
                   help="Train one pollutant or all (default: all)")
    p.add_argument("--arima-station", default="482011039",
                   help="Station ID for ARIMA baseline (default: 482011039 Houston TX)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def print_comparison_table(metrics: dict) -> None:
    """Print a formatted comparison table of all model results."""
    pollutants = list(metrics.keys())
    models = ["GradientBoosting", "RandomForest", "ARIMA"]

    # Header
    col_w = 22
    print("\n" + "=" * 80)
    print("MODEL COMPARISON  —  next-24-hour prediction")
    print("=" * 80)
    header = f"{'':10}" + "".join(f"{'MAE':>8}{'RMSE':>8}{'R²':>7}" for _ in models)
    subhdr = f"{'':10}" + "".join(f"  {m:<21}" for m in models)
    print(subhdr)
    print(f"{'Pollutant':<10}" + "".join(
        f"{'MAE':>7}{'RMSE':>7}{'R²':>7}  " for _ in models
    ))
    print("-" * 80)

    units = {"pm25": "µg/m³", "no2": "ppb", "ozone": "ppm"}
    for p in pollutants:
        row = f"{p + ' (' + units.get(p,'') + ')':<16}"
        for m in models:
            md = metrics.get(p, {}).get(m, {})
            if md.get("mae") is not None:
                row += f"  {md['mae']:>6.3f}  {md['rmse']:>6.3f}  {md['r2']:>5.3f}"
            else:
                row += "     n/a     n/a    n/a"
        print(row)

    print("-" * 80)
    print(f"Train/test split: {int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)} time-based")
    print(f"GBM:  {GBM_PARAMS}")
    print(f"RF:   {RF_PARAMS}")
    print(f"ARIMA order: {ARIMA_ORDER}  (on ARIMA station only)")
    print(f"Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS)}")
    print("=" * 80)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        log.error("Input not found: %s\nRun scripts/clean_data.py first.", args.input)
        sys.exit(1)

    log.info("Loading cleaned data from %s …", args.input)
    df = pd.read_parquet(args.input)
    log.info("Loaded: %d rows × %d cols", *df.shape)

    import json

    if args.pollutant == "all":
        metrics = run_training(
            df=df,
            output_dir=args.output,
            arima_station_id=args.arima_station,
            seed=args.seed,
        )
    else:
        # Single-pollutant mode: run one pollutant, save metrics alongside
        args.output.mkdir(parents=True, exist_ok=True)
        poll_metrics = train_pollutant(
            df=df,
            pollutant=args.pollutant,
            output_dir=args.output,
            arima_station_id=args.arima_station,
            seed=args.seed,
        )
        metrics = {args.pollutant: poll_metrics}
        metrics_path = args.output / "metrics.json"
        existing = {}
        if metrics_path.exists():
            with open(metrics_path) as fh:
                existing = json.load(fh)
        existing.update(metrics)
        with open(metrics_path, "w") as fh:
            json.dump(existing, fh, indent=2, default=str)

    print_comparison_table(metrics)
    log.info("All artifacts saved to: %s/", args.output)


if __name__ == "__main__":
    main()
