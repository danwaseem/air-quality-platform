#!/usr/bin/env python3
"""
Pull hourly AQS data for PM2.5, NO2, and ozone across multiple states
for a full calendar year and save raw results as Parquet in data/raw/.

Target: 500k+ hourly rows.

Strategy
--------
We pull the top-10 most populous US states by county (covering ~30 counties
across CA, TX, NY, FL, IL, PA, OH, GA, NC, AZ).  With three pollutants ×
12 months × ~30 counties × ~700 monitors each you easily exceed 500k rows.

Usage
-----
    python scripts/pull_data.py [--year YYYY] [--out data/raw]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.aqs_client import AQSClient, PARAMETERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Target geography ──────────────────────────────────────────────────────────
# (state_fips, county_fips) pairs chosen for high monitor density.
# Covers the most populated urban counties in 10 large states.
# Each (param, county, year) is one API call; ~90 calls total for 3 params.

TARGET_COUNTIES: list[tuple[str, str]] = [
    # California
    ("06", "037"),  # Los Angeles
    ("06", "073"),  # San Diego
    ("06", "001"),  # Alameda (Oakland/Berkeley)
    ("06", "085"),  # Santa Clara (San Jose)
    ("06", "059"),  # Orange
    # Texas
    ("48", "113"),  # Dallas
    ("48", "201"),  # Harris (Houston)
    ("48", "029"),  # Bexar (San Antonio)
    ("48", "453"),  # Travis (Austin)
    ("48", "141"),  # El Paso
    # New York
    ("36", "061"),  # New York (Manhattan)
    ("36", "047"),  # Kings (Brooklyn)
    ("36", "081"),  # Queens
    ("36", "005"),  # Bronx
    ("36", "059"),  # Nassau
    # Florida
    ("12", "086"),  # Miami-Dade
    ("12", "011"),  # Broward
    ("12", "057"),  # Hillsborough (Tampa)
    ("12", "095"),  # Orange (Orlando)
    # Illinois
    ("17", "031"),  # Cook (Chicago)
    ("17", "043"),  # DuPage
    # Pennsylvania
    ("42", "101"),  # Philadelphia
    ("42", "003"),  # Allegheny (Pittsburgh)
    # Ohio
    ("39", "035"),  # Cuyahoga (Cleveland)
    ("39", "049"),  # Franklin (Columbus)
    ("39", "061"),  # Hamilton (Cincinnati)
    # Georgia
    ("13", "121"),  # Fulton (Atlanta)
    ("13", "089"),  # DeKalb
    # North Carolina
    ("37", "119"),  # Mecklenburg (Charlotte)
    ("37", "183"),  # Wake (Raleigh)
    # Arizona
    ("04", "013"),  # Maricopa (Phoenix)
    ("04", "019"),  # Pima (Tucson)
]

PARAM_CODES = list(PARAMETERS.keys())  # ["88101", "42602", "44201"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--year", type=int, default=2023,
        help="Calendar year to pull (default: 2023 — most complete recent year)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw"),
        help="Output directory for Parquet files (default: data/raw)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between API calls (default: 1.5 — polite rate)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    start = date(args.year, 1, 1)
    end = date(args.year, 12, 31)

    log.info("Initialising AQS client …")
    client = AQSClient()  # reads EPA_AQS_EMAIL / EPA_AQS_KEY from .env

    log.info(
        "Pulling %d parameters × %d counties for %d-%d",
        len(PARAM_CODES), len(TARGET_COUNTIES), args.year, args.year,
    )
    log.info("Parameters: %s", {c: PARAMETERS[c] for c in PARAM_CODES})
    log.info("Total API calls planned: %d", len(PARAM_CODES) * len(TARGET_COUNTIES))

    df = client.pull_hourly(
        param_codes=PARAM_CODES,
        state_counties=TARGET_COUNTIES,
        start=start,
        end=end,
        inter_request_delay=args.delay,
    )

    if df.empty:
        log.error("No data returned — check credentials and county list.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.out / f"aqs_hourly_{args.year}.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    log.info("Saved → %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PULL COMPLETE")
    print("=" * 60)
    print(f"  Total rows        : {len(df):,}")
    print(f"  Output file       : {out_path}")
    print(f"  File size         : {out_path.stat().st_size / 1e6:.1f} MB")

    if "datetime_local" in df.columns:
        print(f"  Date coverage     : {df['datetime_local'].min()}  →  {df['datetime_local'].max()}")

    if "parameter" in df.columns:
        print("\n  Rows by parameter:")
        for param, count in df["parameter"].value_counts().items():
            print(f"    {param:<30} {count:>10,}")

    if "state_name" in df.columns:
        print("\n  Rows by state:")
        for state, count in df["state_name"].value_counts().items():
            print(f"    {state:<30} {count:>10,}")

    print("=" * 60)

    if len(df) < 500_000:
        log.warning(
            "Row count %d is below 500k target. "
            "Try adding more counties or an additional year.",
            len(df),
        )


if __name__ == "__main__":
    main()
