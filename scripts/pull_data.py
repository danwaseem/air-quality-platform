#!/usr/bin/env python3
"""
Pull hourly AQS data for PM2.5, NO2, and ozone across multiple states
for a full calendar year and save raw results as Parquet in data/raw/.

Target: 500k+ hourly rows.

Resume support: each API call's result is saved as a small chunk file in
data/raw/.chunks/<year>/ immediately after it completes.  Re-running the
script skips any chunk that already exists, so a restart picks up where
it left off.  Run with --fresh to ignore existing chunks and start over.

Usage
-----
    python scripts/pull_data.py [--year YYYY] [--out data/raw] [--fresh]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.aqs_client import AQSClient, PARAMETERS, _year_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Target geography ──────────────────────────────────────────────────────────

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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _chunk_path(chunks_dir: Path, param: str, state: str, county: str,
                bdate: date, edate: date) -> Path:
    tag = f"{param}_{state}_{county}_{bdate.year}"
    return chunks_dir / f"{tag}.parquet"


def _row_count(path: Path) -> int:
    """Read row count from parquet metadata — no data deserialization."""
    return pq.read_metadata(path).num_rows


def _load_existing_chunks(chunks_dir: Path) -> tuple[set[Path], int]:
    """Return (set of existing chunk paths, total rows already saved)."""
    existing = set(chunks_dir.glob("*.parquet"))
    total = sum(_row_count(p) for p in existing)
    return existing, total


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between API calls (default: 1.5)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing chunks and re-pull everything")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    chunks_dir = args.out / f".chunks_{args.year}"
    chunks_dir.mkdir(exist_ok=True)

    if args.fresh and chunks_dir.exists():
        for f in chunks_dir.glob("*.parquet"):
            f.unlink()
        log.info("--fresh: cleared existing chunks")

    existing_chunks, existing_rows = _load_existing_chunks(chunks_dir)
    if existing_rows:
        log.info("Resuming — %d chunks already done, %d rows on disk",
                 len(existing_chunks), existing_rows)

    start = date(args.year, 1, 1)
    end = date(args.year, 12, 31)

    client = AQSClient()

    combos = [
        (param, state, county, bdate, edate)
        for param in PARAM_CODES
        for state, county in TARGET_COUNTIES
        for bdate, edate in _year_windows(start, end)
    ]
    total = len(combos)
    log.info("Total API calls planned: %d", total)

    skipped_errors: list[str] = []
    running_rows = existing_rows

    for idx, (param, state, county, bdate, edate) in enumerate(combos, 1):
        chunk = _chunk_path(chunks_dir, param, state, county, bdate, edate)

        if chunk in existing_chunks:
            log.info("[%d/%d] SKIP (cached)  %s state=%s county=%s",
                     idx, total, PARAMETERS[param], state, county)
            continue

        log.info("[%d/%d] %s  state=%s county=%s  %s→%s",
                 idx, total, PARAMETERS[param], state, county, bdate, edate)

        try:
            rows = client._hourly_by_county(param, state, county, bdate, edate)
        except RuntimeError as exc:
            label = f"{PARAMETERS[param]} state={state} county={county}"
            log.warning("Skipping %s — %s", label, exc)
            skipped_errors.append(label)
            # Save empty chunk so we don't retry on resume
            pd.DataFrame().to_parquet(chunk, index=False)
            rows = []

        if rows:
            chunk_df = client._normalize(pd.DataFrame(rows))
            chunk_df.to_parquet(chunk, index=False, compression="snappy")

        running_rows += len(rows)
        log.info("  → %d rows (running total: %d)", len(rows), running_rows)

        if idx < total:
            time.sleep(args.delay)

    # ── Merge all chunks into final file ──────────────────────────────────────
    chunk_files = [f for f in sorted(chunks_dir.glob("*.parquet")) if _row_count(f) > 0]
    parts = [pd.read_parquet(f) for f in chunk_files]

    if not parts:
        log.error("No data collected — check credentials and county list.")
        sys.exit(1)

    log.info("Merging %d chunk files …", len(parts))
    df = pd.concat(parts, ignore_index=True)

    out_path = args.out / f"aqs_hourly_{args.year}.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")

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

    if skipped_errors:
        print(f"\n  Skipped (API errors): {len(skipped_errors)}")
        for s in skipped_errors:
            print(f"    {s}")

    print("=" * 60)

    if len(df) < 500_000:
        log.warning("Row count %d is below 500k target.", len(df))


if __name__ == "__main__":
    main()
