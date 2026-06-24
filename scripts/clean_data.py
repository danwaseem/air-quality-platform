#!/usr/bin/env python3
"""
Run the air quality cleaning pipeline and print a full metrics report.

Reads from data/raw/aqs_hourly_<year>.parquet by default (the merged output
of pull_data.py).  If that file doesn't exist, will merge available chunks
from data/raw/.chunks_<year>/ on the fly so you can clean a partial pull.

Usage
-----
    python scripts/clean_data.py
    python scripts/clean_data.py --year 2023
    python scripts/clean_data.py --input path/to/custom.parquet
    python scripts/clean_data.py --input data/raw/aqs_hourly_2023.parquet \\
                                  --out-data data/processed/clean_hourly.parquet \\
                                  --out-report data/processed/cleaning_report.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cleaning.clean import POLLUTANTS, PHYS_BOUNDS, SHORT_GAP_HOURS, run_pipeline

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
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--input", type=Path, default=None,
                   help="Override input path (skips year-based lookup)")
    p.add_argument("--out-data", type=Path,
                   default=Path("data/processed/clean_hourly.parquet"))
    p.add_argument("--out-report", type=Path,
                   default=Path("data/processed/cleaning_report.json"))
    return p.parse_args()


def resolve_input(args: argparse.Namespace) -> Path:
    """Return the input parquet path, merging chunks on-the-fly if needed."""
    if args.input:
        if not args.input.exists():
            log.error("Input file not found: %s", args.input)
            sys.exit(1)
        return args.input

    merged = Path(f"data/raw/aqs_hourly_{args.year}.parquet")
    if merged.exists():
        log.info("Using merged parquet: %s", merged)
        return merged

    # Fall back to on-the-fly chunk merge
    chunks_dir = Path(f"data/raw/.chunks_{args.year}")
    chunks = sorted(chunks_dir.glob("*.parquet")) if chunks_dir.exists() else []
    if not chunks:
        log.error(
            "No input found. Expected %s or chunks in %s.\n"
            "Run scripts/pull_data.py first.",
            merged, chunks_dir,
        )
        sys.exit(1)

    log.info("Merged parquet not found — merging %d chunk files on the fly …", len(chunks))
    tmp = Path(f"data/raw/aqs_hourly_{args.year}_partial.parquet")
    dfs = [pd.read_parquet(c) for c in chunks if __import__("pyarrow.parquet", fromlist=["read_metadata"]).read_metadata(c).num_rows > 0]
    pd.concat(dfs, ignore_index=True).to_parquet(tmp, index=False, compression="snappy")
    log.info("Wrote temporary merged file: %s  (%d chunks)", tmp, len(dfs))
    return tmp


def print_report(report: dict) -> None:
    schema = report["schema"]
    miss = report["missingness"]
    outliers = report["outliers"]
    out = report["output"]

    w = 62
    print("\n" + "=" * w)
    print("CLEANING REPORT")
    print("=" * w)

    print(f"\n{'── Input':─<{w}}")
    print(f"  Raw rows          : {schema['raw_rows']:>12,}")
    print(f"  After TS parse    : {schema['rows_after_ts_parse']:>12,}")
    print(f"  After param filter: {schema['rows_after_param_filter']:>12,}")
    print(f"\n{'── Schema':─<{w}}")
    print(f"  Pivoted shape     : {schema['pivoted_rows']:,} rows × {schema['pivoted_cols']} cols")
    print(f"  Distinct stations : {schema['station_count']:,}")
    print("  Stations by pollutant:")
    for p, cnt in schema["stations_by_pollutant"].items():
        print(f"    {p:<8}  {cnt:>4} stations")

    print(f"\n{'── Missingness — before gap fill':─<{w}}")
    before = miss["before"]
    for p in POLLUTANTS:
        s = before[p]
        print(f"  {p:<8}  {s['missing_count']:>8,} / {s['total_expected']:>8,}  "
              f"({s['missing_pct']:>5.1f}%)")

    print(f"\n{'── Gap fill (short ≤ ' + str(SHORT_GAP_HOURS) + 'h)':─<{w}}")
    for p in POLLUTANTS:
        interp = miss["interpolated"][p]
        remaining = miss["long_gaps_remaining"][p]
        print(f"  {p:<8}  {interp:>8,} interpolated | {remaining:>8,} in long gaps")

    print(f"\n{'── Missingness — after gap fill':─<{w}}")
    after = miss["after"]
    for p in POLLUTANTS:
        s = after[p]
        print(f"  {p:<8}  {s['missing_count']:>8,} / {s['total_expected']:>8,}  "
              f"({s['missing_pct']:>5.1f}%)")

    print(f"\n{'── Outlier detection':─<{w}}")
    bounds_str = {p: f"[{PHYS_BOUNDS[p][0]:.4g} – {PHYS_BOUNDS[p][1]:.4g}]" for p in POLLUTANTS}
    for p in POLLUTANTS:
        neg = outliers["negative"][p]
        bnd = outliers["beyond_bounds"][p]
        dft = outliers["drift_flagged"][p]
        print(f"  {p:<8}  {neg:>7,} negative | {bnd:>7,} beyond {bounds_str[p]} "
              f"→ NaN  |  {dft:>7,} drift flagged")

    print(f"\n{'── Output':─<{w}}")
    print(f"  Rows              : {out['rows']:>12,}")
    print(f"  Columns           : {len(out['columns']):>12,}")
    print(f"  Path              : {out['path']}")
    print("=" * w)


def main() -> None:
    args = parse_args()
    input_path = resolve_input(args)
    report = run_pipeline(
        raw_path=input_path,
        out_data=args.out_data,
        out_report=args.out_report,
    )
    print_report(report)


if __name__ == "__main__":
    main()
