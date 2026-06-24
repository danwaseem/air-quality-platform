#!/usr/bin/env python3
"""
Bundle dashboard CSVs into a single Excel workbook for Power BI.

Reads from dashboards/data/:
  model_metrics.csv   → sheet "ModelMetrics"
  bias_by_state.csv   → sheet "BiasByState"
  bias_by_region.csv  → sheet "BiasByRegion"

Writes:
  dashboards/data/powerbi_data.xlsx

Requires openpyxl:
  pip install openpyxl

Run from project root:
  python scripts/export_powerbi_workbook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
IN_DIR  = ROOT / "dashboards" / "data"
OUT_XL  = IN_DIR / "powerbi_data.xlsx"

# (csv_filename, sheet_name) — order sets tab order in the workbook
SHEETS: list[tuple[str, str]] = [
    ("model_metrics.csv", "ModelMetrics"),
    ("bias_by_state.csv", "BiasByState"),
    ("bias_by_region.csv", "BiasByRegion"),
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    missing = [IN_DIR / f for f, _ in SHEETS if not (IN_DIR / f).exists()]
    if missing:
        for p in missing:
            print(f"ERROR: missing input: {p}")
        print("Run scripts/export_for_dashboards.py first.")
        sys.exit(1)

    try:
        import openpyxl  # noqa: F401 — checked here for a clear error message
    except ImportError:
        sys.exit("ERROR: openpyxl not installed. Run: pip install openpyxl")

    with pd.ExcelWriter(OUT_XL, engine="openpyxl") as writer:
        for csv_name, sheet_name in SHEETS:
            df = pd.read_csv(IN_DIR / csv_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Verify and report
    wb_size_kb = OUT_XL.stat().st_size / 1024
    print(f"Workbook written: {OUT_XL}  ({wb_size_kb:.1f} KB)")
    print(f"Sheets ({len(SHEETS)}):")
    for csv_name, sheet_name in SHEETS:
        df = pd.read_csv(IN_DIR / csv_name)
        cols = ", ".join(df.columns)
        print(f"  {sheet_name:<16}  {len(df):>4} rows  columns: {cols}")


if __name__ == "__main__":
    main()
