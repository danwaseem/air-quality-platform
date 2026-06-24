"""Unit tests for the cleaning pipeline (no file I/O, no live data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning.clean import (
    PHYS_BOUNDS,
    POLLUTANTS,
    SHORT_GAP_HOURS,
    fill_gaps,
    flag_outliers,
    normalize_schema,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_raw(n: int = 24) -> pd.DataFrame:
    """Minimal raw AQS long-format DataFrame for one station, one pollutant."""
    base = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    rows = []
    for i in range(n):
        rows.append({
            "state_code": "06",
            "county_code": "037",
            "site_number": "0001",
            "parameter_code": "88101",
            "poc": 1,
            "latitude": 34.05,
            "longitude": -118.25,
            "datum": "NAD83",
            "parameter": "PM2.5 - Local Conditions",
            "date_gmt": (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%d"),
            "time_gmt": (base + pd.Timedelta(hours=i)).strftime("%H:%M"),
            "date_local": (base + pd.Timedelta(hours=i)).strftime("%Y-%m-%d"),
            "time_local": (base + pd.Timedelta(hours=i)).strftime("%H:%M"),
            "sample_measurement": float(i + 1),
            "units_of_measure": "Micrograms/cubic meter (LC)",
            "units_of_measure_code": "105",
            "sample_duration": "1 HOUR",
            "sample_duration_code": "1",
            "sample_frequency": "HOURLY",
            "detection_limit": 0.5,
            "uncertainty": np.nan,
            "qualifier": np.nan,
            "method_type": "FEM",
            "method": "TEST",
            "method_code": "170",
            "state": "California",
            "county": "Los Angeles",
            "date_of_last_change": "2023-06-01",
            "cbsa_code": "31080",
            "datetime_local": pd.NaT,
        })
    return pd.DataFrame(rows)


# ── normalize_schema ──────────────────────────────────────────────────────────

def test_normalize_station_id():
    df, _ = normalize_schema(_make_raw())
    # "06" + "037" + "0001" = 9 chars
    assert (df["station_id"] == "060370001").all()


def test_normalize_timestamp_utc():
    df, _ = normalize_schema(_make_raw())
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp_utc"])
    assert str(df["timestamp_utc"].dt.tz) == "UTC"


def test_normalize_pivot_columns():
    df, _ = normalize_schema(_make_raw())
    for p in POLLUTANTS:
        assert p in df.columns


def test_normalize_complete_hourly_grid():
    """All 24 hours should be present after normalization."""
    df, _ = normalize_schema(_make_raw(24))
    station_df = df[df["station_id"] == "060370001"]
    assert len(station_df) == 24


def test_normalize_metrics():
    _, metrics = normalize_schema(_make_raw())
    assert metrics["raw_rows"] == 24
    assert metrics["station_count"] == 1
    assert metrics["stations_by_pollutant"]["pm25"] == 1


def test_normalize_filters_unknown_params():
    df_raw = _make_raw(5)
    df_raw.loc[0, "parameter_code"] = "99999"  # unknown param
    df, metrics = normalize_schema(df_raw)
    assert metrics["rows_after_param_filter"] < metrics["raw_rows"]


# ── fill_gaps ────────────────────────────────────────────────────────────────

def _wide_with_gaps(n_gaps: int = 2) -> pd.DataFrame:
    """Wide DataFrame with a known gap in pm25."""
    df_raw = _make_raw(24)
    df, _ = normalize_schema(df_raw)
    # Introduce n_gaps consecutive NaN values mid-series
    mask = df.index[10: 10 + n_gaps]
    df.loc[mask, "pm25"] = np.nan
    return df


def test_fill_gaps_short_gap_interpolated():
    df = _wide_with_gaps(n_gaps=SHORT_GAP_HOURS)  # exactly at limit — should fill
    df_filled, metrics = fill_gaps(df)
    assert df_filled["pm25"].isna().sum() < df["pm25"].isna().sum()
    assert metrics["interpolated"]["pm25"] == SHORT_GAP_HOURS


def test_fill_gaps_long_gap_not_filled():
    df = _wide_with_gaps(n_gaps=SHORT_GAP_HOURS + 2)  # over limit — should stay NaN
    df_filled, metrics = fill_gaps(df)
    assert metrics["long_gaps_remaining"]["pm25"] > 0


def test_fill_gaps_interp_flag():
    df = _wide_with_gaps(n_gaps=1)
    df_filled, _ = fill_gaps(df)
    assert "pm25_interp_flag" in df_filled.columns
    assert df_filled["pm25_interp_flag"].any()


# ── flag_outliers ─────────────────────────────────────────────────────────────

def _wide_with_outliers() -> pd.DataFrame:
    df_raw = _make_raw(48)
    df, _ = normalize_schema(df_raw)
    df.loc[df.index[5], "pm25"] = -5.0       # negative
    df.loc[df.index[6], "pm25"] = 9999.0     # beyond bounds
    return df


def test_flag_outliers_negative_flagged():
    df = _wide_with_outliers()
    df_out, metrics = flag_outliers(df)
    assert metrics["negative"]["pm25"] >= 1
    assert "pm25_neg_flag" in df_out.columns


def test_flag_outliers_beyond_bounds_nulled():
    df = _wide_with_outliers()
    df_out, metrics = flag_outliers(df)
    assert metrics["beyond_bounds"]["pm25"] >= 2
    # The extreme value should now be NaN
    assert df_out.loc[df_out.index[6], "pm25"] != 9999.0


def test_flag_outliers_drift_column_exists():
    df, _ = normalize_schema(_make_raw(200))
    df_out, _ = flag_outliers(df)
    for p in POLLUTANTS:
        assert f"{p}_drift_flag" in df_out.columns


def test_phys_bounds_cover_all_pollutants():
    assert set(PHYS_BOUNDS.keys()) == set(POLLUTANTS)
