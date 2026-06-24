"""Unit tests for feature engineering — no file I/O, no model training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLS,
    POLLUTANTS,
    _cyclical,
    build_all_features,
    engineer_station_features,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_station_df(
    n_hours: int = 400,
    station_id: str = "060371103",
    pollutant_val: float = 10.0,
    introduce_nans: bool = False,
) -> pd.DataFrame:
    """Minimal cleaned DataFrame for one station, all three pollutants present."""
    base = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    ts = pd.date_range(base, periods=n_hours, freq="h")
    df = pd.DataFrame({
        "station_id": station_id,
        "timestamp_utc": ts,
        "pm25": pollutant_val + np.random.default_rng(0).normal(0, 1, n_hours),
        "no2": 15.0 + np.random.default_rng(1).normal(0, 2, n_hours),
        "ozone": 0.04 + np.random.default_rng(2).normal(0, 0.005, n_hours),
        "latitude": 34.05,
        "longitude": -118.25,
        "state": "California",
        "county": "Los Angeles",
        "pm25_interp_flag": False,
        "no2_interp_flag": False,
        "ozone_interp_flag": False,
        "pm25_drift_flag": False,
        "no2_drift_flag": False,
        "ozone_drift_flag": False,
        "pm25_neg_flag": False,
        "no2_neg_flag": False,
        "ozone_neg_flag": False,
        "pm25_bounds_flag": False,
        "no2_bounds_flag": False,
        "ozone_bounds_flag": False,
    })
    if introduce_nans:
        # Simulate a 5-hour gap in pm25
        df.loc[50:54, "pm25"] = np.nan
    return df


# ── _cyclical ─────────────────────────────────────────────────────────────────

def test_cyclical_bounds():
    s = pd.Series(range(24))
    sin_v, cos_v = _cyclical(s, 24)
    assert sin_v.abs().le(1.0).all()
    assert cos_v.abs().le(1.0).all()


def test_cyclical_periodicity():
    s = pd.Series([0, 24, 48])   # all equivalent at period=24
    sin_v, _ = _cyclical(s, 24)
    assert np.allclose(sin_v.values, 0.0, atol=1e-10)


# ── engineer_station_features ─────────────────────────────────────────────────

def test_all_feature_cols_present():
    df = _make_station_df()
    out = engineer_station_features(df, "pm25")
    for col in FEATURE_COLS:
        assert col in out.columns, f"Missing feature column: {col}"


def test_target_is_24h_ahead():
    """Target value at row i should equal pm25 at row i+24."""
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    # After lag alignment the rows in `out` still carry their original timestamp;
    # pick a row, find its timestamp_utc, look up the value 24h later in df.
    row = out.iloc[50]
    ts = row["timestamp_utc"]
    ts_ahead = ts + pd.Timedelta(hours=24)
    expected_val = df.set_index("timestamp_utc").loc[ts_ahead, "pm25"]
    assert pytest.approx(row["target"], rel=1e-6) == expected_val


def test_no_future_leakage_in_rolling():
    """roll_24h_mean at row t must equal mean of rows [t-24..t-1], not [t-23..t]."""
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    # At row 100 (0-indexed in out), roll_24h_mean should use pm25[76..99] from df
    row = out.iloc[100]
    ts = row["timestamp_utc"]
    # Find the 24 hours before ts in df (excluding ts itself)
    window = df[
        (df["timestamp_utc"] < ts) &
        (df["timestamp_utc"] >= ts - pd.Timedelta(hours=24))
    ]["pm25"]
    expected = window.mean()
    assert pytest.approx(row["roll_24h_mean"], rel=1e-4) == expected


def test_essential_lags_not_null():
    """After dropna, lag_1h and lag_24h must be non-null."""
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    assert out["lag_1h"].isna().sum() == 0
    assert out["lag_24h"].isna().sum() == 0


def test_target_not_null():
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    assert out["target"].isna().sum() == 0


def test_first_rows_dropped():
    """First 24 rows should be dropped (lag_24h is NaN there)."""
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    # Earliest timestamp in output should be at least 24h after start of df
    assert out["timestamp_utc"].min() >= df["timestamp_utc"].iloc[24]


def test_last_rows_dropped():
    """Last 24 rows should be dropped (target is NaN there)."""
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    assert out["timestamp_utc"].max() <= df["timestamp_utc"].iloc[-25]


def test_drift_flag_is_int():
    df = _make_station_df(n_hours=300)
    out = engineer_station_features(df, "pm25")
    assert out["drift_flag"].dtype in (int, np.int64, np.int32)


def test_season_range():
    df = _make_station_df(n_hours=8760)
    out = engineer_station_features(df, "pm25")
    assert out["season"].between(0, 3).all()


# ── build_all_features ────────────────────────────────────────────────────────

def test_build_all_features_single_station():
    df = _make_station_df(n_hours=500)
    out = build_all_features(df, "pm25", min_readings=100)
    assert not out.empty
    assert "target" in out.columns
    for col in FEATURE_COLS:
        assert col in out.columns


def test_build_all_features_skips_sparse_station():
    """Station with too few valid pm25 readings should be skipped."""
    df1 = _make_station_df(n_hours=500, station_id="AAA")
    df2 = _make_station_df(n_hours=500, station_id="BBB")
    df2["pm25"] = np.nan  # BBB has no valid pm25
    df = pd.concat([df1, df2], ignore_index=True)
    out = build_all_features(df, "pm25", min_readings=100)
    assert (out["station_id"] == "BBB").sum() == 0


def test_build_all_features_multi_station_sorted():
    """Output should be sorted by timestamp_utc."""
    df1 = _make_station_df(n_hours=400, station_id="S1")
    df2 = _make_station_df(n_hours=400, station_id="S2")
    df = pd.concat([df1, df2], ignore_index=True)
    out = build_all_features(df, "pm25", min_readings=100)
    assert out["timestamp_utc"].is_monotonic_increasing
