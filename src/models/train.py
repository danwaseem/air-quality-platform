"""
Model training for next-24-hour air quality prediction.

Three models are trained and compared per pollutant:

  GradientBoosting  sklearn HistGradientBoostingRegressor (direct 24h regression)
  RandomForest      sklearn RandomForestRegressor      (direct 24h regression)
  ARIMA             statsmodels ARIMA(2,1,2)           (multi-step forecast on
                    a single representative station — see NOTE below)

NOTE on ARIMA comparison
------------------------
The ML models perform *direct* 24h-ahead regression: given features at time t,
they predict y_{t+24}.  ARIMA is evaluated differently: it is fit on the
training time series and uses `forecast(n_test)` to predict the full test
period in one shot.  Both are compared against the same held-out test values,
but the forecast horizon structure differs.  ARIMA is included as a traditional
time-series baseline; treat its metrics accordingly.

Train / test split
------------------
Time-based, no shuffling.  `test_fraction=0.2` → last ~73 days of the year
form the test set; the rest train.  The cutoff timestamp is the same for all
models, ensuring identical evaluation windows.

Reproducibility
---------------
Pass `seed` everywhere; GBM and RF use `random_state=seed`.  Re-running with
the same seed on the same data produces identical results.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

from src.features.build_features import FEATURE_COLS, POLLUTANTS, build_all_features

log = logging.getLogger(__name__)

# ── Model hyper-parameters ────────────────────────────────────────────────────

GBM_PARAMS: dict[str, Any] = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "loss": "absolute_error",  # robust to outlier spikes; HistGBR has no huber
    # NaN handled natively — no imputation needed
}

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "n_jobs": -1,
}

ARIMA_ORDER: tuple[int, int, int] = (2, 1, 2)
ARIMA_MIN_TRAIN_ROWS: int = 500   # skip ARIMA if station has too little history

TEST_FRACTION: float = 0.2       # last 20% of timeline → test set


# ── Evaluation ────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "n_test": int(len(y_true)),
    }


# ── Train / test split ────────────────────────────────────────────────────────

def time_split(
    df_feat: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split feature DataFrame on a timestamp cutoff (no shuffling).

    Returns (df_train, df_test, cutoff_ts).
    """
    ts_sorted = df_feat["timestamp_utc"].sort_values()
    cutoff_idx = int(len(ts_sorted) * (1 - test_fraction))
    cutoff_ts = ts_sorted.iloc[cutoff_idx]

    df_train = df_feat[df_feat["timestamp_utc"] < cutoff_ts].copy()
    df_test = df_feat[df_feat["timestamp_utc"] >= cutoff_ts].copy()

    log.info(
        "  Split cutoff: %s  |  train %d rows  |  test %d rows",
        cutoff_ts.date(), len(df_train), len(df_test),
    )
    return df_train, df_test, cutoff_ts


# ── ML model trainers ─────────────────────────────────────────────────────────

def train_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(**GBM_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(**RF_PARAMS, random_state=seed)
    model.fit(X_train, y_train)
    return model


# ── ARIMA baseline ────────────────────────────────────────────────────────────

def train_arima_baseline(
    df_all: pd.DataFrame,
    pollutant: str,
    station_id: str,
    cutoff_ts: pd.Timestamp,
) -> tuple[Any | None, dict[str, float]]:
    """
    Fit ARIMA on one station's training series and forecast the test period.

    Returns (fitted_results, metrics_dict).  Returns (None, {}) if the station
    has insufficient data.
    """
    log.info("  ARIMA: fitting on station %s for %s", station_id, pollutant)

    st = df_all[df_all["station_id"] == station_id].sort_values("timestamp_utc")
    series = st.set_index("timestamp_utc")[pollutant]

    train_series = series[series.index < cutoff_ts]
    test_series = series[series.index >= cutoff_ts]

    if len(train_series) < ARIMA_MIN_TRAIN_ROWS or len(test_series) < 24:
        log.warning("  ARIMA: insufficient data for station %s — skipping", station_id)
        return None, {}

    # Fill remaining NaN with forward/backward fill so ARIMA gets a clean series
    train_filled = train_series.ffill().bfill()

    try:
        model = ARIMA(train_filled, order=ARIMA_ORDER)
        fitted = model.fit()
        preds = fitted.forecast(steps=len(test_series))
        preds = np.array(preds)
        actual = test_series.values
        m = _metrics(actual, preds)
        log.info(
            "  ARIMA %-6s  MAE=%.4f  RMSE=%.4f  R²=%.4f  (n=%d)",
            pollutant, m["mae"], m["rmse"], m["r2"], m["n_test"],
        )
        return fitted, m
    except Exception as exc:
        log.warning("  ARIMA fitting failed for %s/%s: %s", station_id, pollutant, exc)
        return None, {}


# ── Per-pollutant training orchestrator ───────────────────────────────────────

def train_pollutant(
    df: pd.DataFrame,
    pollutant: str,
    output_dir: Path,
    arima_station_id: str,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Build features, train all three models, evaluate, and save artifacts
    for a single pollutant.

    Returns {model_name: metrics_dict}.
    """
    log.info("=" * 60)
    log.info("Training pollutant: %s", pollutant.upper())
    log.info("=" * 60)

    # ── Feature engineering ───────────────────────────────────────────────────
    df_feat = build_all_features(df, pollutant)
    if df_feat.empty:
        log.error("No features for %s — skipping", pollutant)
        return {}

    df_train, df_test, cutoff_ts = time_split(df_feat)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train["target"]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test["target"]

    results: dict[str, dict[str, float]] = {}

    # ── GradientBoosting ──────────────────────────────────────────────────────
    log.info("  Training GradientBoosting …")
    gbm = train_gbm(X_train, y_train, seed=seed)
    gbm_pred = gbm.predict(X_test)
    m = _metrics(y_test.values, gbm_pred)
    results["GradientBoosting"] = m
    log.info("  GBM %-6s  MAE=%.4f  RMSE=%.4f  R²=%.4f", pollutant, m["mae"], m["rmse"], m["r2"])
    joblib.dump(gbm, output_dir / f"gbm_{pollutant}.joblib")

    # ── RandomForest ─────────────────────────────────────────────────────────
    log.info("  Training RandomForest …")
    rf = train_rf(X_train, y_train, seed=seed)
    rf_pred = rf.predict(X_test)
    m = _metrics(y_test.values, rf_pred)
    results["RandomForest"] = m
    log.info("  RF  %-6s  MAE=%.4f  RMSE=%.4f  R²=%.4f", pollutant, m["mae"], m["rmse"], m["r2"])
    joblib.dump(rf, output_dir / f"rf_{pollutant}.joblib")

    # ── ARIMA baseline ────────────────────────────────────────────────────────
    arima_fitted, m = train_arima_baseline(df, pollutant, arima_station_id, cutoff_ts)
    if arima_fitted is not None:
        results["ARIMA"] = m
        arima_fitted.save(str(output_dir / f"arima_{pollutant}.pkl"))
    else:
        results["ARIMA"] = {"mae": None, "rmse": None, "r2": None, "n_test": 0}

    return results


# ── Model card writer ─────────────────────────────────────────────────────────

def write_model_card(
    metrics: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    arima_station_id: str,
    cutoff_date: str,
) -> None:
    """Write models/model_card.md with approach, features, split, and results."""

    def _row(pollutant: str, model: str) -> str:
        m = metrics.get(pollutant, {}).get(model, {})
        mae = f"{m['mae']:.4f}" if m.get("mae") is not None else "n/a"
        rmse = f"{m['rmse']:.4f}" if m.get("rmse") is not None else "n/a"
        r2 = f"{m['r2']:.4f}" if m.get("r2") is not None else "n/a"
        n = m.get("n_test", "-")
        return f"| {pollutant:<6} | {model:<20} | {mae:>8} | {rmse:>8} | {r2:>8} | {n:>7} |"

    models = ["GradientBoosting", "RandomForest", "ARIMA"]
    table_rows = "\n".join(
        _row(p, m)
        for p in POLLUTANTS
        for m in models
    )

    card = textwrap.dedent(f"""\
    # Air Quality Prediction — Model Card

    ## Task
    Next-24-hour prediction of PM2.5 (µg/m³), NO2 (ppb), and Ozone (ppm)
    at EPA AQS monitoring stations across 10 US states (2023).

    ## Data
    - Source: EPA Air Quality System (AQS) hourly sample data
    - Coverage: 215 stations, Jan–Dec 2023, hourly cadence
    - Preprocessing: schema normalization, short-gap interpolation (≤3 h),
      outlier flagging (negatives, beyond physical bounds, rolling z-score)

    ## Features ({len(FEATURE_COLS)} total)
    | Group | Features |
    |-------|----------|
    | Lag values | lag_1h, lag_24h, lag_168h |
    | Rolling stats | roll_24h_mean/std, roll_168h_mean/std |
    | Cyclical time | hour sin/cos, day-of-week sin/cos, month sin/cos |
    | Calendar | season (0–3) |
    | Geography | latitude, longitude |
    | Quality | drift_flag (1 if rolling z-score > 3.5) |

    ## Models
    | Model | Description |
    |-------|-------------|
    | GradientBoosting | sklearn HistGBR, loss=absolute_error, max_iter=300, lr=0.05, depth=5 |
    | RandomForest | sklearn RF, n_estimators=200, max_features=sqrt |
    | ARIMA | statsmodels ARIMA{ARIMA_ORDER}, fit on station {arima_station_id} only |

    ## Train / Test Split
    - Strategy: time-based (no shuffling) to prevent temporal leakage
    - Cutoff: {cutoff_date}
    - Train: all rows with timestamp < cutoff (~80% of data)
    - Test: all rows with timestamp ≥ cutoff (~20% of data)

    ## ARIMA Caveat
    GradientBoosting and RandomForest perform *direct* 24h-ahead regression
    across all stations simultaneously.  ARIMA is univariate, fit on a single
    representative station ({arima_station_id}), and uses `forecast(n_test)`
    to predict the full test period in one shot — not rolling 24h-ahead
    forecasts.  ARIMA metrics are indicative, not strictly comparable.

    ## Results

    | Pollutant | Model | MAE | RMSE | R² | n_test |
    |-----------|-------|----:|-----:|---:|-------:|
    {table_rows}

    ## Artifacts
    | File | Contents |
    |------|----------|
    | `gbm_<poll>.joblib` | Fitted GradientBoostingRegressor |
    | `rf_<poll>.joblib` | Fitted RandomForestRegressor |
    | `arima_<poll>.pkl` | Fitted statsmodels ARIMA results |
    | `metrics.json` | All MAE/RMSE/R² values (machine-readable) |
    | `model_card.md` | This document |
    """)

    (output_dir / "model_card.md").write_text(card)
    log.info("Model card → %s/model_card.md", output_dir)


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_training(
    df: pd.DataFrame,
    output_dir: Path,
    arima_station_id: str = "482011039",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train all three models for every pollutant.

    Parameters
    ----------
    df               : full cleaned DataFrame
    output_dir       : directory for model artifacts
    arima_station_id : AQS station used for the ARIMA baseline
    seed             : random seed for reproducibility

    Returns
    -------
    Full metrics dict: {pollutant: {model: {mae, rmse, r2, n_test}}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    all_metrics: dict[str, Any] = {}
    cutoff_date: str = "unknown"

    for pollutant in POLLUTANTS:
        poll_metrics = train_pollutant(
            df=df,
            pollutant=pollutant,
            output_dir=output_dir,
            arima_station_id=arima_station_id,
            seed=seed,
        )
        all_metrics[pollutant] = poll_metrics

        # Record the cutoff from the feature split (consistent across pollutants)
        if cutoff_date == "unknown" and poll_metrics:
            # Approximate from data
            df_feat = build_all_features(df, pollutant)
            if not df_feat.empty:
                ts_sorted = df_feat["timestamp_utc"].sort_values()
                idx = int(len(ts_sorted) * (1 - TEST_FRACTION))
                cutoff_date = str(ts_sorted.iloc[idx].date())

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)
    log.info("Metrics → %s", metrics_path)

    write_model_card(all_metrics, output_dir, arima_station_id, cutoff_date)
    return all_metrics
