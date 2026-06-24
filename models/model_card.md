    # Air Quality Prediction — Model Card

    ## Task
    Next-24-hour prediction of PM2.5 (µg/m³), NO2 (ppb), and Ozone (ppm)
    at EPA AQS monitoring stations across 10 US states (2023).

    ## Data
    - Source: EPA Air Quality System (AQS) hourly sample data
    - Coverage: 215 stations, Jan–Dec 2023, hourly cadence
    - Preprocessing: schema normalization, short-gap interpolation (≤3 h),
      outlier flagging (negatives, beyond physical bounds, rolling z-score)

    ## Features (20 total)
    | Group | Features |
    |-------|----------|
    | Lag values | lag_1h, lag_24h, lag_168h |
    | Rolling stats | roll_24h_mean/std, roll_168h_mean/std |
    | Cyclical time | hour sin/cos, day-of-week sin/cos, month sin/cos |
    | Calendar | season (0–3) |
    | Geography | latitude, longitude |
    | Quality | drift_flag (1 if rolling z-score > 3.5) |
    | Weather | temperature_2m (°C), wind_speed_10m (m/s), relative_humidity_2m (%) |

    Weather features are current-hour values aligned to each prediction row (no leakage).
    Source: Open-Meteo historical archive for training; OpenWeatherMap/Open-Meteo live API for /predict/live.
    Rows with null weather (~0.55%) are dropped at training time. HistGBR handles NaN natively in inference.

    ## Models
    | Model | Description |
    |-------|-------------|
    | GradientBoosting | sklearn HistGBR, loss=absolute_error, max_iter=300, lr=0.05, depth=5 |
    | RandomForest | sklearn RF, n_estimators=200, max_features=sqrt |
    | ARIMA | statsmodels ARIMA(2, 1, 2), fit on station 482011039 only |

    ## Train / Test Split
    - Strategy: time-based (no shuffling) to prevent temporal leakage
    - Cutoff: 2023-10-26
    - Train: all rows with timestamp < cutoff (~80% of data)
    - Test: all rows with timestamp ≥ cutoff (~20% of data)

    ## ARIMA Caveat
    GradientBoosting and RandomForest perform *direct* 24h-ahead regression
    across all stations simultaneously.  ARIMA is univariate, fit on a single
    representative station (482011039), and uses `forecast(n_test)`
    to predict the full test period in one shot — not rolling 24h-ahead
    forecasts.  ARIMA metrics are indicative, not strictly comparable.

    ## Results

    GBM metrics are from the weather-augmented model (20 features). RF/ARIMA are
    from the original 17-feature training run. See `models/weather_ablation.json`
    for the full baseline vs. augmented comparison.

    | Pollutant | Model | MAE | RMSE | R² | n_test |
    |-----------|-------|----:|-----:|---:|-------:|
    | pm25   | GradientBoosting (weather) |   4.3781 |   8.0367 |   0.1131 |  158193 |
    | pm25   | RandomForest               |   4.6463 |   7.8765 |   0.1448 |  159690 |
    | pm25   | ARIMA                      |   3.6333 |   5.0677 |  -0.0019 |    1606 |
    | no2    | GradientBoosting (weather) |   5.8871 |   8.0122 |   0.4008 |  162171 |
    | no2    | RandomForest               |   6.1430 |   8.0738 |   0.3916 |  162171 |
    | no2    | ARIMA                      |   4.8770 |   6.2235 |  -0.0018 |    1736 |
    | ozone  | GradientBoosting (weather) |   0.0072 |   0.0093 |   0.5990 |  223694 |
    | ozone  | RandomForest               |   0.0075 |   0.0094 |   0.5951 |  223694 |
    | ozone  | ARIMA                      |   0.0224 |   0.0253 |  -3.0465 |    1879 |

    ## Artifacts
    | File | Contents |
    |------|----------|
    | `gbm_<poll>.joblib` | Fitted GradientBoostingRegressor |
    | `rf_<poll>.joblib` | Fitted RandomForestRegressor |
    | `arima_<poll>.pkl` | Fitted statsmodels ARIMA results |
    | `metrics.json` | All MAE/RMSE/R² values (machine-readable) |
    | `model_card.md` | This document |
