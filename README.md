# Air Quality Platform

End-to-end air quality forecasting platform: EPA data ingestion → cleaning → feature engineering → GBM model training → REST prediction API → geographic bias audit → dashboard exports.

Predicts 24-hour-ahead concentrations of **PM2.5**, **NO2**, and **ozone** for any US monitoring station using 20 engineered features including lag/rolling statistics and live weather.

---

## Live Deployments

| Resource | Link |
|---|---|
| Prediction API (Cloud Run) | _coming soon_ |
| Tableau Public Dashboard | [Tableau Link](https://public.tableau.com/views/USAirQualityTrends2023/U_S_AirQualityTrends2023EPAMonitoringNetwork?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) |

---

## Architecture

```
EPA AQS API ──► data/raw/          ──► scripts/clean_data.py
Open-Meteo  ──► data/processed/    ──► scripts/train_weather.py
                                            │
                                     models/gbm_*.joblib
                                            │
AirNow API ──────────────────────► src/api/main.py  (/predict/live)
OpenWeather ─────────────────────►     │
                                   FastAPI → Cloud Run
```

---

## Model Performance (GBM, weather-augmented, 20 features)

80/20 time-based split — no temporal leakage. Test set is the last 20% of the 2023 timeline.

| Pollutant | MAE | RMSE | R² | n_test |
|---|---|---|---|---|
| PM2.5 (µg/m³) | 4.3781 | 8.0367 | 0.113 | 158,193 |
| NO2 (ppb) | 5.8871 | 8.0122 | 0.401 | 162,171 |
| Ozone (ppm) | 0.0072 | 0.0093 | 0.599 | 223,694 |

Weather features (temperature_2m, wind_speed_10m, relative_humidity_2m) improved all three models over the baseline. Full comparison: [`models/weather_ablation.json`](models/weather_ablation.json).

---

## Repository Layout

```
src/
  ingestion/
    aqs_client.py          EPA AQS bulk-pull client (hourly concentrations)
    airnow_client.py       AirNow real-time client (live /predict endpoint)
    historical_weather.py  Open-Meteo weather join for training data
    weather_client.py      Live weather fetch (OpenWeather + Open-Meteo fallback)
  cleaning/
    clean.py               Schema normalisation, gap-fill, outlier flagging
  features/
    build_features.py      20-feature engineering (lags, rolling stats, weather)
  models/
    train.py               GBM / RF / ARIMA training + time-split utilities
  api/
    main.py                FastAPI app (/predict, /predict/live, /health)
    schemas.py             Pydantic request/response models

scripts/
  pull_data.py             Pull raw AQS data with resume support
  clean_data.py            Run cleaning pipeline → data/processed/
  train_models.py          Train baseline models (17 features)
  train_weather.py         Retrain GBMs with weather features (20 features)
  bias_audit.py            Geographic bias audit → audits/
  export_for_dashboards.py Export CSVs → dashboards/data/
  export_powerbi_workbook.py  Bundle CSVs into Excel → powerbi_data.xlsx

models/
  gbm_pm25.joblib          Trained GBM — PM2.5  (gitignored binary)
  gbm_no2.joblib           Trained GBM — NO2    (gitignored binary)
  gbm_ozone.joblib         Trained GBM — ozone  (gitignored binary)
  metrics.json             Test-set metrics for all models
  model_card.md            Feature list, training details, known limitations
  weather_ablation.json    Baseline vs weather-augmented comparison

audits/
  bias_audit.json          Per-state / per-Census-region accuracy breakdown
  bias_audit.md            Plain-language audit summary

dashboards/data/
  air_quality_daily.csv    Daily mean per station (trends)
  monthly_state_summary.csv  Monthly mean per state
  model_metrics.csv        Flat metrics table
  bias_by_state.csv        Per-state GBM accuracy
  bias_by_region.csv       Per-region GBM accuracy
  station_aqi.csv          Per-station annual average + EPA AQI category
  powerbi_data.xlsx        Excel workbook (ModelMetrics, BiasByState, BiasByRegion)

deploy/
  cloudrun_deploy.sh       Build + push + deploy to Google Cloud Run
```

---

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install openpyxl        # for export_powerbi_workbook.py only
```

### 2. Configure credentials

```bash
cp .env.example .env
# Fill in EPA_AQS_EMAIL, EPA_AQS_KEY, AIRNOW_KEY, OPENWEATHER_KEY
```

### 3. Pull and prepare data

```bash
python scripts/pull_data.py --year 2023        # ~500k rows, ~90 API calls
python scripts/clean_data.py
python src/ingestion/historical_weather.py     # joins Open-Meteo weather
```

### 4. Train

```bash
python scripts/train_weather.py               # trains all 3 GBMs, saves metrics
```

### 5. Run the API locally

```bash
uvicorn src.api.main:app --reload --port 8080
```

Or with Docker:

```bash
docker build -t air-quality-api .
docker run --rm -p 8080:8080 --env-file .env air-quality-api
```

### 6. Export dashboard data

```bash
python scripts/bias_audit.py
python scripts/export_for_dashboards.py
python scripts/export_powerbi_workbook.py
```

---

## API Endpoints

### `POST /predict`

Predict 24h-ahead concentrations from a historical station observation.

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"station_id": "060374008", "timestamp_utc": "2023-08-15T14:00:00Z"}'
```

### `POST /predict/live`

Fetch real-time observations from AirNow + live weather, then predict.

```bash
curl -X POST http://localhost:8080/predict/live \
  -H "Content-Type: application/json" \
  -d '{"latitude": 34.0522, "longitude": -118.2437}'
```

Response includes `feature_provenance` (live / derived / unavailable) and `data_sources` metadata for every input.

### `GET /health`

```bash
curl http://localhost:8080/health
```

---

## Feature Engineering (20 features)

| Group | Features |
|---|---|
| Lag | lag_1h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_168h |
| Rolling mean | roll_mean_6h, roll_mean_24h, roll_mean_72h, roll_mean_168h |
| Rolling std | roll_std_24h, roll_std_168h |
| Temporal | hour_sin, hour_cos, day_of_year_sin, day_of_year_cos |
| Weather | temperature_2m (°C), wind_speed_10m (m/s), relative_humidity_2m (%) |

Weather values are current-hour observations aligned to each prediction row (no leakage into the 24h forecast horizon).

---

## Geographic Bias Audit

The audit measures GBM accuracy on the held-out test set broken down by US Census region (West / South / Midwest / Northeast) and state. Key findings:

| Pollutant | Best region | Worst region | Spread |
|---|---|---|---|
| PM2.5 | South (MAE 3.67) | West (MAE 5.16) | 1.41× |
| NO2 | South (MAE 5.26) | Northeast (MAE 6.78) | 1.29× |
| Ozone | West (MAE 0.0069) | Midwest (MAE 0.0078) | 1.13× |

No states were flagged as substantially worse (≥1.5× overall MAE) in any pollutant. Full breakdown: [`audits/bias_audit.md`](audits/bias_audit.md).

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `EPA_AQS_EMAIL` | Training | — | EPA AQS API email |
| `EPA_AQS_KEY` | Training | — | EPA AQS API key |
| `AIRNOW_KEY` | Live API | — | AirNow API key |
| `OPENWEATHER_KEY` | Live API | — | OpenWeatherMap API key (optional; Open-Meteo is the fallback) |
| `AIRNOW_TIMEOUT` | No | `60` | AirNow HTTP read timeout in seconds |
| `AIRNOW_BBOX_LAT` | No | `0.15` | Bounding-box half-height in degrees (~10 mi) |
| `AIRNOW_BBOX_LON` | No | `0.18` | Bounding-box half-width in degrees (~10 mi) |

---

## Deploy to Cloud Run

```bash
gcloud auth login
gcloud auth configure-docker
gcloud config set project YOUR_PROJECT_ID
bash deploy/cloudrun_deploy.sh
```

---

## Tests

```bash
pytest
```
