#!/usr/bin/env bash
# API smoke-test examples.
# Run against the local dev server:  bash docs/api_examples.sh
# Or set BASE_URL to the Cloud Run URL for production tests.

BASE_URL="${BASE_URL:-http://localhost:8080}"

echo "──────────────────────────────────────────────"
echo " GET /health"
echo "──────────────────────────────────────────────"
curl -s "${BASE_URL}/health" | python3 -m json.tool

echo ""
echo "──────────────────────────────────────────────"
echo " GET /models  (lists loaded models + metrics)"
echo "──────────────────────────────────────────────"
curl -s "${BASE_URL}/models" | python3 -m json.tool

echo ""
echo "──────────────────────────────────────────────"
echo " POST /predict  (manual feature input)"
echo " Model: GradientBoosting (20 features)"
echo " Weather features absent → NaN passthrough"
echo "──────────────────────────────────────────────"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pollutant": "pm25",
    "station":   {"latitude": 33.484, "longitude": -112.143},
    "readings": {
      "lag_1h":         12.5,
      "lag_24h":        10.2,
      "lag_168h":       11.0,
      "roll_24h_mean":  11.3,
      "roll_24h_std":    2.1,
      "roll_168h_mean": 10.9,
      "roll_168h_std":   1.8,
      "drift_flag":      0
    },
    "prediction_time": "2023-11-15T14:00:00Z"
  }' | python3 -m json.tool

echo ""
echo "──────────────────────────────────────────────"
echo " POST /predict  — NO2, all optional fields"
echo "──────────────────────────────────────────────"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pollutant": "no2",
    "station":   {"latitude": 29.749, "longitude": -95.367},
    "readings": {
      "lag_1h":         18.0,
      "lag_24h":        14.5,
      "lag_168h":       16.0,
      "roll_24h_mean":  16.2,
      "roll_24h_std":    3.4,
      "roll_168h_mean": 15.8,
      "roll_168h_std":   2.9,
      "drift_flag":      0
    }
  }' | python3 -m json.tool

echo ""
echo "──────────────────────────────────────────────"
echo " POST /predict/live  — PM2.5, San Jose CA"
echo " Fetches: AirNow observations (lag/rolling)"
echo "          + live weather from OpenWeather"
echo "            or Open-Meteo (free fallback)"
echo " All 20 features assembled automatically."
echo " Requires AIRNOW_KEY in .env"
echo "──────────────────────────────────────────────"
curl -s -X POST "${BASE_URL}/predict/live" \
  -H "Content-Type: application/json" \
  -d '{
    "pollutant": "pm25",
    "latitude":  37.3382,
    "longitude": -121.8863
  }' | python3 -m json.tool

echo ""
echo "──────────────────────────────────────────────"
echo " POST /predict/live  — NO2, Houston TX"
echo "──────────────────────────────────────────────"
curl -s -X POST "${BASE_URL}/predict/live" \
  -H "Content-Type: application/json" \
  -d '{
    "pollutant": "no2",
    "latitude":  29.7604,
    "longitude": -95.3698
  }' | python3 -m json.tool

echo ""
echo "──────────────────────────────────────────────"
echo " POST /predict/live  — Ozone, Phoenix AZ"
echo "──────────────────────────────────────────────"
curl -s -X POST "${BASE_URL}/predict/live" \
  -H "Content-Type: application/json" \
  -d '{
    "pollutant": "ozone",
    "latitude":  33.4484,
    "longitude": -112.0740
  }' | python3 -m json.tool
