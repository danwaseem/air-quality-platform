"""Pydantic request / response schemas for the Air Quality Prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Request schemas ───────────────────────────────────────────────────────────

class StationInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Station latitude (decimal degrees)")
    longitude: float = Field(..., ge=-180, le=180, description="Station longitude (decimal degrees)")


class ReadingsInput(BaseModel):
    """
    Historical pollutant readings used to build the feature vector.

    lag_1h and lag_24h are required.  The remaining rolling / weekly fields
    are optional — GradientBoosting handles their absence natively.
    RandomForest requires all fields; omitting any will return HTTP 422.
    """

    lag_1h: float = Field(
        ...,
        description="Pollutant value 1 hour before prediction_time",
    )
    lag_24h: float = Field(
        ...,
        description="Pollutant value 24 hours before prediction_time",
    )
    lag_168h: float | None = Field(
        None,
        description="Pollutant value 1 week (168 h) before prediction_time. "
                    "Omit if unavailable (GradientBoosting will handle NaN).",
    )
    roll_24h_mean: float | None = Field(
        None,
        description="Mean of the past 24 hourly readings (before prediction_time).",
    )
    roll_24h_std: float | None = Field(
        None,
        description="Std dev of the past 24 hourly readings.",
    )
    roll_168h_mean: float | None = Field(
        None,
        description="Mean of the past 7 days of hourly readings.",
    )
    roll_168h_std: float | None = Field(
        None,
        description="Std dev of the past 7 days of hourly readings.",
    )
    drift_flag: int = Field(
        0,
        ge=0,
        le=1,
        description="Set to 1 if an anomalous sensor drift was detected upstream.",
    )


class PredictRequest(BaseModel):
    pollutant: Literal["pm25", "no2", "ozone"] = Field(
        ...,
        description="Target pollutant to predict.",
    )
    model: Literal["GradientBoosting", "RandomForest"] | None = Field(
        None,
        description="Model variant. Defaults to GradientBoosting. "
                    "RandomForest requires all optional readings fields to be non-null.",
    )
    station: StationInput
    readings: ReadingsInput
    prediction_time: datetime | None = Field(
        None,
        description="UTC timestamp of the most recent reading. "
                    "Defaults to the current UTC time. "
                    "The prediction target is 24 h after this time.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pollutant": "pm25",
                    "station": {"latitude": 33.484, "longitude": -112.143},
                    "readings": {
                        "lag_1h": 12.5,
                        "lag_24h": 10.2,
                        "lag_168h": 11.0,
                        "roll_24h_mean": 11.3,
                        "roll_24h_std": 2.1,
                        "roll_168h_mean": 10.9,
                        "roll_168h_std": 1.8,
                        "drift_flag": 0,
                    },
                    "prediction_time": "2023-11-15T14:00:00Z",
                }
            ]
        }
    }


# ── Response schemas ──────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    mae: float | None = Field(None, description="Mean absolute error on held-out test set")
    rmse: float | None = Field(None, description="Root mean squared error on test set")
    r2: float | None = Field(None, description="R² coefficient on test set")
    n_test: int | None = Field(None, description="Number of test-set rows used for evaluation")


class PredictResponse(BaseModel):
    pollutant: str
    unit: str = Field(..., description="Physical unit of the prediction value")
    prediction_24h_ahead: float = Field(
        ...,
        description="Predicted pollutant concentration 24 hours after prediction_time",
    )
    model_used: str
    model_metrics: ModelMetrics = Field(
        ...,
        description="Test-set performance of the model used, for context",
    )
    prediction_time: str = Field(..., description="ISO-8601 UTC timestamp of the input readings")
    target_time: str = Field(..., description="ISO-8601 UTC timestamp of the prediction target")
    features_used: dict[str, float | None] = Field(
        ...,
        description="Assembled feature vector passed to the model",
    )


class ModelInfo(BaseModel):
    pollutant: str
    available_models: list[str]
    metrics: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: dict[str, list[str]]
    total_models: int


class RootResponse(BaseModel):
    name: str
    version: str
    description: str
    docs_url: str
    endpoints: list[str]


# ── /predict/live schemas ─────────────────────────────────────────────────────

class LivePredictRequest(BaseModel):
    """
    Request body for POST /predict/live.

    The endpoint fetches current AirNow observations for the location,
    constructs lag/rolling features automatically, and returns a 24h prediction.
    """
    pollutant: Literal["pm25", "no2", "ozone"] = Field(
        ...,
        description="Target pollutant to predict.",
    )
    latitude: float = Field(..., ge=-90,  le=90,  description="Decimal degrees, WGS84")
    longitude: float = Field(..., ge=-180, le=180, description="Decimal degrees, WGS84")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pollutant": "pm25",
                    "latitude":  37.3382,
                    "longitude": -121.8863,
                }
            ]
        }
    }


# Literal union for the three provenance labels
ProvenanceLabel = Literal["live", "derived", "input", "unavailable/backfilled", "derived/assumed"]


class LivePredictResponse(BaseModel):
    """
    Response from POST /predict/live.

    Every feature fed to the model is labelled with its provenance so callers
    know which inputs are genuinely real-time vs. inferred or missing.
    """
    pollutant:            str
    unit:                 str
    prediction_24h_ahead: float
    model_used:           str
    model_metrics:        ModelMetrics

    prediction_time: str = Field(..., description="UTC time of most recent observation used as 't=0'")
    target_time:     str = Field(..., description="UTC time of the 24h-ahead prediction target")

    features_used:      dict[str, float | None]

    feature_provenance: dict[str, str] = Field(
        ...,
        description=(
            "Per-feature data source label:\n"
            "  'live'                  — directly observed from AirNow API\n"
            "  'derived'               — computed from the live AirNow time series\n"
            "  'input'                 — taken directly from request (lat/lon)\n"
            "  'unavailable/backfilled' — not available; NaN passed to model "
            "(GBM handles this natively)\n"
            "  'derived/assumed'       — inferred with an assumption (e.g. drift_flag=0)"
        ),
    )

    data_sources: dict[str, Any] = Field(
        ...,
        description="Metadata about the real-time data retrieved.",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during feature assembly.",
    )
