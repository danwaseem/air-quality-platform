# ── Air Quality Prediction API ────────────────────────────────────────────────
# python:3.11-slim keeps the image lean while matching a stable sklearn ABI.
# Cloud Run injects PORT at runtime; we default to 8080 for local `docker run`.
#
# RF models (900 MB–1.2 GB each) are deliberately excluded — only the
# lightweight GBM artifacts (~1 MB each) are baked in.
# To serve RF models, mount them as a Cloud Run volume or GCS-backed secret.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.14-slim

WORKDIR /app

# ── System deps (none beyond python:slim defaults) ────────────────────────────

# ── Python runtime dependencies ───────────────────────────────────────────────
# Install only what the API needs at serving time (no training deps).
# Pinned minor versions so rebuilds are deterministic.
# Versions must match the training environment exactly — joblib pickles are
# sensitive to numpy and sklearn major versions.
# Training env: Python 3.14.5 / numpy 2.4.6 / sklearn 1.9.0 / pandas 3.0.3
RUN pip install --no-cache-dir \
    "fastapi>=0.111" \
    "uvicorn[standard]>=0.29" \
    "pandas~=3.0" \
    "numpy~=2.4" \
    "scikit-learn~=1.9" \
    "joblib>=1.3" \
    "python-dotenv>=1.0" \
    "requests>=2.31"

# ── Application source ────────────────────────────────────────────────────────
COPY src/ src/

# ── Model artifacts ───────────────────────────────────────────────────────────
# Only the small GBM files (~1 MB each) are included.
# RF models are excluded via .dockerignore.
COPY models/gbm_pm25.joblib  models/
COPY models/gbm_no2.joblib   models/
COPY models/gbm_ozone.joblib models/
COPY models/metrics.json     models/
COPY models/model_card.md    models/

# ── Runtime config ────────────────────────────────────────────────────────────
# PORT is injected by Cloud Run; default 8080 for local docker run.
ENV PORT=8080
ENV MODELS_DIR=models
ENV MAX_MODEL_SIZE_MB=200

EXPOSE 8080

# Use shell form so ${PORT} is expanded at container start time.
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]
