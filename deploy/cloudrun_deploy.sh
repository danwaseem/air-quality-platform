#!/usr/bin/env bash
# Deploy the air-quality-api container to Google Cloud Run.
# Prerequisites:
#   gcloud auth login
#   gcloud auth configure-docker
#   gcloud config set project $PROJECT_ID
set -euo pipefail

# Repo root is the parent of this script's directory — works regardless of cwd
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT_ID="air-quality-platform-500007"
REGION="us-central1"
SERVICE="air-quality-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "==> Building and pushing image to GCR …"
gcloud builds submit \
    --tag "${IMAGE}" \
    --project "${PROJECT_ID}" \
    "${REPO_ROOT}"

echo ""
echo "==> Deploying to Cloud Run …"
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --min-instances 0 \
    --max-instances 2 \
    --project "${PROJECT_ID}"

echo ""
echo "==> Fetching service URL …"
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format "value(status.url)")

echo "Service URL: ${SERVICE_URL}"
echo ""
echo "==> Verifying /health …"
curl -sf "${SERVICE_URL}/health" | python3 -m json.tool

echo ""
echo "Deploy complete."
