# Cloud Run Deployment

Deploys the `air-quality-api` container to Google Cloud Run using `cloudrun_deploy.sh`.

## Prerequisites

```bash
gcloud auth login
gcloud auth configure-docker
gcloud config set project air-quality-platform-500007
```

## Run

```bash
cd deploy/
bash cloudrun_deploy.sh
```

The script runs from the `deploy/` directory and passes `..` as the build context
so `gcloud builds submit` picks up the `Dockerfile` at the repo root.

## What each flag does

| Flag | Value | Reason |
|---|---|---|
| `--image` | `gcr.io/$PROJECT_ID/air-quality-api` | Pushed to Container Registry in your project |
| `--platform managed` | — | Fully managed Cloud Run (no Kubernetes cluster needed) |
| `--allow-unauthenticated` | — | Public endpoint; remove this to require a bearer token |
| `--port 8080` | 8080 | Must match the `EXPOSE` and `$PORT` in the Dockerfile |
| `--memory 1Gi` | 1 GB | GBM models are ~3 MB total; 1 Gi is comfortable headroom |
| `--min-instances 0` | 0 | **Scales to zero when idle — keeps cost at ~$0 between requests** |
| `--max-instances 2` | 2 | Caps concurrency to avoid runaway billing during testing |

## Cost note

With `--min-instances 0` the service incurs no charges when no requests are being
served. The first request after an idle period takes ~2–3 s to cold-start (container
boot + model load). Set `--min-instances 1` to eliminate cold starts at the cost of
one always-on instance (~$10–15/month on us-central1).

## Updating after retraining

Rebuilding and redeploying is a single command — Cloud Run performs a zero-downtime
rollout automatically:

```bash
bash deploy/cloudrun_deploy.sh
```

## Restricting access

To lock the endpoint to authenticated callers only, remove `--allow-unauthenticated`
from the script and call the API with:

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     https://<SERVICE_URL>/health
```
