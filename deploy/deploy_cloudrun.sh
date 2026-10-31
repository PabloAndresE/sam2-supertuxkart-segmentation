#!/usr/bin/env bash
# Build and deploy the demo to Cloud Run.
#
#   PROJECT=my-project bash deploy/deploy_cloudrun.sh
#
# Cost: the service scales to zero, so it bills only while a request is in
# flight plus a short idle window. The always-free tier covers 360,000 GiB-s and
# 180,000 vCPU-s per month, which at 2 GiB is about 50 hours of active
# container time -- far more than a portfolio demo uses.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-stkseg-demo}"

if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  echo "set PROJECT, or run: gcloud config set project <id>" >&2
  exit 1
fi

echo "==> project=$PROJECT region=$REGION service=$SERVICE"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

IMAGE="gcr.io/$PROJECT/$SERVICE"
echo "==> building $IMAGE with Cloud Build"
gcloud builds submit --project "$PROJECT" --tag "$IMAGE" --timeout=30m \
  --config /dev/null . 2>/dev/null \
  || gcloud builds submit --project "$PROJECT" --tag "$IMAGE" --timeout=30m \
       --file deploy/Dockerfile .

echo "==> deploying"
# 2 GiB: measured peak RSS is ~1.03 GB. 1 GiB OOMs on the first request.
# cpu-boost shortens the cold start, which is dominated by importing torch.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --image "$IMAGE" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --cpu-boost \
  --concurrency 4 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8080

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
      --region "$REGION" --format 'value(status.url)')
echo
echo "==> live at $URL"
echo "    first request cold-starts in roughly 30-60 s; later ones are a few seconds"
