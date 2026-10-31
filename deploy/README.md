# Deploying the demo

Measured peak RSS for CPU inference is **1.03 GB**, which decides the host:

| Host | Free memory | Verdict |
|---|---|---|
| Railway | no free tier since 2023 (trial credit only) | no |
| Render free | 512 MB | OOM |
| Fly.io free | 256 MB | OOM |
| Koyeb free | 512 MB | OOM |
| Hugging Face Spaces | Gradio/Docker need PRO | paid |
| **Google Cloud Run** | up to 32 GB, scales to zero | **yes** |

Cloud Run's always-free tier covers 360,000 GiB-seconds and 180,000 vCPU-seconds per
month. At the 2 GiB this service needs, that is roughly 50 hours of active container
time — far beyond what a portfolio demo uses, since the service scales to zero between
requests. **A billing account must still be attached**, free tier or not.

## Deploy

```bash
gcloud auth login                      # personal account, not a work one
gcloud config set project <project-id>
PROJECT=<project-id> bash deploy/deploy_cloudrun.sh
```

The script enables the required APIs, builds with Cloud Build (nothing is built on your
machine) and deploys with 2 GiB, 2 vCPU, CPU boost and `--min-instances 0`.

## Why the image bakes in the weights

Both the frozen SAM 2 encoder and the trained decoder are downloaded at **build** time.
Fetching them on first request would add 30+ seconds to an already slow cold start, and
Cloud Run gives no persistent disk to cache them between instances.

`HF_HUB_OFFLINE=1` is set in the image, so a missing cache fails the build loudly rather
than failing a user request silently.

## Cold starts

The first request after idle takes roughly 30-60 s, dominated by importing torch.
Subsequent requests are a few seconds. If that matters, `--min-instances 1` removes it —
and leaves the free tier, since the container then runs continuously.

## Cheaper alternative

The [Colab notebook](../notebooks/demo_colab.ipynb) needs no infrastructure, no billing
account and no cold start, and gives reviewers a free GPU. Cloud Run is worth it only if
you want a permanent URL.
