#!/usr/bin/env bash
# One-shot environment setup for a CUDA node.
#
# Run this on a LOGIN node, not a compute node: it needs internet to fetch the
# wheels and the SAM 2 encoder, and many clusters firewall compute nodes.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CUDA_TAG="${CUDA_TAG:-cu124}"
PY="${PYTHON:-python3}"

echo "==> creating .venv with $($PY --version)"
"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip wheel

echo "==> installing torch for $CUDA_TAG"
.venv/bin/pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> installing the project"
.venv/bin/pip install -e ".[dev]"

# Compute nodes are often offline. Warm the Hugging Face cache here so the
# encoder is on disk before any job starts, and point HF_HOME at the repo so
# the cache travels with it.
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
echo "==> warming the encoder cache into $HF_HOME"
.venv/bin/python - <<'PY'
from transformers import Sam2Model
Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
print("SAM 2 encoder cached")
PY
.venv/bin/python - <<'PY'
from torchvision.models import convnext_tiny
convnext_tiny(weights="DEFAULT")
print("ConvNeXt-Tiny cached")
PY

echo "==> dataset and weights"
if [ ! -d seg_dataset ]; then
  bash scripts/fetch_assets.sh
else
  echo "    seg_dataset/ present: $(find seg_dataset -name '*.png' | wc -l | tr -d ' ') png files"
fi

echo
echo "==> verifying"
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q

cat <<EOF

Setup complete. Next:

  export HF_HOME=$HF_HOME
  .venv/bin/python scripts/preflight.py --config configs/server.yaml --amp
  sbatch scripts/slurm/suite.sbatch          # or: bash scripts/run_suite.sh

EOF
