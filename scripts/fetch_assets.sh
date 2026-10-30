#!/usr/bin/env bash
# Pull the dataset and the trained decoder from the repo's GitHub releases.
#
# Neither is tracked in git: the dataset is 302 MB and weights do not belong in
# history. Run this straight after cloning on a new machine.
#
#   bash scripts/fetch_assets.sh            # dataset + weights
#   bash scripts/fetch_assets.sh --data     # dataset only
#   bash scripts/fetch_assets.sh --weights  # weights only
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

WANT_DATA=1
WANT_WEIGHTS=1
case "${1:-}" in
  --data) WANT_WEIGHTS=0 ;;
  --weights) WANT_DATA=0 ;;
  "") ;;
  *) echo "usage: $0 [--data|--weights]" >&2; exit 2 ;;
esac

if ! command -v gh >/dev/null; then
  echo "gh CLI not found. Install it, or download the assets manually from" >&2
  echo "  https://github.com/PabloAndresE/sam2-supertuxkart-segmentation/releases" >&2
  exit 1
fi
# The repository is private, so an authenticated gh is required to read assets.
gh auth status >/dev/null 2>&1 || { echo "run 'gh auth login' first" >&2; exit 1; }

if [ "$WANT_DATA" = 1 ]; then
  if [ -d seg_dataset ] && [ "$(find seg_dataset -name '*.png' | wc -l | tr -d ' ')" -ge 3000 ]; then
    echo "==> seg_dataset/ already present, skipping"
  else
    echo "==> downloading dataset (302 MB)"
    gh release download data-v1 --pattern 'seg_dataset.zip' --clobber
    unzip -q -o seg_dataset.zip -d .
    rm -f seg_dataset.zip
    echo "    images=$(find seg_dataset/images -name '*.png' | wc -l | tr -d ' ')" \
         "masks=$(find seg_dataset/masks -name '*.png' | wc -l | tr -d ' ')"
  fi
fi

if [ "$WANT_WEIGHTS" = 1 ]; then
  echo "==> downloading trained decoder (7.9 MB)"
  mkdir -p runs/baseline
  gh release download v0.1.0 --pattern 'decoder_best.pt' --dir runs/baseline --clobber
fi

echo "==> done"
