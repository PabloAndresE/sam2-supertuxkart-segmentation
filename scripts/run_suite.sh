#!/usr/bin/env bash
# Run the full suite sequentially on a single GPU box (no scheduler).
#
#   bash scripts/run_suite.sh                     # decisive suite
#   SUITE=configs/ablations.yaml bash scripts/run_suite.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SUITE="${SUITE:-configs/decisive.yaml}"
BASE="${BASE:-configs/server.yaml}"
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"

echo "==> preflight"
.venv/bin/python scripts/preflight.py --config "$BASE" --amp

echo "==> running $SUITE"
.venv/bin/python scripts/run_experiments.py \
  --suite "$SUITE" \
  --base "$BASE" \
  --amp

SUITE_NAME="$(basename "${SUITE%.*}")"
echo "==> evaluating each finished run on val"
for dir in runs/"$SUITE_NAME"/*/; do
  [ -f "$dir/decoder_best.pt" ] || continue
  .venv/bin/python -m stkseg.evaluate \
    --config "$dir/config.json" \
    --checkpoint "$dir/decoder_best.pt" \
    --split val \
    --out-dir "$dir" || echo "    evaluation failed for $dir"
done

.venv/bin/python scripts/run_experiments.py --suite "$SUITE" --aggregate-only
echo "==> done: runs/$SUITE_NAME/summary.md"
