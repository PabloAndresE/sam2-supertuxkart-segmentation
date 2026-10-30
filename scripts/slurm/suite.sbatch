#!/bin/bash
#SBATCH --job-name=stkseg
#SBATCH --output=logs/stkseg_%A_%a.out
#SBATCH --error=logs/stkseg_%A_%a.err
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# One array task per experiment, so the five runs execute in parallel instead of
# queueing behind each other. Adjust --array to match the suite length:
#   python -c "import yaml;print(len(yaml.safe_load(open('configs/decisive.yaml')))-1)"
#
#   sbatch scripts/slurm/suite.sbatch
#   sbatch --export=ALL,SUITE=configs/ablations.yaml --array=0-14 scripts/slurm/suite.sbatch
#
# Partition and account are cluster-specific; add them here or pass on the
# command line, e.g. sbatch -p gpu -A myaccount scripts/slurm/suite.sbatch

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p logs

SUITE="${SUITE:-configs/decisive.yaml}"
BASE="${BASE:-configs/server.yaml}"
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"

# Pick this task's experiment by index.
NAME=$(.venv/bin/python - "$SUITE" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys, yaml
entries = yaml.safe_load(open(sys.argv[1]))
print(entries[int(sys.argv[2])]["name"])
PY
)

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} -> ${NAME} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv/bin/python scripts/run_experiments.py \
  --suite "$SUITE" \
  --base "$BASE" \
  --only "$NAME" \
  --amp

SUITE_NAME="$(basename "${SUITE%.*}")"
RUN_DIR="runs/${SUITE_NAME}/${NAME}"
.venv/bin/python -m stkseg.evaluate \
  --config "${RUN_DIR}/config.json" \
  --checkpoint "${RUN_DIR}/decoder_best.pt" \
  --split val \
  --out-dir "${RUN_DIR}"

echo "[$(date)] ${NAME} finished"
