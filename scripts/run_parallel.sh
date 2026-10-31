#!/usr/bin/env bash
# Run several suites on one multi-GPU box without a scheduler, N runs at a time.
#
#   GPUS="0 1" PAR=6 bash scripts/run_parallel.sh configs/seeds.yaml configs/ablations.yaml
#
# Each run trains with --only, then evaluates on val (never test). Runs are dealt
# round-robin over GPUS. Events go to logs/queue/_events.log, one log per run beside it.
#
# On a shared host keep OMP_NUM_THREADS low: unbounded, each run took ~26 cores and
# saturated a 224-core machine, leaving the GPUs idle. Thread count does not change the
# protocol.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/.hf_cache}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export BASE="${BASE:-configs/server.yaml}" GPUS="${GPUS:-0}"
mkdir -p logs/queue

job() {
  local idx=$1 suite=$2 name=$3
  local gpus=($GPUS)
  local gpu=${gpus[$(( idx % ${#gpus[@]} ))]}
  local sname; sname=$(basename "${suite%.*}")
  local log=logs/queue/${sname}__${name}.log
  local dir=runs/${sname}/${name}
  echo "[$(date -u +%T)] start $sname/$name gpu=$gpu" | tee -a logs/queue/_events.log
  local t0; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/run_experiments.py \
      --suite "$suite" --base "$BASE" --only "$name" --amp >"$log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m stkseg.evaluate \
      --config "$dir/config.json" --checkpoint "$dir/decoder_best.pt" \
      --split val --out-dir "$dir" >>"$log" 2>&1
  local rc=$?
  echo "[$(date -u +%T)] end   $sname/$name gpu=$gpu rc=$rc secs=$(( $(date +%s) - t0 ))" \
    | tee -a logs/queue/_events.log
}
export -f job

for suite in "$@"; do
  .venv/bin/python - "$suite" <<'PY'
import sys, yaml
for entry in yaml.safe_load(open(sys.argv[1])):
    print(sys.argv[1], entry["name"])
PY
done | nl -v0 -w1 -s' ' | xargs -P "${PAR:-1}" -L1 bash -c 'job "$0" "$1" "$2"'

for suite in "$@"; do
  .venv/bin/python scripts/run_experiments.py --suite "$suite" --aggregate-only >/dev/null
done
echo "==> done; failures:"
grep ' end ' logs/queue/_events.log | grep -v 'rc=0' || echo "    none"
