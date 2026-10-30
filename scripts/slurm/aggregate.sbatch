#!/bin/bash
#SBATCH --job-name=stkseg-agg
#SBATCH --output=logs/stkseg_agg_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#
# Collect whatever the array finished into one table. Submit it with a
# dependency so it runs after the suite:
#
#   JOB=$(sbatch --parsable scripts/slurm/suite.sbatch)
#   sbatch --dependency=afterany:$JOB scripts/slurm/aggregate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SUITE="${SUITE:-configs/decisive.yaml}"
.venv/bin/python scripts/run_experiments.py --suite "$SUITE" --aggregate-only
cat "runs/$(basename "${SUITE%.*}")/summary.md"
