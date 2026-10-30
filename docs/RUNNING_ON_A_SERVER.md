# Running the experiments on a GPU server

The suite is too slow for a laptop: a measured training epoch costs **245 s** on an
Apple M1 Pro (512px, batch 4), which is ~4 h per run and ~24 h for the suite. On a
datacentre GPU with AMP and a larger batch it is minutes per epoch.

Everything below assumes the repo is checked out on the server.

## 0. Clone

```bash
git clone https://github.com/PabloAndresE/sam2-supertuxkart-segmentation.git
cd sam2-supertuxkart-segmentation
```

The repository is private, so the server needs an authenticated `gh` (or a
personal access token in the clone URL). `gh auth login` once per machine.

## 1. Setup (on a login node)

Compute nodes are frequently firewalled, so fetch wheels and model weights first:

```bash
CUDA_TAG=cu124 bash scripts/setup_server.sh
```

This also calls `scripts/fetch_assets.sh`, which pulls the dataset (302 MB) and the
trained decoder (7.9 MB) from the GitHub releases. Neither is tracked in git.

This creates `.venv`, installs CUDA torch, installs the project, **pre-downloads the
SAM 2 and ConvNeXt weights into `.hf_cache/`**, and runs lint + tests. Set `CUDA_TAG`
to match the driver (`cu121`, `cu124`, `cu118`).

## 2. Dataset

Handled by `scripts/fetch_assets.sh` (run automatically by the setup script):

```bash
bash scripts/fetch_assets.sh            # dataset + weights
bash scripts/fetch_assets.sh --data     # dataset only
```

It downloads one 302 MB zip rather than 3001 loose files — per-file overhead dominates
any sync of this dataset (iCloud served the loose files at 0.6 files/s, 6.4 h total,
versus 3.5 s to unzip the archive).

Expect `seg_dataset/images` with 1501 files and `seg_dataset/masks` with 1500. The
script skips the download when the dataset is already complete, so it is safe to re-run.

## 3. Preflight

Never queue hours of GPU time without this. It verifies the dataset, the splits, that
the weights load **offline**, and measures a real training step to project the runtime
and the peak memory:

```bash
export HF_HOME=$PWD/.hf_cache
.venv/bin/python scripts/preflight.py --config configs/server.yaml --amp
```

Typical failure it catches: `batch_size: 16` OOM-ing on a 16 GB card. Lower
`data.batch_size` in `configs/server.yaml` to 8 and re-run.

## 4. Launch

### With SLURM (one array task per experiment, run in parallel)

```bash
JOB=$(sbatch --parsable scripts/slurm/suite.sbatch)
sbatch --dependency=afterany:$JOB scripts/slurm/aggregate.sbatch
squeue -u $USER
```

Add your cluster's partition and account: `sbatch -p gpu -A myaccount ...`.
The array range must match the suite length:

```bash
sbatch --export=ALL,SUITE=configs/ablations.yaml --array=0-14 scripts/slurm/suite.sbatch
```

### On a plain GPU box (sequential, no scheduler)

```bash
bash scripts/run_suite.sh                              # decisive suite
SUITE=configs/ablations.yaml bash scripts/run_suite.sh # full 15-run suite
```

Use `tmux` or `nohup` so an SSH drop does not kill it.

## 5. Results

```
runs/decisive/
  split_random/            config.json  history.json  decoder_best.pt
                           metrics_val.json  training_curves.png
                           confusion_val.png  predictions_val.png
  split_temporal_block/
  split_track_holdout/
  backbone_convnext_frozen/
  backbone_unet_scratch/
  summary.json
  summary.md               <- the table to paste into the README
```

Re-aggregate at any point, including while jobs are still running — partial suites
report whatever has finished:

```bash
.venv/bin/python scripts/run_experiments.py --suite configs/decisive.yaml --aggregate-only
```

## 6. The test split

Only after the model is chosen, once:

```bash
.venv/bin/python -m stkseg.evaluate \
  --config runs/decisive/split_temporal_block/config.json \
  --checkpoint runs/decisive/split_temporal_block/decoder_best.pt \
  --split test --confirm-final \
  --out-dir runs/decisive/split_temporal_block
```

Without `--confirm-final` it refuses. Each test read is appended to
`test_evaluations.jsonl` with a timestamp, so repeated peeking is visible in the repo.

## Bringing results home

```bash
rsync -av --exclude='*.pt' <user>@<host>:/path/to/repo/runs/ ./runs/
```

Excluding `*.pt` keeps it small; pull the single checkpoint you intend to publish
separately and attach it to a GitHub release.

## What has not been verified here

The SLURM scripts were written without access to a cluster. The Python they call is
tested; the `#SBATCH` directives, partition names and module system are cluster-specific
and will likely need one round of adjustment. `scripts/run_suite.sh` has no such
dependency and is the safer first attempt.
