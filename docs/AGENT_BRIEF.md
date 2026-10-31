# Agent brief: run the decisive experiment suite on this GPU server

You are running on a university GPU server. Everything below is self-contained — do not
assume knowledge of any prior conversation.

## What this repository is

Semantic segmentation of SuperTuxKart frames into 7 classes, using a **frozen** SAM 2
Hiera-Tiny vision encoder with a small trainable decoder on top (2.07M of 29.3M
parameters, 7%). The code is installable, tested (42 tests) and has CLIs for training,
evaluation and inference.

## Why you are here

The repository currently reports **mIoU 0.8534**, and that number is not trustworthy:

1. It was produced with a **uniformly random split**, but the dataset is 6 tracks ×
   250 *consecutive video frames*. Neighbouring frames differ by MAE 0.066 versus 0.126
   for distant ones. Under `random`, **691 adjacent frame pairs straddle a split
   boundary**; under `temporal_block` only 26; under `track_holdout`, zero.
2. All 6 tracks appear in both train and validation, so it never measured generalisation
   to unseen scenery.
3. The checkpoint was **selected** by validation mIoU and then that same validation mIoU
   was reported — a selection score, not a generalisation estimate.

The corrected protocol is already implemented. Your job is to **run it and replace the
legacy number with measured results.**

## The two questions the suite answers

`configs/decisive.yaml` defines 5 runs, all at 512px / 60 epochs:

| Run | Question |
|---|---|
| `split_random`, `split_temporal_block`, `split_track_holdout` | How much did the split protocol inflate 0.8534? |
| `split_temporal_block` (= SAM 2), `backbone_convnext_frozen`, `backbone_unet_scratch` | Is a frozen backbone worth it, and does it have to be SAM 2? |

The ConvNeXt baseline is the important one: 27.8M frozen / 2.41M trainable versus SAM 2's
27.2M / 2.07M. Near-identical budgets, so any gap is attributable to what the backbone
learned, not to capacity.

## Hard rules — do not violate these

- **Never read the test split except once, at the very end, on the single chosen model.**
  `stkseg.evaluate --split test` refuses without `--confirm-final` and appends every test
  read to `test_evaluations.jsonl`. That gate exists on purpose. Do not remove it, do not
  loop over it, do not use test scores to pick anything.
- **Do not change the protocol partway through a suite.** Every run must share the same
  epochs, resolution and schedule or the comparison is void. If you must reduce the
  budget (time limit, queue pressure), reduce it for *all* runs and say so explicitly in
  the README.
- **Report only numbers you measured.** Never estimate, interpolate or carry over a
  figure from this brief into results. If a run failed, say it failed.
- **Keep failure classes in the mean.** `Projectile` appears in 2 of 1500 frames and
  scores 0.0000. It stays in the reported mIoU with a sentence explaining why. Do not
  drop it to lift the average.
- **Do not compare across protocols in one table.** The legacy 0.8534 and your new
  numbers belong in separate rows that are clearly labelled, or separate tables.

## Procedure

### 1. Setup (login node — compute nodes are often firewalled)

```bash
CUDA_TAG=cu124 bash scripts/setup_server.sh     # match your driver: cu118 / cu121 / cu124
export HF_HOME=$PWD/.hf_cache
```

This creates `.venv`, installs CUDA torch, pre-downloads the SAM 2 and ConvNeXt weights
into `.hf_cache/` (critical if compute nodes have no internet), fetches the dataset and
trained decoder from GitHub releases, and runs lint + tests.

Verify: `find seg_dataset -name '*.png' | wc -l` should be 3001.

### 2. Preflight — mandatory before queueing anything

```bash
.venv/bin/python scripts/preflight.py --config configs/server.yaml --amp
```

It checks the dataset, builds the splits, confirms the weights load **offline**, and
times real dataloader batches to project runtime and peak memory.

- If it reports OOM, lower `data.batch_size` in `configs/server.yaml` (16 → 8 → 4) and
  re-run until it passes.
- **The epoch estimate is a lower bound.** It warms the page cache; a measured Apple M1
  Pro epoch came in 1.6× above its own estimate. Size SLURM `--time` with headroom.

### 3. Launch

SLURM, one array task per experiment:

```bash
JOB=$(sbatch --parsable scripts/slurm/suite.sbatch)
sbatch --dependency=afterany:$JOB scripts/slurm/aggregate.sbatch
squeue -u $USER
```

**The `#SBATCH` directives in `scripts/slurm/suite.sbatch` were written without access to
a cluster and are the most likely thing to need fixing.** Expect to add `--partition`
and `--account`, and possibly `module load` lines. Inspect the cluster's documentation or
an existing working job script before submitting. The `--array` range must match the
suite length (currently `0-4` for 5 entries).

If SLURM is unavailable or fights you, there is no-scheduler fallback that is known to
work:

```bash
tmux new -s stkseg
bash scripts/run_suite.sh
```

### 4. Monitor

Partial suites aggregate fine — you do not need to wait for everything:

```bash
.venv/bin/python scripts/run_experiments.py --suite configs/decisive.yaml --aggregate-only
cat runs/decisive/summary.md
```

### 5. Evaluate each finished run on validation

`scripts/run_suite.sh` does this automatically; the SLURM array does it per task. If you
ran something manually:

```bash
for dir in runs/decisive/*/; do
  [ -f "$dir/decoder_best.pt" ] || continue
  .venv/bin/python -m stkseg.evaluate \
    --config "$dir/config.json" --checkpoint "$dir/decoder_best.pt" \
    --split val --out-dir "$dir"
done
```

This writes `metrics_val.json` (per-class IoU, boundary IoU), `confusion_val.png` and
`predictions_val.png` per run.

### 6. Test split — once, at the end

Pick the model **on validation**, then:

```bash
D=runs/decisive/split_temporal_block
.venv/bin/python -m stkseg.evaluate \
  --config "$D/config.json" --checkpoint "$D/decoder_best.pt" \
  --split test --confirm-final --out-dir "$D"
```

### 7. Update the README

Replace the "Results" section. It currently carries a `> Status:` blockquote saying the
headline number predates the protocol — that blockquote must go once you have real
numbers. Write:

- A table of val mIoU per split strategy, so the inflation from `random` is visible as a
  number.
- A table of val mIoU per backbone at matched parameter budgets.
- Per-class IoU **and boundary IoU** for the chosen model.
- The single test-split score, labelled as a one-shot final measurement.
- Keep the existing Limitations section and update it: remove items you resolved, add
  any new ones (for example, still a single seed unless you also ran `configs/seeds.yaml`).

Paste the figures that `evaluate` produced into `docs/assets/` and reference them.

### 8. Commit and push

```bash
git checkout -b results/decisive-suite
git add README.md docs/assets runs/decisive/summary.md runs/decisive/*/metrics_*.json
git commit -m "Report measured results under the corrected split protocol"
git push -u origin results/decisive-suite
```

Do **not** commit `*.pt` checkpoints, `seg_dataset/`, or `.hf_cache/` — they are
gitignored; keep it that way. If you lack push rights, write the summary to
`runs/decisive/summary.md` and print it so it can be copied manually.

## Optional, if GPU time is cheap

```bash
# variance: 3 seeds of the headline config
sbatch --export=ALL,SUITE=configs/seeds.yaml --array=0-2 scripts/slurm/suite.sbatch

# the full 15-run ablation suite: FPN levels, loss composition, oversampling
sbatch --export=ALL,SUITE=configs/ablations.yaml --array=0-14 scripts/slurm/suite.sbatch
```

A variance estimate matters: without it, "0.853 vs 0.845" is not a result.

## Report back

State plainly:

1. Which runs completed, which failed, and why.
2. The `runs/decisive/summary.md` table verbatim.
3. Measured mIoU under each split strategy, and how far the legacy 0.8534 fell.
4. SAM 2 versus ConvNeXt-frozen at matched budgets — and whether the gap is large enough
   to mean anything given you likely have a single seed.
5. Any `#SBATCH` changes you had to make, so the scripts can be fixed in the repo.
6. Wall-clock cost per run.


---

# Round 2: what the first suite changed

The first suite ran 29 jobs and settled some questions while exposing two defects in
this repository's own protocol. Both are now fixed in code; the brief below supersedes
the "Optional" section above.

## What the first suite found

- **Neighbour leakage did not measurably inflate the score.** On the six classes every
  split shares, `temporal_block` (0.8897) scored *higher* than `random` (0.8630). The
  0.10 gap in the seven-class column was entirely the `Projectile` term. The original
  hypothesis — that 691 adjacent frame pairs crossing split boundaries inflated the
  0.8534 — is not supported. Say so plainly; do not quietly drop it.
- **Unseen scenery is the real gap.** `track_holdout` scored 0.6910, about 0.20 below
  `temporal_block` on the same classes.
- **The selection score lied by 0.117.** Same split, same seed, same six classes:
  validation 0.8897, test 0.7731.
- **SAM 2 over frozen ConvNeXt: +0.0193 ± 0.0129** paired across four splits, 95% CI
  [-0.001, +0.040]. Consistent in direction, not yet significant.

## Two defects that are now fixed

1. **`data.seed` drove both the split and the weight initialisation**, so the first seed
   sweep varied both at once and could not separate them. `data.init_seed` now exists and
   defaults to `data.seed` for backward compatibility.
2. **`mIoU` averages over whichever classes are present**, so splits with different class
   sets are not comparable, and a hallucinated class enters at 0.0. `evaluate` now also
   reports **`mIoU_shared6`** over the six classes every split contains. The old `mIoU`
   is unchanged, so previously published numbers still mean what they said.

Also new: `make_splits(..., holdout_tracks=("val_track", "test_track"))` pins which
tracks are held out instead of drawing them from the seed.

## Run these, in this order

### 1. Leave-one-track-out (6 runs) — highest value

The headline "unseen scenery costs 0.20" rests on **one** held-out track. Six runs make
every track the unseen one exactly once, turning a single draw into a mean with a spread.

```bash
sbatch --export=ALL,SUITE=configs/loto.yaml --array=0-5 scripts/slurm/suite.sbatch
```

Report mean ± sd of `mIoU_shared6` across the six, and the per-track spread. If the cost
varies a lot by track, that is itself the finding.

### 2. Initialisation noise on a fixed split (3 runs)

```bash
sbatch --export=ALL,SUITE=configs/init_seeds.yaml --array=0-2 scripts/slurm/suite.sbatch
```

The previous ±0.0022 rerun estimate came from accidental duplicates inside the ablation
suite. This measures it deliberately: `seed` pinned at 42, `init_seed` varying.

### 3. More paired backbone splits (optional, 12 runs)

Only if 1 and 2 are done and GPU time is free. Extends the paired comparison toward
significance. Note the effect size is ~0.02; decide whether that is worth the hours
before spending them.

## Rules still in force

Everything under "Hard rules" above still applies — especially: the test split is read
once, the protocol stays fixed within a suite, failure classes stay in the mean, and no
number is reported that was not measured.

Use `mIoU_shared6` for every cross-split comparison. Keep the plain `mIoU` column beside
it so the two are never confused.
