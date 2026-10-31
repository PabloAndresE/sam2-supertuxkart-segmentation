# Handoff: decisive suite, seeds and ablations — done

Status as of 2026-10-31, branch `results/decisive-suite`. This picks up from
[`AGENT_BRIEF.md`](AGENT_BRIEF.md); read that first for the rules. Everything the brief
asked for is done, including both optional suites, plus one extra suite for a paired
backbone comparison. The README Results section is the write-up; this file is what the
next session needs to know that the README does not say.

## What ran

| Suite | Runs | Completed | Notes |
|---|---|---|---|
| `configs/decisive.yaml` | 5 | 5 | ConvNeXt run trained twice; first checkpoint was incomplete (bug below) |
| `configs/seeds.yaml` | 3 | 3 | SAM 2 at seeds 1–3 |
| `configs/seeds_backbones.yaml` (new) | 6 | 6 | ConvNeXt and U-Net at seeds 1–3, for paired differences |
| `configs/ablations.yaml` | 15 | 15 | base `configs/server.yaml`, not `default.yaml` |

All 29 runs completed 60 epochs. All used `configs/server.yaml` (512px, batch 16, AMP).
Checkpoints, `history.json` and figures are in `runs/` on the server, which is
gitignored; only `metrics_*.json`, `summary.md` and the test log are committed.

**The test split was read exactly once**: `runs/decisive/split_temporal_block`,
2026-10-30T23:31:42, mIoU 0.7731. Logged in that directory's `test_evaluations.jsonl`.
No other run has a test read. Do not add one.

## Headline findings

1. **No measurable inflation from frame leakage.** `random` vs `temporal_block` differs
   by 0.10 on the 7-class mean, but all of it is `Projectile` (0.0) appearing only in the
   `temporal_block` val set. On the six shared classes `temporal_block` is higher.
2. **Unseen tracks are the real drop**: `track_holdout` 0.6910 (one track, `abyss`).
3. **The split dominates variance**: same split rerun sd 0.0022; across four splits sd
   0.0305.
4. **SAM 2 > ConvNeXt on 4/4 paired splits**, +0.0193 ± 0.0129, 95% CI
   [−0.0011, +0.0398]. Consistent, not yet significant.
5. **Ablations** (one split): all three FPN levels and the CE+Dice blend matter well
   beyond noise; class weights do not; oversampling helps slightly (`Bomb`).

## Things the next session must know

- **Bug fixed in `src/stkseg/train.py`** (commit "Save the full ConvNeXt baseline…"):
  checkpoints saved only `model.decoder` for any model with that attribute, dropping
  ConvNeXt's trainable `lateral` convs. Now `checkpoint_state()` saves decoder-only only
  for models with `load_decoder` (SAM 2). Regression test: `tests/test_checkpoint.py`.
  Any ConvNeXt checkpoint made before this commit is unusable.
- **`data.seed` drives both the split and `torch.manual_seed`.** "Seed" runs are
  different splits, not just different initialisations. Projectile is in validation
  only at seed 42. Separating the two needs a `train.seed` field — not done, because it
  would change what existing seed numbers mean.
- **`mIoU` counts classes present in prediction OR target.** A predicted-only class adds
  a 0.0 (happened: SAM 2 seed 2, 0.8279 → 0.7097). All cross-run comparisons in the
  README use the six-class mean computed from `iou_per_class` for this reason. Changing
  the metric itself would change every committed number; if you do it, rerun evaluation
  on all runs and say so.
- **`summary.md` files**: the `val_iou` column is the training loop's batch-averaged
  selection score, and the "mean over N runs" line averages across protocols. Do not
  quote either.
- **SLURM was never used.** This server has no scheduler, so `scripts/slurm/*.sbatch`
  remain untested on a real cluster. Runs went through `scripts/run_suite.sh` and the new
  `scripts/run_parallel.sh`.

## This server (if you resume here)

- Repo at `/root/work/sam2-supertuxkart-segmentation`, venv `.venv` (made with
  `virtualenv`; system `python3 -m venv` lacks `ensurepip`), torch 2.6.0+cu124,
  weights cached in `.hf_cache/` and `~/.cache/torch`.
- `gh` is at `/root/.local/bin/gh`, logged in.
- 8× H200 **shared with other users**; GPUs 2–7 were busy throughout. All runs used
  GPUs 0–1. The host CPU saturates easily: keep `OMP_NUM_THREADS=4`
  (`run_parallel.sh` defaults to it).
- An empty, broken `/root/.venv` was created by mistake and could not be removed from
  the session; it is unused and safe to delete.

## Suggested next steps, cheapest first

1. More splits for the backbone comparison: seeds 4–7 for SAM 2, ConvNeXt and U-Net
   (~12 runs, ~1 h on two GPUs with `PAR=6`). Put them in a **new** suite file:
   `run_parallel.sh` runs every entry of each suite it is given, so appending to
   `seeds.yaml` would retrain and overwrite seeds 1–3. This is what turns finding 4 into
   a result or kills it.
2. Leave-one-track-out for `track_holdout` (six runs) to replace the single-track
   estimate in finding 2.
3. A separate training seed (`train.seed`) to split init noise from split noise.
4. Repeat the strongest ablation contrasts (FPN, loss blend) on a second split.

Commands:

```bash
export HF_HOME=$PWD/.hf_cache
GPUS="0 1" PAR=6 bash scripts/run_parallel.sh configs/seeds_more.yaml   # a new file
```
