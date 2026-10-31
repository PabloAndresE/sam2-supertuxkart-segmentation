# Handoff: decisive suite, seeds, ablations, LOTO and init seeds — done

Status as of 2026-10-31 (after Round 2), branch `results/decisive-suite`. This picks up from
[`AGENT_BRIEF.md`](AGENT_BRIEF.md); read that first for the rules. Everything the brief
asked for is done, including both optional suites, plus one extra suite for a paired
backbone comparison. The README Results section is the write-up; this file is what the
next session needs to know that the README does not say.

## What ran

| Suite | Runs | Completed | Notes |
|---|---|---|---|
| `configs/decisive.yaml` | 5 | 5 | ConvNeXt run trained twice; first checkpoint was incomplete (bug below) |
| `configs/seeds.yaml` | 3 | 3 | SAM 2 at seeds 1–3 (seed moved split and init together) |
| `configs/seeds_backbones.yaml` | 6 | 6 | ConvNeXt and U-Net at seeds 1–3, for paired differences |
| `configs/ablations.yaml` | 15 | 15 | base `configs/server.yaml`, not `default.yaml` |
| `configs/loto.yaml` (Round 2) | 6 | 6 | each track held out once for val |
| `configs/init_seeds.yaml` (Round 2) | 3 | 3 | split fixed at seed 42, `init_seed` 1–3 |

All 38 runs completed 60 epochs on `configs/server.yaml` (512px, batch 16, AMP). In
Round 2 the 29 earlier runs were re-evaluated on val (no retraining) to add
`mIoU_shared6`; their `mIoU`, per-class and boundary values reproduced exactly, and
`mIoU_shared6` matches the six-class mean previously computed by hand.
Checkpoints, `history.json` and figures are in `runs/` on the server, which is
gitignored; only `metrics_*.json`, `summary.md` and the test log are committed.

**The test split was read exactly once**: `runs/decisive/split_temporal_block`,
2026-10-30T23:31:42, mIoU 0.7731 (six classes, so also its `mIoU_shared6`). Logged in
that directory's `test_evaluations.jsonl`, which still has one line. No other run has a
test read; LOTO and init-seed runs evaluated val only. Do not add one.

## Headline findings

1. **Selection score overstated generalisation by 0.117**: chosen model, same split and
   six classes, val 0.8897 vs test 0.7731. Mostly rare classes (`Bomb` −0.36).
2. **Neighbour-leakage hypothesis rejected.** `temporal_block` 0.8897 > `random` 0.8630
   on `mIoU_shared6`. The 7-class gap was the `Projectile` term.
3. **Unseen track: 0.7108 ± 0.0911** `mIoU_shared6` over six held-out tracks, range
   0.5723 (`volcano_island`) to 0.8105 (`lighthouse`). The old "costs 0.20" was `abyss`
   alone; the mean cost vs 0.8897 is 0.18, from 0.08 to 0.32 by track.
4. **Noise ladder** (SAM 2, `mIoU_shared6` sd): GPU nondeterminism 0.0022 < init 0.0056
   < split 0.0305 < held-out track 0.0911.
5. **SAM 2 > ConvNeXt on 4/4 paired splits**, +0.0193 ± 0.0129, 95% CI
   [−0.0011, +0.0398]. Consistent, not yet significant.
6. **Ablations** (one split): all three FPN levels and the CE+Dice blend matter well
   beyond noise; class weights and oversampling are within noise.

## Things the next session must know

- **Bug fixed in `src/stkseg/train.py`** (commit "Save the full ConvNeXt baseline…"):
  checkpoints saved only `model.decoder` for any model with that attribute, dropping
  ConvNeXt's trainable `lateral` convs. Now `checkpoint_state()` saves decoder-only only
  for models with `load_decoder` (SAM 2). Regression test: `tests/test_checkpoint.py`.
  Any ConvNeXt checkpoint made before this commit is unusable.
- **Seeds:** `data.seed` draws the split; `data.init_seed` (new in 54e911c) sets the
  initialisation and falls back to `data.seed`. The `seeds` and `seeds_backbones` suites
  predate it, so their seeds moved split and init together.
- **`mIoU` vs `mIoU_shared6`:** plain `mIoU` counts any class in prediction OR target,
  so a predicted-only class adds 0.0 (SAM 2 seed 2: 0.8279 → 0.7097). Use
  `mIoU_shared6` for comparisons. **Open defect:** it still includes a class that is
  absent from the ground truth if the model predicts it. Only matters for single-track
  validation: `hacienda` has no `Bomb` in any frame, the model predicted some, and
  `loto_hacienda` reads 0.6517 instead of 0.7820 over its five real classes. The README
  reports the metric as defined with a footnote. A fix (return `None`, or average only
  classes present in the target) changes that one LOTO value; it was not applied.
- **Class presence per track** (frames of 250 containing the class): `hacienda` has 0
  `Bomb`; `volcano_island` 31 `Nitro` and 20 `Bomb`; `olivermath` 20 `Bomb`;
  `Projectile` appears once each in `hacienda` and `volcano_island`.
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

1. Decide the `mIoU_shared6` defect above, then re-evaluate only `runs/loto/*` (val).
2. More paired backbone splits (Round 2 brief, item 3, ~12 runs, ~1 h with `PAR=6`).
   The effect is ~0.02, about a fifth of the spread between held-out tracks; decide
   whether it is worth it. Put new entries in a **new** suite file: `run_parallel.sh`
   runs every entry of each suite it is given, so appending to an existing file would
   retrain and overwrite finished runs.
3. LOTO for ConvNeXt: does SAM 2's lead hold on unseen tracks, where the variance is?
4. Repeat the strongest ablation contrasts (FPN, loss blend) on a second split.

Commands:

```bash
export HF_HOME=$PWD/.hf_cache
GPUS="0 1" PAR=6 bash scripts/run_parallel.sh configs/<new_suite>.yaml
```
