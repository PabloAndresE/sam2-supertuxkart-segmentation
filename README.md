# SuperTuxKart Semantic Segmentation with a Frozen SAM 2 Encoder

Pixel-level segmentation of 7 game classes by freezing Meta's SAM 2 Hiera-Tiny vision
encoder and training only a 2.1M-parameter multi-scale decoder on top of it —
**7% of the network is trainable**, and the trained artifact is an **8 MB** decoder
rather than a 117 MB checkpoint.

![Single-frame inference: input, segmentation, overlay](docs/assets/example_prediction.png)

<sub>Produced by `python -m stkseg.predict docs/assets/examples/abyss_frame_0100.png`.</sub>

---

## The question

The usual transfer-learning recipe fine-tunes a backbone. Here the backbone never
receives a gradient: SAM 2 already knows what an object boundary looks like, so the
only thing worth learning is how to map its FPN features onto a 7-class taxonomy it has
never seen. That isolates a clean question — **how far do frozen foundation-model
features get you, and does it have to be SAM 2?**

The repository is built to answer that with controlled comparisons rather than a single
number, which is why the evaluation protocol below gets more attention than the result.

## Evaluation protocol (read this before the numbers)

The dataset is **not 1500 independent images**. It is 6 tracks × 250 *consecutive frames*
from a race. Measured on `abyss`, a frame differs from its neighbour by a mean absolute
difference of **0.066**, versus **0.126** for frames more than 25 apart — neighbours are
about twice as similar as distant frames. They are not duplicates, but they are
correlated, and a uniformly random split puts that correlation on both sides of the
boundary.

Three split strategies are implemented in [`src/stkseg/splits.py`](src/stkseg/splits.py):

| Strategy | Adjacent frame pairs crossing a split | Tracks in val/test | Measures |
|---|---|---|---|
| `random` | **691** | all 6 (also in train) | optimistic; reproduces the original coursework number |
| `temporal_block` | **26** (block boundaries only) | 5 / 4 | generalisation to unseen moments of a known track |
| `track_holdout` | **0** | 1 / 1, disjoint from train | generalisation to **unseen scenery** |

`temporal_block` is the default. `track_holdout` is the honest hard setting: a random
split cannot answer the question "does this work on a track it has never seen" at all,
because every track appears in training.

Two further guarantees:

- **Three-way 70/15/15 split.** Model selection reads `val`. The original setup selected
  the best checkpoint by validation mIoU and then reported that same validation mIoU —
  a selection score, not a generalisation estimate.
- **The test split is gated.** `stkseg.evaluate --split test` refuses to run without
  `--confirm-final`, and every test evaluation is appended to
  `test_evaluations.jsonl` with its timestamp, strategy and seed. The gate is not
  tamper-proof; it exists to make a test read deliberate and auditable.

## Results

Every run here shares one protocol (`configs/server.yaml`: 512×512, 60 epochs, batch 16,
AMP, Adam 5e-4 with `ReduceLROnPlateau`) and was trained on NVIDIA H200s. Validation
scores are from `stkseg.evaluate` (dataset-level IoU), not from the training loop. The
headline tables come from `configs/decisive.yaml` at seed 42; the variance estimate uses
`configs/seeds.yaml` and `configs/seeds_backbones.yaml` (seeds 1–3), and the ablation
table `configs/ablations.yaml` (seed 42). 29 training runs in total, all completed (plus
one discarded ConvNeXt run whose checkpoint was incomplete; see the git log).

**Two sources of noise, measured separately:**

- *Same split, same seed, rerun:* five runs of the chosen configuration (four of them
  duplicates inside the ablation suite) score 0.8881 ± 0.0022 (six-class mIoU, mean ± sd).
  Initialisation and GPU nondeterminism are small.
- *Different split:* `data.seed` also draws the `temporal_block` split, so seeds 1–3 put
  different frames in validation (0–75 of 225 shared with seed 42). Across those four
  splits the same model scores 0.8521 ± 0.0305 — fourteen times the rerun noise. **Which
  frames land in validation matters far more than anything else measured here.**

### How much did the split protocol inflate 0.8534?

| Split | Val tracks | Classes in val | Val mIoU | Val mIoU, 6 shared classes | Boundary mIoU |
|---|---|---|---|---|---|
| `random` | all 6 | 6 | 0.8630 | 0.8630 | 0.4262 |
| `temporal_block` | 5 | 7 | **0.7626** | 0.8897 | 0.3232 |
| `track_holdout` | 1 (`abyss`) | 6 | **0.6910** | 0.6910 | 0.1804 |

The legacy figure, from a different run under the old protocol, is kept apart:

| Legacy (coursework protocol, not comparable to the table above) | mIoU |
|---|---|
| `random` split, checkpoint selected and scored on the same validation set | 0.8534 |

Read the "6 shared classes" column before drawing a conclusion. `mIoU` averages over the
classes present in that validation set, and the three splits do not contain the same
classes: `Projectile` (2 frames in the whole dataset) lands in the `temporal_block`
validation set, where it scores 0.0000 and pulls the 7-class mean down, but is absent
from the `random` and `track_holdout` validation sets. Restricting every split to the six
classes they all contain separates the two effects:

- **Leakage between neighbouring frames did not measurably inflate the score.** On the
  same six classes, `temporal_block` (0.8897) scored *higher* than `random` (0.8630).
  The 0.10 gap in the headline column is entirely the Projectile term
  (0.8897 × 6/7 = 0.7626), not leakage. The
  validation sets differ in content as well as size, and the split comparison was only
  run at seed 42 (a rerun reproduced it: 0.8699 `random`, 0.8898 `temporal_block`), so
  the honest reading is "no inflation detected", not "temporal_block is easier".
- **Unseen scenery is the real drop.** Held out on a track it never trained on, the
  model scores 0.6910, 0.16 below the legacy 0.8534 and 0.20 below `temporal_block` on
  the same classes. The losses are in the large classes (`Background` 0.6654,
  `Track` 0.5274), i.e. the scenery itself, and boundary mIoU falls to 0.1804. Its best
  validation epoch was 20 of 60: after that, fitting the four training tracks hurt the
  fifth. A rerun gave 0.6791. That is one held-out track (`abyss`, 250 frames), so it is
  a single draw of "unseen scenery", not an average over tracks.

![track_holdout: predictions on the unseen track](docs/assets/decisive/track_holdout_predictions_val.png)

### Is a frozen backbone worth it, and does it have to be SAM 2?

Same split (`temporal_block`), same validation set, same schedule:

| Backbone | Frozen | Trainable | Val mIoU | Val mIoU, 6 classes | Boundary mIoU |
|---|---|---|---|---|---|
| SAM 2 Hiera-Tiny (frozen) | 27.2M | 2.07M | **0.7626** | **0.8897** | **0.3232** |
| ConvNeXt-Tiny ImageNet (frozen) | 27.8M | 2.41M | 0.7429 | 0.8652 | 0.2714 |
| U-Net from scratch | — | 7.77M | 0.7251 | 0.8460 | 0.3154 |

On this one split SAM 2 leads ConvNeXt by 0.025 (six classes), more than ten times the
rerun noise. But the split itself is the larger source of variance, so the comparison
that answers the question is **paired across splits**: every backbone trained on the
same four `temporal_block` splits (seeds 42, 1, 2, 3), six-class validation mIoU:

| Seed (split) | SAM 2 | ConvNeXt frozen | U-Net scratch | SAM 2 − ConvNeXt | SAM 2 − U-Net |
|---|---|---|---|---|---|
| 42 | 0.8897 | 0.8652 | 0.8460 | +0.0245 | +0.0437 |
| 1 | 0.8643 | 0.8573 | 0.8659 | +0.0069 | −0.0016 |
| 2 | 0.8279 | 0.8170 | 0.7240 | +0.0109 | +0.1039 |
| 3 | 0.8266 | 0.7916 | 0.7438 | +0.0350 | +0.0828 |
| **mean ± sd** | 0.8521 ± 0.0305 | 0.8328 ± 0.0346 | 0.7949 ± 0.0714 | **+0.0193 ± 0.0129** | +0.0572 ± 0.0465 |
| 95% CI of the paired difference (t, 3 df) | | | | [−0.0011, +0.0398] | [−0.0168, +0.1312] |

Boundary mIoU (six classes), same pairing: SAM 2 − ConvNeXt +0.0391 ± 0.0289,
95% CI [−0.0068, +0.0850]; SAM 2 − U-Net +0.0154 ± 0.0228, CI [−0.0209, +0.0516].

What this supports:

- **SAM 2 beat ConvNeXt on all four splits**, by +0.019 on average at near-identical
  parameter budgets, and on boundary IoU in three of four. With four pairs the 95%
  interval just touches zero (a 4/4 sign test gives p = 0.125 two-sided). The direction
  is consistent; its size is not pinned down. More splits would settle it cheaply.
- **SAM 2 versus a U-Net trained from scratch is not established.** U-Net is far less
  stable across splits (sd 0.071) and matched SAM 2 on seed 1.
- The boundary claim — that SAM 2's value is edge quality — points the right way against
  ConvNeXt but is not significant at four splits, and U-Net's boundary IoU is close to
  SAM 2's.

### Chosen model: SAM 2, `temporal_block`

Selected on validation mIoU among the three runs that share a validation set (seed 42).
That split turned out to be the most favourable of the four measured: its six-class
validation mIoU (0.8897) is the highest, against a four-split mean of 0.8521 ± 0.0305.
Per-class results:

| Class | Val IoU | Val boundary IoU | Test IoU | Test boundary IoU |
|---|---|---|---|---|
| Background | 0.9367 | 0.2041 | 0.9007 | 0.1969 |
| Track | 0.9236 | 0.2624 | 0.8894 | 0.2809 |
| Kart | 0.9243 | 0.5096 | 0.9339 | 0.5288 |
| Pickup | 0.8964 | 0.5203 | 0.7641 | 0.4479 |
| Nitro | 0.8541 | 0.3881 | 0.7033 | 0.2255 |
| Bomb | 0.8031 | 0.3781 | 0.4471 | 0.1226 |
| Projectile | 0.0000 | 0.0000 | *not in test split* | — |
| **mean** | **0.7626** | **0.3232** | **0.7731** | **0.3004** |

**Test mIoU 0.7731** (boundary 0.3004) is a one-shot final measurement: the test split
was read once, on this model only, after selection, and that read is logged in
`runs/decisive/split_temporal_block/test_evaluations.jsonl`. It averages six classes
because `Projectile` does not occur in the test split; it is not comparable digit-for-digit
with the seven-class validation mean. `Bomb` drops from 0.80 on validation to 0.45 on
test, the largest per-class gap.

`Projectile` stays in the validation mean at 0.0000. It appears in 2 of 1500 frames;
oversampling, a 15× loss weight and Dice did not fix it, because the problem is missing
data, not the objective. Dropping it would raise the validation mean to 0.8897, which is
why the six-class column is labelled separately rather than used as the headline.

![Predictions versus ground truth, validation](docs/assets/decisive/sam2_predictions_val.png)

![Confusion matrix, validation](docs/assets/decisive/sam2_confusion_val.png)

![Confusion matrix, test](docs/assets/decisive/sam2_confusion_test.png)

![Training curves](docs/assets/decisive/sam2_training_curves.png)

### Cost

Wall-clock per run on one H200 (shared with other jobs), training plus checkpointing:
`split_random` 7.8 min, `split_temporal_block` 10.5 min, `split_track_holdout` 8.7 min,
`backbone_convnext_frozen` 7.2 min, `backbone_unet_scratch` 13.3 min. Preflight projected
~3 s/epoch (~3 min per run); measured runs took 2.5–4× that. The seed and ablation runs
ran six at a time across two H200s on a CPU-saturated shared host, at 10–42 min each;
those times measure contention, not the model.

The aggregate table and the raw metrics are in
[`runs/decisive/summary.md`](runs/decisive/summary.md) and
`runs/decisive/*/metrics_val.json`. Its `val_iou` column is the training loop's
batch-averaged selection score and its final "mean over 5 runs" line averages across
different protocols; neither should be quoted as a result.

### Metrics

Besides IoU, evaluation reports **boundary IoU** (IoU restricted to a 2-pixel band
around class boundaries) and a row-normalised **confusion matrix**. Boundary IoU exists
because plain IoU is dominated by region interiors: a model can score 0.43 on a class
while producing no boundaries at all. Since the premise of this project is that SAM 2
contributes boundary quality, measuring it directly is the relevant check.

**A caveat on `mIoU` as implemented.** It averages over every class whose union of
prediction and ground truth is non-empty. A class absent from the ground truth but
*predicted* by the model therefore enters the mean as 0.0. This happened on seed 2:
`Projectile` is not in that validation set, SAM 2 predicted a few pixels of it, and its
mIoU dropped from 0.8279 (six classes) to 0.7097. The six-class columns above are
immune to this and are the ones used for every comparison across runs.

## Baselines and ablations

Every entry runs through the same trainer, splits and schedule, so the comparisons are
controlled. Defined in [`configs/ablations.yaml`](configs/ablations.yaml):

**Backbone** — is a frozen backbone worth it, and does it have to be SAM 2?

| Model | Frozen | Trainable | Trainable % |
|---|---|---|---|
| `sam2` (Hiera-Tiny) | 27.2M | 2.07M | 7.1% |
| `convnext_frozen` (ImageNet ConvNeXt-Tiny) | 27.8M | 2.41M | 8.0% |
| `unet` (from scratch) | — | 7.8M | 100% |

The ConvNeXt baseline is the one that matters: nearly the same frozen and trainable
budget, so any gap is attributable to *what* the backbone learned, not to capacity.
Results across four splits are in Results above.

**Ablation results** — SAM 2, `temporal_block`, seed 42, one axis changed at a time.
Six-class validation mIoU; the reference configuration scores 0.8881 ± 0.0022 over five
reruns, so differences under ~0.005 are noise. All rows share one split, so they show
what matters *on that split*; they have not been repeated across splits.

| Change from reference | mIoU (6 cls) | Δ vs 0.8881 | Boundary (6 cls) | Pickup | Nitro | Bomb |
|---|---|---|---|---|---|---|
| none (reference: all FPN levels, CE 0.3 + Dice 0.7, class weights, oversampling ×5) | 0.8881 | — | 0.3755 | 0.89 | 0.85 | 0.80 |
| FPN `high` + `mid` only | 0.8512 | −0.037 | 0.3188 | 0.83 | 0.84 | 0.73 |
| FPN `low` only | 0.8050 | −0.083 | 0.1729 | 0.85 | 0.65 | 0.63 |
| FPN `high` only | 0.7661 | −0.122 | 0.2740 | 0.60 | 0.82 | 0.55 |
| Dice only | 0.8544 | −0.034 | 0.3342 | 0.87 | 0.85 | 0.80 |
| CE only | 0.8168 | −0.071 | 0.2601 | 0.82 | 0.75 | 0.56 |
| no class weights (blend kept) | 0.8933 | +0.005 | 0.3916 | 0.90 | 0.86 | 0.80 |
| no oversampling | 0.8788 | −0.009 | 0.3614 | 0.89 | 0.84 | 0.76 |

- **Multi-scale fusion earns its keep.** Every subset of FPN levels loses 0.04–0.12;
  `low` alone keeps region IoU but halves boundary IoU, `high` alone loses small objects.
- **The CE + Dice blend matters; class weights do not.** CE alone collapses on `Bomb`
  (0.56); Dice alone loses the large classes (`Background` 0.85, `Track` 0.83). Removing
  the class weights changed nothing beyond noise (+0.005, at the edge of the rerun range).
- **Oversampling helps a little**, mostly on `Bomb` (0.76 without, ~0.80 with).
- **Nothing fixes `Projectile`.** Its best score in any ablation is 0.031 (CE only).

Per-run numbers: `runs/ablations/summary.md`, `runs/seeds/summary.md`,
`runs/seeds_backbones/summary.md` and each run's `metrics_val.json`.

**Other axes:** FPN level subsets (`high` / `low` / `high+mid` / all three), loss
composition (CE only, Dice only, the 30/70 blend, unweighted CE), oversampling on/off,
and the three split strategies.

```bash
make ablations          # full suite, writes runs/ablations/summary.md
make seeds              # 3 seeds of the headline config, reports mean +/- std
python scripts/run_experiments.py --suite configs/ablations.yaml --dry-run
```

## Performance

Measured on an Apple M1 Pro (16 GB), full encoder + decoder + upsample path, after
warm-up (`make bench`):

| Resolution | Latency (median) | FPS | fp16 |
|---|---|---|---|
| 512px (training resolution) | 87.4 ms | 11.4 | 79.4 ms → 12.6 |
| 384px | 50.8 ms | 19.7 | 46.4 ms → 21.5 |
| 256px | 29.0 ms | 34.5 | — |

fp16 buys only ~10% on MPS: at this size the workload is bandwidth- and
launch-overhead-bound, not compute-bound.

## Handling a 200:1 class imbalance

Background and Track cover >90% of pixels. Without compensation the model collapses to
predicting those two. Three mechanisms stack, and the ablation suite measures each:

1. **Oversampling** — `WeightedRandomSampler` gives frames containing Nitro/Bomb/Projectile
   5× selection probability.
2. **Weighted cross-entropy** — inverse-frequency weights from 0.5 (Background) to 15.0
   (Projectile).
3. **Dice loss at 70%** — computed per class over the batch, so a class covering 0.1% of
   pixels contributes as much to the gradient as the background does.

## Architecture

```
input 512×512×3
        │
        ▼
┌───────────────────────────────┐
│ SAM 2 Hiera-Tiny vision enc.  │  27.2M params — FROZEN, always in eval()
└───────────────────────────────┘
        │  FPN hidden states (256 ch each)
        ├── 128×128  "high" ──┐
        ├──  64×64   "mid"  ──┤  bilinear resample to the highest selected level
        └──  32×32   "low"  ──┘
                              ▼
            concat → fuse (3×3, BN, ReLU, Dropout 0.1) → 128 ch
                              ▼
                 Conv 1×1 → 7 logits
                              ▼
            bilinear upsample → 512×512×7     2.1M params — TRAINABLE
```

Design notes:

- The encoder stays in `eval()` even during `model.train()` — otherwise its BatchNorm
  running statistics drift and the "frozen" claim quietly becomes false.
- All selected FPN levels are projected to the *highest* available resolution rather
  than progressively upsampled: small classes live in a handful of pixels and lose too
  much in a cascade. Whether this is worth it is what the FPN ablation tests.
- Masks are always resized with `NEAREST`. Bilinear interpolation on a label map invents
  class ids that do not exist.

## Quickstart

```bash
git clone https://github.com/PabloAndresE/sam2-supertuxkart-segmentation
cd sam2-supertuxkart-segmentation
pip install -e ".[app,dev]"
bash scripts/fetch_assets.sh     # dataset (302 MB) + trained decoder (7.9 MB)
```

```bash
make app                       # Gradio demo at http://localhost:7860
docker compose up --build      # same thing, containerised

python -m stkseg.train    --config configs/default.yaml --data-root seg_dataset
python -m stkseg.evaluate --split val
python -m stkseg.evaluate --split test --confirm-final      # once, at the end
python -m stkseg.predict  path/to/frame.png --out prediction.png
```

Useful overrides: `--model {sam2,convnext_frozen,unet}`,
`--strategy {random,temporal_block,track_holdout}`, `--seed N`, `--amp`,
`--batch-size N`, `--num-workers N`.

## Running the suite on a GPU server

A training epoch costs **245 s** on an Apple M1 Pro (512px, batch 4) — about 4 h per run
and 24 h for the suite, so the experiments belong on a GPU. Full instructions in
[`docs/RUNNING_ON_A_SERVER.md`](docs/RUNNING_ON_A_SERVER.md):

```bash
CUDA_TAG=cu124 bash scripts/setup_server.sh      # venv, CUDA torch, weights pre-cached
export HF_HOME=$PWD/.hf_cache
python scripts/preflight.py --config configs/server.yaml --amp

bash scripts/run_suite.sh                        # single box, sequential
sbatch scripts/slurm/suite.sbatch                # SLURM, one array task per experiment
```

`scripts/preflight.py` is the gate: it checks the dataset, builds the splits, confirms
the weights load **offline** (compute nodes are often firewalled) and times real
dataloader batches to project runtime and peak memory, so a `batch_size` that OOMs is
caught in seconds rather than after the job starts.

## Dataset and weights

Neither is tracked in git; both are GitHub release assets, fetched by
`bash scripts/fetch_assets.sh`:

| Asset | Release | Size |
|---|---|---|
| `Archive.zip` → `seg_dataset/` | `data-v1` | 302 MB |
| `decoder_best.pt` → `runs/baseline/` | `v0.1.0` | 7.9 MB |

The dataset is 1500 frame/mask pairs (400×400 RGB) exported from
[`pystk`](https://github.com/philkr/pystk), the Python binding to the SuperTuxKart
engine, which gives exact pixel-level ground truth:

```
seg_dataset/
  images/abyss_frame_0128.png
  masks/abyss_mask_combined_0128.png   # uint8, values 0–6
```

The decoder checkpoint is 7.9 MB rather than 117 MB because the frozen encoder is not
stored — it is pulled from Hugging Face at load time. To regenerate it from a full
checkpoint that still contains the encoder:

```bash
python scripts/extract_decoder.py model.th runs/baseline/decoder_best.pt   # 117 MB -> 8 MB
```

## Layout

```
src/stkseg/
  constants.py   class names, palette, device selection
  config.py      dataclass config, YAML loading, per-axis overrides
  splits.py      random / temporal_block / track_holdout strategies
  data.py        dataset, augmentation, oversampled dataloaders
  model.py       SegmentationDecoder (level-selectable) + SAM2Segmenter
  baselines.py   UNet from scratch, frozen ConvNeXt-Tiny
  registry.py    build_model(cfg) — one entry point for every runner
  losses.py      Dice and the CE/Dice blend
  metrics.py     IoU, boundary IoU, confusion matrix
  viz.py         colourised masks, overlays, curve plots
  train.py       training CLI (selects on val, never reads test)
  evaluate.py    evaluation CLI, gated test split, audit log
  predict.py     single-image CLI
app/app.py       Gradio demo
configs/         default config, ablation suite, seed suite
scripts/         decoder extraction, benchmark, experiment runner
tests/           40 tests: splits, data, losses, metrics, decoder, registry
notebooks/       the original coursework notebook, unmodified
docs/assets/     figures and the full Spanish report (PDF)
```

## Limitations

- **Four splits is not many.** The SAM 2 − ConvNeXt paired difference (+0.019) is
  positive on all four but its 95% interval still touches zero. Adding seeds to
  `configs/seeds.yaml` and `configs/seeds_backbones.yaml` is the cheapest way to settle it.
- **`data.seed` controls both the split and the initialisation**, so the seed suites
  measure the two together. The split dominates (sd 0.031 vs 0.002 for reruns on a fixed
  split), but separating them properly needs a training seed independent of the split
  seed.
- **The chosen model's split was the most favourable of the four on validation** (0.8897
  vs a four-split mean of 0.8521). The model was chosen and the test split read once
  before the other splits were run. Whether seed 42's *test* split is similarly easy was
  not measured — doing so would mean reading other test splits, which the protocol forbids.
- **Ablations are on one split.** Their ranking is far outside rerun noise, but they
  were not repeated across splits.
- **`mIoU` counts predicted-only classes as 0.0** (see Metrics); the six-class columns
  are the like-for-like comparison.
- **Validation sets differ in class content across split strategies** (`Projectile` is
  present only in the `temporal_block` validation set), so the headline mIoU column is
  not a like-for-like comparison of splits; the six-class column is the control.
- **`track_holdout` is one held-out track** (`abyss` for validation, `volcano_island`
  for test). A leave-one-track-out loop over all six would give a mean and spread for
  unseen scenery.
- The test split of `temporal_block` contains no `Projectile`, so the test mean is over
  six classes and the validation mean over seven.
- `configs/default.yaml` (laptop) pins batch size at 4; the results above used
  `configs/server.yaml` at batch 16 (2.4 GB peak in preflight on an H200).
- Only Hiera-Tiny was tried. Hiera-Small/Base would separate encoder capacity from
  decoder design.
- `Projectile` is unlearnable from 2 frames. Copy-paste augmentation is the obvious
  next attempt.
- Frames come from video but are treated as independent images. SAM 2's memory
  attention — the part that makes it "2" — is unused.

## Reference

Built for the Computer Vision course of the MSc in Artificial Intelligence,
Universidad San Francisco de Quito. The original methodology write-up (Spanish) is in
[`docs/assets/report_es.pdf`](docs/assets/report_es.pdf); it describes the coursework
protocol, not the corrected one above.

- Ravi et al., *SAM 2: Segment Anything in Images and Videos* (2024)
- Lin et al., *Feature Pyramid Networks for Object Detection* (2017)
- Milletari et al., *V-Net* (2016) — Dice loss
- Cheng et al., *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation* (2021)

MIT licensed.
