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

> **Status: the headline number predates this protocol.** mIoU **0.8534** was obtained
> with a `random` split, measured on the same validation set used to select the
> checkpoint, from a single seed. It is reported here as the *legacy* figure and is
> expected to drop under `temporal_block`, and to drop further under `track_holdout`.
> Re-running the suite is the next step; the table below will be replaced with
> mean ± std over three seeds for each protocol.

Legacy per-class IoU (random split, validation, seed 42):

| Class | IoU | Pixel share |
|---|---|---|
| Background | 0.9102 | ~55% |
| Track | 0.9044 | ~38% |
| Kart | **0.9156** | ~5% |
| Pickup | 0.8594 | <1% |
| Nitro | 0.8037 | <0.5% |
| Bomb | 0.7269 | <0.5% |
| Projectile | **0.0000** | 2 frames out of 1500 |
| **mean** | **0.8534** | |

Two things worth reading honestly:

- **Kart scores highest (0.9156) despite being a minority class.** It is a large object
  with a crisp silhouette — exactly the prior SAM 2 was pre-trained on.
- **Projectile is 0.0000 and no amount of rebalancing fixes it.** It appears in 2 of
  1500 frames. Oversampling, a 15× loss weight and Dice all failed, because the problem
  is the absence of data, not the objective. It stays in the mean rather than being
  quietly dropped.

![Predictions versus ground truth](docs/assets/predictions.png)

![Training curves](docs/assets/training_curves.png)

The mIoU plateau between epochs 14–46 is a learning-rate artifact: at `5e-4` the
optimizer oscillates around the minimum. `ReduceLROnPlateau` cut it to `2.5e-4` at
epoch 47 and `1.25e-4` at epoch 55, which is where the final 0.80 → 0.8534 gain comes
from. A cosine schedule would likely get there sooner.

### Metrics

Besides IoU, evaluation reports **boundary IoU** (IoU restricted to a 2-pixel band
around class boundaries) and a row-normalised **confusion matrix**. Boundary IoU exists
because plain IoU is dominated by region interiors: a model can score 0.43 on a class
while producing no boundaries at all. Since the premise of this project is that SAM 2
contributes boundary quality, measuring it directly is the relevant check.

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

- **The headline number predates the current protocol** and will change. See Results.
- Batch size is pinned at 4 by encoder activation memory at 512×512. Gradient
  accumulation would allow a larger effective batch.
- Only Hiera-Tiny was tried. Hiera-Small/Base would separate encoder capacity from
  decoder design.
- Single seed so far; the seed suite exists but has not been run on the real dataset.
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
