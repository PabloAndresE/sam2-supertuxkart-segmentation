# Model card: SuperTuxKart segmentation decoder

A 2.07M-parameter segmentation decoder trained on top of a **frozen** SAM 2 Hiera-Tiny
vision encoder. Only the decoder is distributed (7.9 MB); the encoder is fetched from
Hugging Face at load time.

| | |
|---|---|
| Task | Multiclass semantic segmentation, 7 classes |
| Input | RGB frame, resized to 512×512 (384px also validated) |
| Output | Per-pixel class map at input resolution |
| Trainable parameters | 2,067,591 of 29,286,727 (7.06%) |
| Encoder | `facebook/sam2.1-hiera-tiny`, frozen, kept in `eval()` |
| Artifact | `decoder_best.pt`, 7.9 MB, release `v0.1.0` |
| Licence | MIT (code and weights) |

## Intended use

Research and teaching material for frozen-backbone transfer learning and for evaluation
protocol design. It segments frames from the SuperTuxKart racing game into Background,
Track, Kart, Pickup, Nitro, Bomb and Projectile.

**Not intended for** anything safety-relevant, for real-world driving or road scenes, or
for SuperTuxKart footage captured outside the `pystk` renderer. The training frames come
from one renderer at one resolution with fixed graphics settings; gameplay captured from
the retail game is a different domain and has not been evaluated.

## Training data

1500 frame/mask pairs exported from [`pystk`](https://github.com/philkr/pystk), the
Python binding to the SuperTuxKart engine, which yields pixel-exact ground truth. Six
tracks × 250 **consecutive** frames: `abyss`, `gran_paradiso_island`, `hacienda`,
`lighthouse`, `olivermath`, `volcano_island`.

The class distribution is severely imbalanced: Background and Track cover over 90% of
pixels, while Nitro, Bomb and Projectile are each under 0.5%. `Projectile` occurs in
**2 frames out of 1500**. `hacienda` contains no `Bomb` at all.

Because the frames are consecutive video, they are not independent samples. See
Evaluation.

## Evaluation

Reported on the `temporal_block` split (contiguous blocks of frames assigned whole, so
validation frames do not sit adjacent to training frames), seed 42.

| Class | Validation IoU | Test IoU |
|---|---|---|
| Background | 0.9132 | 0.9007 |
| Track | 0.8974 | 0.8894 |
| Kart | 0.9241 | 0.9339 |
| Pickup | 0.8631 | 0.7641 |
| Nitro | 0.8485 | 0.7033 |
| Bomb | 0.7453 | 0.4471 |
| Projectile | 0.0000 | not in split |
| **mean (shared six classes)** | **0.8897** | **0.7731** |

The test split was read **once**, after model selection, and the read is logged in
`runs/decisive/split_temporal_block/test_evaluations.jsonl`.

### Numbers that qualify the ones above

- **The validation score overstates generalisation by 0.117.** The checkpoint was
  selected on validation; test is 0.7731 against 0.8897 on the same six classes.
- **Unseen scenery costs 0.157 ± 0.090.** Leave-one-track-out over all six tracks ranges
  from 0.8105 (`lighthouse`) to 0.5723 (`volcano_island`).
- **Which track is held out dominates every other factor measured**: sd 0.090 against
  0.031 for which frames land in validation and 0.0056 for weight initialisation.
- **Boundary IoU is far below region IoU** (0.32 validation, 0.30 test): edges are
  substantially worse than the region scores suggest.

## Known failure modes

- `Projectile` is never predicted correctly. Two training frames is not enough, and
  oversampling, a 15× loss weight and Dice loss all failed to change it.
- `Bomb` degrades sharply out of distribution (0.75 validation → 0.45 test → 0.00 on
  `hacienda`, which contains none and where the model hallucinates them).
- Large-class accuracy collapses on unfamiliar scenery: on `abyss` held out, `Track`
  falls to 0.53 and `Background` to 0.67.
- The model has only been trained and evaluated at 512×512 with ImageNet normalisation.

## Inference cost

Apple M1 Pro, full encoder + decoder + upsample, after warm-up:

| Resolution | Latency | FPS |
|---|---|---|
| 512px | 87 ms | 11.4 |
| 384px | 51 ms | 19.7 |
| 256px | 29 ms | 34.5 |

fp16 gains about 10% on MPS; the workload is bandwidth- and launch-overhead-bound.

## Licensing note

Code and decoder weights are MIT. The training frames are rendered from SuperTuxKart,
which is GPL-licensed with CC-BY-SA assets; redistributing rendered frames carries that
inheritance. The dataset archive is distributed separately from the code for that reason.

## Reproducing

```bash
bash scripts/fetch_assets.sh
python -m stkseg.train    --config configs/server.yaml
python -m stkseg.evaluate --split val
```

Full protocol and ablations: [`README.md`](README.md). Server instructions:
[`docs/RUNNING_ON_A_SERVER.md`](docs/RUNNING_ON_A_SERVER.md).
