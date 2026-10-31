"""Evaluate a checkpoint on val or test.

The test split is gated behind ``--confirm-final`` and writes a stamped record of
every evaluation it performs. The point is not to make cheating impossible, but
to make it deliberate and visible: a test score is only trustworthy if it was
produced once, after the model was frozen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from stkseg.config import Config
from stkseg.constants import CLASS_NAMES, DEVICE
from stkseg.data import build_dataloaders
from stkseg.metrics import (
    SHARED_CLASSES,
    boundary_iou,
    confusion_matrix,
    iou_per_class,
    mean_iou_over,
)
from stkseg.registry import build_model
from stkseg.viz import add_class_legend, denormalize, mask_to_color, overlay

TEST_LOG = "test_evaluations.jsonl"


def collect(model, loader, device):
    preds, targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            preds.append(model(images.to(device)).argmax(dim=1).cpu())
            targets.append(masks)
    return torch.cat(preds), torch.cat(targets)


def score(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Dataset-level IoU, boundary IoU and the mean over present classes."""
    per_class = iou_per_class(preds, targets)
    boundary = {}
    for i in range(preds.shape[0]):
        for name, value in boundary_iou(preds[i], targets[i]).items():
            boundary.setdefault(name, []).append(value)

    return {
        "iou_per_class": per_class,
        "mIoU": float(np.mean(list(per_class.values()))),
        # Comparable across splits; the plain mIoU above is not, because the
        # class set present in a split varies.
        "mIoU_shared6": mean_iou_over(preds, targets, SHARED_CLASSES),
        "boundary_iou_per_class": {k: float(np.mean(v)) for k, v in boundary.items()},
        "boundary_mIoU": float(np.mean([np.mean(v) for v in boundary.values()])),
    }


def plot_confusion(preds, targets, out_path: Path) -> None:
    matrix = confusion_matrix(preds, targets)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set(xlabel="predicted", ylabel="ground truth", title="Row-normalised confusion")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0.01:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def qualitative_grid(model, dataset, device, num_samples, out_path) -> None:
    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)), squeeze=False)

    for row, idx in enumerate(indices):
        image, gt = dataset[int(idx)]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
        rgb = denormalize(image).permute(1, 2, 0).numpy()
        panels = [
            (rgb, "input"),
            (mask_to_color(gt.numpy()), "ground truth"),
            (mask_to_color(pred.numpy()), "prediction"),
            (overlay(rgb, pred.numpy()), "overlay"),
        ]
        for col, (img, title) in enumerate(panels):
            axes[row][col].imshow(img)
            axes[row][col].set_title(title)
            axes[row][col].axis("off")

    add_class_legend(fig)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/baseline/decoder_best.pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="required for --split test; acknowledges this is a one-shot final measurement",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default="runs/baseline")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "refusing to read the test split without --confirm-final.\n"
            "Tune on --split val; touch test once, when the model is frozen."
        )

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg = cfg.with_overrides(data={"root": args.data_root})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        cfg.data.root,
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        strategy=cfg.data.strategy,
        fractions=tuple(cfg.data.fractions),
        seed=cfg.data.seed,
        block_size=cfg.data.block_size,
        holdout_tracks=cfg.data.holdout_tracks,
    )

    model = build_model(cfg.model).to(args.device)
    if hasattr(model, "load_decoder"):
        model.load_decoder(args.checkpoint, map_location=args.device)
    else:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state.get("decoder", state))
    model.eval()

    preds, targets = collect(model, loaders[args.split], args.device)
    results = score(preds, targets)
    results.update(
        split=args.split,
        strategy=cfg.data.strategy,
        seed=cfg.data.seed,
        checkpoint=args.checkpoint,
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
    )

    (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(results, indent=2))
    if args.split == "test":
        with (out_dir / TEST_LOG).open("a") as handle:
            handle.write(json.dumps(results) + "\n")

    print(f"split={args.split} strategy={cfg.data.strategy}")
    for name, value in results["iou_per_class"].items():
        boundary = results["boundary_iou_per_class"].get(name, float("nan"))
        print(f"  {name:12s} IoU {value:.4f}   boundary {boundary:.4f}")
    print(f"  {'mIoU':12s} {results['mIoU']:.4f}   boundary {results['boundary_mIoU']:.4f}")
    shared = results.get("mIoU_shared6")
    print(f"  {'mIoU (6)':12s} " + (f"{shared:.4f}" if shared is not None else "n/a"))

    plot_confusion(preds, targets, out_dir / f"confusion_{args.split}.png")
    qualitative_grid(
        model, loaders[args.split].dataset, args.device, args.samples,
        out_dir / f"predictions_{args.split}.png",
    )


if __name__ == "__main__":
    main()
