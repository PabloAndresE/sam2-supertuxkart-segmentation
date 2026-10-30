"""Evaluate a trained decoder: dataset-level IoU per class and qualitative figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from stkseg.config import Config
from stkseg.constants import DEVICE
from stkseg.data import create_dataloaders
from stkseg.metrics import iou_per_class
from stkseg.model import SAM2Segmenter
from stkseg.viz import add_class_legend, denormalize, mask_to_color, overlay


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
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default="runs/baseline")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = create_dataloaders(
        cfg.data.root,
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        seed=cfg.data.seed,
    )

    model = SAM2Segmenter(
        encoder_name=cfg.model.encoder,
        fpn_channels=cfg.model.fpn_channels,
        width=cfg.model.decoder_width,
        dropout=cfg.model.dropout,
    ).to(args.device)
    model.load_decoder(args.checkpoint, map_location=args.device)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            preds.append(model(images.to(args.device)).argmax(dim=1).cpu())
            targets.append(masks)

    # Concatenated over the whole split: a class absent from one batch is still
    # scored, which is why this number is the honest one to report.
    scores = iou_per_class(torch.cat(preds), torch.cat(targets))
    scores["mIoU"] = float(np.mean(list(scores.values())))

    (out_dir / "metrics.json").write_text(json.dumps(scores, indent=2))
    for name, value in scores.items():
        print(f"  {name:12s} {value:.4f}")

    qualitative_grid(
        model, val_loader.dataset, args.device, args.samples, out_dir / "predictions.png"
    )


if __name__ == "__main__":
    main()
