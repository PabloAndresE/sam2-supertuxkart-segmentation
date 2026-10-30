"""Run a trained segmenter on a single image and save a side-by-side figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image

from stkseg.config import Config
from stkseg.constants import CLASS_NAMES, DEVICE, IMAGENET_MEAN, IMAGENET_STD
from stkseg.registry import build_model
from stkseg.viz import add_class_legend, mask_to_color


def load_model(config_path: str, checkpoint: str, device: str) -> torch.nn.Module:
    cfg = Config.from_yaml(config_path)
    model = build_model(cfg.model).to(device)
    if hasattr(model, "load_decoder"):
        model.load_decoder(checkpoint, map_location=device)
    else:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state.get("decoder", state))
    model.eval()
    return model


def segment(model, image: Image.Image, img_size=(512, 512), device: str = "cpu"):
    """Return the predicted class map plus the resized RGB input."""
    resized = image.convert("RGB").resize(tuple(img_size), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), mean=IMAGENET_MEAN, std=IMAGENET_STD)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
    return logits.argmax(dim=1).squeeze(0).cpu().numpy(), np.array(resized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/baseline/decoder_best.pt")
    parser.add_argument("--out", default="prediction.png")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    model = load_model(args.config, args.checkpoint, args.device)
    pred, rgb = segment(model, Image.open(args.image), cfg.data.img_size, args.device)

    colored = mask_to_color(pred)
    blended = (rgb * 0.5 + colored * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axes, (rgb, colored, blended), ("input", "segmentation", "overlay"), strict=True
    ):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    add_class_legend(fig)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    print(f"classes detected: {[CLASS_NAMES[c] for c in np.unique(pred)]}")
    print(f"saved {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
