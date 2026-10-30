"""Visualisation helpers: colourised masks, overlays and figure builders."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

from stkseg.constants import CLASS_NAMES, COLORS, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    return COLORS[mask]


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a float [0,1] RGB image with its colourised mask."""
    return (image_rgb * (1 - alpha) + mask_to_color(mask) / 255.0 * alpha)


def add_class_legend(fig) -> None:
    handles = [
        mpatches.Patch(color=np.array(COLORS[i]) / 255, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=NUM_CLASSES, fontsize=9)


def plot_history(history: dict, out_path: str) -> None:
    """Loss / mIoU / learning-rate curves for a finished run."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss (0.3 CE + 0.7 Dice)")
    axes[0].legend()

    best = max(history["val_iou"])
    axes[1].plot(epochs, history["val_iou"], color="green")
    axes[1].axhline(best, color="red", ls="--", label=f"best {best:.4f}")
    axes[1].set(xlabel="epoch", ylabel="mIoU", title="Validation mIoU")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], color="orange")
    axes[2].set(xlabel="epoch", ylabel="lr", title="Learning rate schedule")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
