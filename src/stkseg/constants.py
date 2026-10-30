"""Dataset-wide constants shared by every module."""

from __future__ import annotations

import numpy as np
import torch

NUM_CLASSES = 7
CLASS_NAMES = ["Background", "Track", "Kart", "Pickup", "Nitro", "Bomb", "Projectile"]

#: Classes that appear in <0.5% of pixels and drive the oversampling strategy.
RARE_CLASSES = {4, 5, 6}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

COLORS = np.array(
    [
        [0, 0, 0],        # Background
        [128, 128, 128],  # Track
        [255, 0, 0],      # Kart
        [0, 255, 0],      # Pickup
        [0, 0, 255],      # Nitro
        [255, 255, 0],    # Bomb
        [255, 0, 255],    # Projectile
    ],
    dtype=np.uint8,
)


def default_device() -> str:
    """Pick CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = default_device()
