"""SuperTuxKart segmentation dataset, leakage-aware splits and dataloaders."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from stkseg.constants import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, RARE_CLASSES
from stkseg.splits import SampleKey, describe, make_splits

FRAME_RE = re.compile(r"^(?P<track>.+)_frame_(?P<index>\d+)$")


class SuperTuxKartDataset(Dataset):
    """Paired frames and pixel-level masks exported from the pystk engine.

    Expected layout::

        root/
            images/abyss_frame_0128.png
            masks/abyss_mask_combined_0128.png

    Pairing is by track name and frame index, not by directory order.
    """

    def __init__(self, root: str | Path, img_size=(512, 512), augment: bool = False):
        self.root = Path(root)
        self.img_size = tuple(img_size)
        self.augment = augment
        self.samples: list[tuple[Path, Path]] = []
        self.keys: list[SampleKey] = []
        self.sample_classes: list[set[int]] = []

        images_dir = self.root / "images"
        masks_dir = self.root / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"missing images directory: {images_dir}")

        for frame_path in sorted(images_dir.glob("*.png")):
            matched = FRAME_RE.match(frame_path.stem)
            if not matched:
                continue
            track, index = matched["track"], matched["index"]
            mask_path = masks_dir / f"{track}_mask_combined_{index}.png"
            if not mask_path.exists():
                continue
            self.samples.append((frame_path, mask_path))
            self.keys.append(SampleKey(track=track, frame_index=int(index)))
            self.sample_classes.append(set(np.unique(np.array(Image.open(mask_path))).tolist()))

        if not self.samples:
            raise RuntimeError(f"no image/mask pairs found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image, mask: Image.Image):
        if random.random() > 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        if random.random() > 0.3:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
        if random.random() > 0.3:
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
        if random.random() > 0.3:
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
        return image, mask

    def __getitem__(self, index: int):
        frame_path, mask_path = self.samples[index]

        image = Image.open(frame_path).convert("RGB").resize(self.img_size, Image.BILINEAR)
        # NEAREST only: bilinear would invent class ids that do not exist.
        mask = Image.open(mask_path).convert("L").resize(self.img_size, Image.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask = torch.from_numpy(np.array(mask)).long().clamp(0, NUM_CLASSES - 1)
        return image, mask


def build_dataloaders(
    root: str | Path,
    img_size=(512, 512),
    batch_size: int = 4,
    strategy: str = "temporal_block",
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    block_size: int = 25,
    holdout_tracks: tuple[str, str] | None = None,
    oversample_factor: float = 5.0,
    num_workers: int = 0,
    verbose: bool = True,
) -> dict[str, DataLoader]:
    """Train/val/test loaders with augmentation and oversampling on train only.

    Validation and test read from a non-augmented dataset instance so reported
    scores describe untouched frames.
    """
    clean = SuperTuxKartDataset(root, img_size=img_size, augment=False)
    augmented = SuperTuxKartDataset(root, img_size=img_size, augment=True)

    splits = make_splits(
        clean.keys,
        strategy=strategy,
        fractions=fractions,
        seed=seed,
        block_size=block_size,
        holdout_tracks=holdout_tracks,
    )
    if verbose:
        print(f"split strategy: {strategy} (seed {seed})")
        print(describe(clean.keys, splits))

    train_idx = splits["train"]
    weights = [
        oversample_factor if clean.sample_classes[i] & RARE_CLASSES else 1.0 for i in train_idx
    ]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loaders = {
        "train": DataLoader(
            Subset(augmented, train_idx),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    }
    for split in ("val", "test"):
        loaders[split] = DataLoader(
            Subset(clean, splits[split]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return loaders
