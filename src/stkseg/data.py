"""SuperTuxKart segmentation dataset and dataloader construction."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from stkseg.constants import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, RARE_CLASSES


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
        self.sample_classes: list[set[int]] = []

        images_dir = self.root / "images"
        masks_dir = self.root / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"missing images directory: {images_dir}")

        for frame_path in sorted(images_dir.glob("*.png")):
            parts = frame_path.stem.split("_frame_")
            if len(parts) != 2:
                continue
            track_name, idx = parts
            mask_path = masks_dir / f"{track_name}_mask_combined_{idx}.png"
            if not mask_path.exists():
                continue
            self.samples.append((frame_path, mask_path))
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


def create_dataloaders(
    root: str | Path,
    img_size=(512, 512),
    batch_size: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    oversample_factor: float = 5.0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split train/val on a fixed seed; augment and oversample the train half only.

    Validation reuses the non-augmented dataset instance so that reported mIoU
    is measured on untouched frames.
    """
    clean = SuperTuxKartDataset(root, img_size=img_size, augment=False)
    augmented = SuperTuxKartDataset(root, img_size=img_size, augment=True)

    val_size = int(len(clean) * val_split)
    train_set, val_set = random_split(
        clean,
        [len(clean) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_aug = Subset(augmented, train_set.indices)

    weights = [
        oversample_factor if clean.sample_classes[i] & RARE_CLASSES else 1.0
        for i in train_set.indices
    ]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_aug, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
