"""Intersection-over-Union metrics."""

from __future__ import annotations

import numpy as np
import torch

from stkseg.constants import CLASS_NAMES, NUM_CLASSES


def confusion_counts(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES):
    """Return per-class intersection and union pixel counts."""
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter[cls] = (p & t).sum()
        union[cls] = (p | t).sum()
    return inter, union


def iou_per_class(pred, target, num_classes: int = NUM_CLASSES) -> dict[str, float]:
    """IoU for every class that is present in the prediction or the target."""
    inter, union = confusion_counts(pred, target, num_classes)
    return {
        CLASS_NAMES[c]: float(inter[c] / union[c])
        for c in range(num_classes)
        if union[c] > 0
    }


def mean_iou(pred, target, num_classes: int = NUM_CLASSES) -> float:
    """Mean IoU averaged over the classes present. Absent classes are skipped.

    Note: a class never present in a batch is excluded rather than scored 0,
    so batch-level mIoU is optimistic compared to the dataset-level number.
    """
    values = list(iou_per_class(pred, target, num_classes).values())
    return float(np.mean(values)) if values else 0.0
