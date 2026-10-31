"""Segmentation metrics: IoU, boundary IoU and confusion matrix."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

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
        CLASS_NAMES[c]: float(inter[c] / union[c]) for c in range(num_classes) if union[c] > 0
    }


def mean_iou(pred, target, num_classes: int = NUM_CLASSES) -> float:
    """Mean IoU averaged over the classes present. Absent classes are skipped.

    Note: a class never present in a batch is excluded rather than scored 0,
    so batch-level mIoU is optimistic compared to the dataset-level number.
    """
    values = list(iou_per_class(pred, target, num_classes).values())
    return float(np.mean(values)) if values else 0.0


def classes_in_ground_truth(target, classes: Sequence[str], num_classes: int = NUM_CLASSES):
    """Which of ``classes`` actually occur in the ground truth."""
    present = set(torch.unique(target.reshape(-1)).tolist())
    return [name for name in classes if CLASS_NAMES.index(name) in present]


def hallucinated_classes(pred, target, classes: Sequence[str], num_classes: int = NUM_CLASSES):
    """Classes the model predicted that do not occur in the ground truth at all."""
    predicted = set(torch.unique(pred.reshape(-1)).tolist())
    truth = set(torch.unique(target.reshape(-1)).tolist())
    return [
        name
        for name in classes
        if CLASS_NAMES.index(name) in predicted - truth
    ]


def mean_iou_over(
    pred, target, classes: Sequence[str], num_classes: int = NUM_CLASSES
) -> float | None:
    """Mean IoU over the requested classes **that occur in the ground truth**.

    Two failure modes this avoids:

    - :func:`iou_per_class` keys on a non-empty union, so a class the model
      invents in a split where it does not exist enters the mean at 0.0. That is
      a false positive being scored as a missed class; it belongs in
      :func:`hallucinated_classes`, not in the average.
    - Splits containing different class sets otherwise average over different
      denominators and are not comparable.

    Returns ``None`` when none of the requested classes occur, rather than
    inventing a number. Scoring only ground-truth classes means the denominator
    is a property of the data, not of the model's mistakes.
    """
    scoreable = classes_in_ground_truth(target, classes, num_classes)
    if not scoreable:
        return None
    scores = iou_per_class(pred, target, num_classes)
    return float(np.mean([scores.get(name, 0.0) for name in scoreable]))


#: The six classes present in every split of this dataset. Projectile occurs in
#: 2 of 1500 frames, so whether it lands in a given validation set is luck, and
#: its 0.0000 shifts a 7-class mean by ~0.13.
SHARED_CLASSES = ("Background", "Track", "Kart", "Pickup", "Nitro", "Bomb")


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Row-normalised confusion matrix; entry (i, j) is P(predict j | truth i)."""
    pred = pred.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    flat = torch.bincount(
        target * num_classes + pred, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    counts = flat.to(torch.float64).numpy()
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def _boundary_region(mask: torch.Tensor, num_classes: int, dilation: int) -> torch.Tensor:
    """Pixels within ``dilation`` of a class boundary, as a (C, H, W) bool tensor."""
    one_hot = F.one_hot(mask.to(torch.int64), num_classes)
    one_hot = one_hot.permute(2, 0, 1).float().unsqueeze(0)
    kernel = 2 * dilation + 1
    eroded = -F.max_pool2d(-one_hot, kernel_size=kernel, stride=1, padding=dilation)
    return (one_hot - eroded).squeeze(0).bool()


def boundary_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    dilation: int = 2,
) -> dict[str, float]:
    """IoU restricted to a thin band around object boundaries.

    Standard IoU is dominated by region interiors, so a model can score well
    while producing ragged edges. Since the whole premise here is that SAM 2
    contributes boundary quality, measuring it directly is the honest check.
    Expects a single 2-D mask pair.
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("boundary_iou expects 2-D masks (H, W)")

    pred_band = _boundary_region(pred, num_classes, dilation)
    target_band = _boundary_region(target, num_classes, dilation)

    scores = {}
    for cls in range(num_classes):
        p, t = pred_band[cls], target_band[cls]
        union = (p | t).sum().item()
        if union > 0:
            scores[CLASS_NAMES[cls]] = (p & t).sum().item() / union
    return scores
