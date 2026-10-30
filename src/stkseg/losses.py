"""Loss functions tuned for a severely imbalanced 7-class mask."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stkseg.constants import NUM_CLASSES


class DiceLoss(nn.Module):
    """Multiclass soft Dice.

    Dice is computed per class over the whole batch, so a class covering 0.1% of
    pixels contributes as much to the loss as the background does.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted cross-entropy blended with Dice (default 30/70)."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        ce_weight: float = 0.3,
        dice_weight: float = 0.7,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )
