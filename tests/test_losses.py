import torch

from stkseg.constants import NUM_CLASSES
from stkseg.losses import CombinedLoss, DiceLoss


def _logits_matching(target, confidence=20.0):
    one_hot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    return one_hot * confidence


def test_dice_loss_near_zero_for_confident_correct_prediction():
    target = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    assert DiceLoss()(_logits_matching(target), target).item() < 0.05


def test_dice_loss_is_bounded():
    target = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    loss = DiceLoss()(torch.randn(2, NUM_CLASSES, 8, 8), target)
    assert 0.0 <= loss.item() <= 1.0


def test_combined_loss_respects_weights():
    target = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    logits = torch.randn(2, NUM_CLASSES, 8, 8)
    weights = torch.ones(NUM_CLASSES)

    ce_only = CombinedLoss(weights, ce_weight=1.0, dice_weight=0.0)(logits, target)
    dice_only = CombinedLoss(weights, ce_weight=0.0, dice_weight=1.0)(logits, target)
    blended = CombinedLoss(weights, ce_weight=0.3, dice_weight=0.7)(logits, target)

    assert abs(blended.item() - (0.3 * ce_only.item() + 0.7 * dice_only.item())) < 1e-5
