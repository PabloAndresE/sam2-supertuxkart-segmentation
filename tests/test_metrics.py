import torch

from stkseg.metrics import iou_per_class, mean_iou


def test_perfect_prediction_scores_one():
    target = torch.tensor([[0, 1], [2, 2]])
    assert mean_iou(target.clone(), target) == 1.0


def test_absent_classes_are_skipped():
    target = torch.tensor([[0, 0], [1, 1]])
    scores = iou_per_class(target.clone(), target)
    assert set(scores) == {"Background", "Track"}


def test_half_overlap():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 1, 1])
    scores = iou_per_class(pred, target)
    # Background: 1 correct, union 2 -> 0.5 ; Track: 2 correct, union 3 -> 0.667
    assert abs(scores["Background"] - 0.5) < 1e-6
    assert abs(scores["Track"] - 2 / 3) < 1e-6
