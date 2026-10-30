import numpy as np
import pytest
import torch

from stkseg.metrics import boundary_iou, confusion_matrix, iou_per_class, mean_iou


def test_perfect_prediction_scores_one():
    target = torch.tensor([[0, 1], [2, 2]])
    assert mean_iou(target.clone(), target) == 1.0


def test_absent_classes_are_skipped():
    target = torch.tensor([[0, 0], [1, 1]])
    assert set(iou_per_class(target.clone(), target)) == {"Background", "Track"}


def test_half_overlap():
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 1, 1])
    scores = iou_per_class(pred, target)
    assert abs(scores["Background"] - 0.5) < 1e-6
    assert abs(scores["Track"] - 2 / 3) < 1e-6


def test_confusion_matrix_rows_sum_to_one_where_class_present():
    target = torch.tensor([[0, 0], [1, 1]])
    pred = torch.tensor([[0, 1], [1, 1]])
    matrix = confusion_matrix(pred, target)
    assert np.isclose(matrix[0].sum(), 1.0)
    assert np.isclose(matrix[1].sum(), 1.0)
    assert np.isclose(matrix[2].sum(), 0.0)  # class absent from the target
    assert np.isclose(matrix[0, 0], 0.5)


def test_confusion_matrix_is_perfect_for_exact_prediction():
    target = torch.randint(0, 7, (16, 16))
    matrix = confusion_matrix(target.clone(), target)
    assert np.allclose(np.diag(matrix)[np.diag(matrix) > 0], 1.0)


def test_boundary_iou_perfect_on_identical_masks():
    mask = torch.zeros(32, 32, dtype=torch.long)
    mask[8:24, 8:24] = 2
    scores = boundary_iou(mask.clone(), mask)
    assert all(abs(v - 1.0) < 1e-6 for v in scores.values())


def test_boundary_iou_punishes_a_shifted_mask_more_than_plain_iou():
    truth = torch.zeros(64, 64, dtype=torch.long)
    truth[16:48, 16:48] = 2
    shifted = torch.zeros(64, 64, dtype=torch.long)
    shifted[19:51, 19:51] = 2

    plain = iou_per_class(shifted, truth)["Kart"]
    band = boundary_iou(shifted, truth)["Kart"]
    assert band < plain


def test_boundary_iou_rejects_batched_input():
    with pytest.raises(ValueError, match="2-D masks"):
        boundary_iou(torch.zeros(2, 8, 8, dtype=torch.long), torch.zeros(2, 8, 8, dtype=torch.long))
