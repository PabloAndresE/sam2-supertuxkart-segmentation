import torch

from stkseg.constants import NUM_CLASSES
from stkseg.model import SegmentationDecoder


def test_decoder_output_shape_matches_highest_fpn_level():
    decoder = SegmentationDecoder()
    features = [
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
    ]
    out = decoder(features)
    assert out.shape == (2, NUM_CLASSES, 32, 32)


def test_decoder_parameter_budget():
    params = sum(p.numel() for p in SegmentationDecoder().parameters())
    # The selling point of the project: the trainable head stays around 2.1M.
    assert 1_500_000 < params < 2_500_000
