import pytest
import torch

from stkseg.config import ModelConfig
from stkseg.model import SegmentationDecoder
from stkseg.registry import build_model


def _pyramid(batch=2, channels=256):
    return [
        torch.randn(batch, channels, 32, 32),
        torch.randn(batch, channels, 16, 16),
        torch.randn(batch, channels, 8, 8),
    ]


@pytest.mark.parametrize(
    "levels,expected_inputs",
    [(("high",), 1), (("high", "mid"), 2), (("high", "mid", "low"), 3)],
)
def test_decoder_accepts_level_subsets(levels, expected_inputs):
    decoder = SegmentationDecoder(levels=levels)
    assert len(decoder.project) == expected_inputs
    assert decoder(_pyramid()).shape == (2, 7, 32, 32)


def test_decoder_uses_highest_selected_resolution():
    """With only the low level selected, output follows that level's size."""
    decoder = SegmentationDecoder(levels=("low",))
    assert decoder(_pyramid()).shape == (2, 7, 8, 8)


def test_decoder_rejects_unknown_level():
    with pytest.raises(ValueError, match="unknown FPN levels"):
        SegmentationDecoder(levels=("high", "ultra"))


def test_decoder_rejects_empty_levels():
    with pytest.raises(ValueError, match="at least one FPN level"):
        SegmentationDecoder(levels=())


def test_fewer_levels_means_fewer_parameters():
    counts = [
        sum(p.numel() for p in SegmentationDecoder(levels=lv).parameters())
        for lv in (("high",), ("high", "mid"), ("high", "mid", "low"))
    ]
    assert counts[0] < counts[1] < counts[2]


def test_unet_baseline_runs_and_is_fully_trainable():
    model = build_model(ModelConfig(name="unet"))
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 7, 64, 64)
    assert model.param_counts["trainable_pct"] == 100.0


def test_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_model(ModelConfig(name="segformer"))


def test_legacy_checkpoint_keys_are_remapped():
    """Checkpoints from before the ModuleDict rewrite must still load."""
    from stkseg.model import remap_legacy_decoder_keys

    decoder = SegmentationDecoder()
    current = decoder.state_dict()
    legacy = {
        key.replace("project.high.", "conv_high.")
        .replace("project.mid.", "conv_mid.")
        .replace("project.low.", "conv_low."): value
        for key, value in current.items()
    }
    assert any(k.startswith("conv_high.") for k in legacy)
    decoder.load_state_dict(remap_legacy_decoder_keys(legacy))


def test_remap_is_a_noop_for_current_checkpoints():
    from stkseg.model import remap_legacy_decoder_keys

    current = SegmentationDecoder().state_dict()
    assert remap_legacy_decoder_keys(current) is current
