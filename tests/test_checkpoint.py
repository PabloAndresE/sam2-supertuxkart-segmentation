import torch

from stkseg.baselines import ConvNeXtSegmenter
from stkseg.config import ModelConfig
from stkseg.registry import build_model
from stkseg.train import checkpoint_state


def _trainable_names(model):
    return {name for name, p in model.named_parameters() if p.requires_grad}


def test_convnext_checkpoint_keeps_lateral_convs():
    """Regression: saving only `decoder` silently dropped the trainable laterals."""
    model = ConvNeXtSegmenter(weights=None)
    state = checkpoint_state(model)
    assert _trainable_names(model) <= set(state)
    assert any(key.startswith("lateral.") for key in state)


def test_convnext_checkpoint_round_trips_through_load_state_dict():
    source = ConvNeXtSegmenter(weights=None)
    target = ConvNeXtSegmenter(weights=None)
    target.load_state_dict(checkpoint_state(source))
    for name, param in target.named_parameters():
        assert torch.equal(param, source.state_dict()[name])


def test_unet_checkpoint_is_the_full_model():
    model = build_model(ModelConfig(name="unet"))
    assert set(checkpoint_state(model)) == set(model.state_dict())
