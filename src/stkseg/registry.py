"""Build a model from its config, so every runner shares one entry point."""

from __future__ import annotations

from torch import nn

from stkseg.config import ModelConfig

MODELS = ("sam2", "convnext_frozen", "unet")


def build_model(cfg: ModelConfig, num_classes: int = 7) -> nn.Module:
    """Instantiate the model named in ``cfg.name``."""
    if cfg.name == "sam2":
        from stkseg.model import SAM2Segmenter

        return SAM2Segmenter(
            encoder_name=cfg.encoder,
            num_classes=num_classes,
            fpn_channels=cfg.fpn_channels,
            width=cfg.decoder_width,
            dropout=cfg.dropout,
            fpn_levels=tuple(cfg.fpn_levels),
        )

    if cfg.name == "convnext_frozen":
        from stkseg.baselines import ConvNeXtSegmenter

        return ConvNeXtSegmenter(
            num_classes=num_classes,
            fpn_channels=cfg.fpn_channels,
            width=cfg.decoder_width,
            dropout=cfg.dropout,
            fpn_levels=tuple(cfg.fpn_levels),
        )

    if cfg.name == "unet":
        from stkseg.baselines import UNet

        return UNet(num_classes=num_classes, width=cfg.decoder_width // 4)

    raise ValueError(f"unknown model {cfg.name!r}, expected one of {MODELS}")
