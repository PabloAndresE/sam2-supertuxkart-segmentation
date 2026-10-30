"""Frozen SAM 2 vision encoder plus a trainable multi-scale segmentation decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stkseg.constants import NUM_CLASSES

LEVEL_NAMES = ("high", "mid", "low")

#: The original coursework decoder used one Sequential per level named
#: ``conv_<level>``. The level-selectable rewrite stores them in a ModuleDict,
#: so checkpoints trained before that change need their keys renamed.
LEGACY_PREFIXES = {f"conv_{name}.": f"project.{name}." for name in LEVEL_NAMES}


def remap_legacy_decoder_keys(state: dict) -> dict:
    """Rename ``conv_high.*`` style keys to the current ``project.high.*`` layout.

    Returns the state dict unchanged when it already uses the current naming.
    """
    if not any(key.startswith(old) for key in state for old in LEGACY_PREFIXES):
        return state

    remapped = {}
    for key, value in state.items():
        for old, new in LEGACY_PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        remapped[key] = value
    return remapped


class SegmentationDecoder(nn.Module):
    """Fuse FPN levels into a dense class map.

    Each selected level is projected to ``width`` channels, resampled to the
    highest requested resolution and concatenated, so fine spatial detail from
    the shallow level survives while deep semantics still reach the classifier.

    ``levels`` selects which FPN outputs participate, which is what the
    multi-scale ablation varies.
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        num_classes: int = NUM_CLASSES,
        width: int = 128,
        dropout: float = 0.1,
        levels: tuple[str, ...] = LEVEL_NAMES,
    ):
        super().__init__()
        unknown = set(levels) - set(LEVEL_NAMES)
        if unknown:
            raise ValueError(f"unknown FPN levels {sorted(unknown)}, expected {LEVEL_NAMES}")
        if not levels:
            raise ValueError("at least one FPN level is required")

        self.levels = tuple(name for name in LEVEL_NAMES if name in levels)

        self.project = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(fpn_channels, width, 3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
                for name in self.levels
            }
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width * len(self.levels), width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(width, num_classes, 1)

    def forward(self, fpn_features: list[torch.Tensor]) -> torch.Tensor:
        by_name = dict(zip(LEVEL_NAMES, fpn_features, strict=True))
        target_size = by_name[self.levels[0]].shape[2:]

        projected = [
            self.project[name](
                by_name[name]
                if by_name[name].shape[2:] == target_size
                else F.interpolate(
                    by_name[name], size=target_size, mode="bilinear", align_corners=False
                )
            )
            for name in self.levels
        ]
        return self.classifier(self.fuse(torch.cat(projected, dim=1)))


class SAM2Segmenter(nn.Module):
    """SAM 2 Hiera-Tiny encoder (frozen) + :class:`SegmentationDecoder` (trainable)."""

    def __init__(
        self,
        encoder_name: str = "facebook/sam2.1-hiera-tiny",
        num_classes: int = NUM_CLASSES,
        fpn_channels: int = 256,
        width: int = 128,
        dropout: float = 0.1,
        fpn_levels: tuple[str, ...] = LEVEL_NAMES,
    ):
        super().__init__()
        from transformers import Sam2Model

        self.encoder = Sam2Model.from_pretrained(encoder_name).vision_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = SegmentationDecoder(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            levels=fpn_levels,
        )

    def train(self, mode: bool = True) -> SAM2Segmenter:
        """Keep the encoder in eval mode so its BatchNorm statistics stay frozen."""
        super().train(mode)
        self.encoder.eval()
        return self

    @property
    def param_counts(self) -> dict[str, int]:
        frozen = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return {
            "frozen": frozen,
            "trainable": trainable,
            "total": frozen + trainable,
            "trainable_pct": round(100 * trainable / (frozen + trainable), 2),
        }

    def trainable_parameters(self):
        return self.decoder.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.encoder(x)
        logits = self.decoder(list(encoded.fpn_hidden_states))
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    def load_decoder(self, path: str, map_location: str = "cpu") -> None:
        """Load a decoder-only checkpoint (~8 MB) instead of the full 117 MB state dict.

        Accepts checkpoints saved before the level-selectable decoder rewrite.
        """
        state = torch.load(path, map_location=map_location, weights_only=True)
        state = state.get("decoder", state)
        self.decoder.load_state_dict(remap_legacy_decoder_keys(state))
