"""Frozen SAM 2 vision encoder plus a trainable multi-scale segmentation decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from stkseg.constants import NUM_CLASSES


class SegmentationDecoder(nn.Module):
    """Fuse three FPN levels into a dense class map.

    Each level is projected to ``width`` channels, resampled to the highest
    available resolution and concatenated, so fine spatial detail from the
    shallow level survives while deep semantics still reach the classifier.
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        num_classes: int = NUM_CLASSES,
        width: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        def project() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(fpn_channels, width, 3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )

        self.conv_high = project()
        self.conv_mid = project()
        self.conv_low = project()

        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(width, num_classes, 1)

    def forward(self, fpn_features: list[torch.Tensor]) -> torch.Tensor:
        high, mid, low = fpn_features
        target_size = high.shape[2:]

        def up(tensor: torch.Tensor) -> torch.Tensor:
            return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

        high = self.conv_high(high)
        mid = self.conv_mid(up(mid))
        low = self.conv_low(up(low))

        return self.classifier(self.fuse(torch.cat([high, mid, low], dim=1)))


class SAM2Segmenter(nn.Module):
    """SAM 2 Hiera-Tiny encoder (frozen) + :class:`SegmentationDecoder` (trainable)."""

    def __init__(
        self,
        encoder_name: str = "facebook/sam2.1-hiera-tiny",
        num_classes: int = NUM_CLASSES,
        fpn_channels: int = 256,
        width: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import Sam2Model

        self.encoder = Sam2Model.from_pretrained(encoder_name).vision_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.decoder = SegmentationDecoder(
            fpn_channels=fpn_channels, num_classes=num_classes, width=width, dropout=dropout
        )

    def train(self, mode: bool = True) -> SAM2Segmenter:
        """Keep the encoder in eval mode so its BatchNorm statistics stay frozen."""
        super().train(mode)
        self.encoder.eval()
        return self

    @property
    def param_counts(self) -> dict[str, int]:
        frozen = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.decoder.parameters())
        return {
            "frozen": frozen,
            "trainable": trainable,
            "total": frozen + trainable,
            "trainable_pct": round(100 * trainable / (frozen + trainable), 2),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.encoder(x)
        logits = self.decoder(list(encoded.fpn_hidden_states))
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    def load_decoder(self, path: str, map_location: str = "cpu") -> None:
        """Load a decoder-only checkpoint (~8 MB) instead of the full 117 MB state dict."""
        state = torch.load(path, map_location=map_location, weights_only=True)
        state = state.get("decoder", state)
        self.decoder.load_state_dict(state)
