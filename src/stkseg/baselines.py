"""Baselines trained under the identical protocol, so comparisons mean something.

``unet``
    Trained from scratch. Answers "is a pre-trained backbone worth it at all?"
``convnext_frozen``
    An ImageNet backbone frozen and fitted with the same decoder. This is the
    baseline that matters: it isolates whether SAM 2 specifically helps, or
    whether any frozen feature extractor would do.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stkseg.constants import NUM_CLASSES
from stkseg.model import LEVEL_NAMES, SegmentationDecoder


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Compact U-Net trained from scratch (~1.2M parameters by default)."""

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 32, depth: int = 4):
        super().__init__()
        channels = [width * 2**i for i in range(depth)]

        self.downs = nn.ModuleList()
        in_ch = 3
        for ch in channels:
            self.downs.append(_block(in_ch, ch))
            in_ch = ch

        self.bottleneck = _block(channels[-1], channels[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(channels):
            self.up_convs.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.ups.append(_block(ch * 2, ch))

        self.classifier = nn.Conv2d(channels[0], num_classes, 1)

    @property
    def param_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"frozen": 0, "trainable": total, "total": total, "trainable_pct": 100.0}

    def trainable_parameters(self):
        return self.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips), strict=True):
            x = up_conv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([skip, x], dim=1))

        return self.classifier(x)


class ConvNeXtSegmenter(nn.Module):
    """Frozen ImageNet ConvNeXt-Tiny exposing three scales to the shared decoder."""

    #: Indices into torchvision's ``convnext_tiny().features`` after each stage.
    STAGE_OUTPUTS = (3, 5, 7)
    STAGE_CHANNELS = (192, 384, 768)

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        fpn_channels: int = 256,
        width: int = 128,
        dropout: float = 0.1,
        fpn_levels: tuple[str, ...] = LEVEL_NAMES,
        weights: str | None = "DEFAULT",
    ):
        super().__init__()
        from torchvision.models import convnext_tiny

        self.features = convnext_tiny(weights=weights).features
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        # 1x1 projections put every stage on the decoder's expected channel count.
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, fpn_channels, 1) for ch in self.STAGE_CHANNELS]
        )
        self.decoder = SegmentationDecoder(
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            levels=fpn_levels,
        )

    def train(self, mode: bool = True) -> ConvNeXtSegmenter:
        super().train(mode)
        self.features.eval()
        return self

    @property
    def param_counts(self) -> dict[str, int]:
        frozen = sum(p.numel() for p in self.features.parameters())
        trainable = sum(
            p.numel()
            for module in (self.lateral, self.decoder)
            for p in module.parameters()
            if p.requires_grad
        )
        return {
            "frozen": frozen,
            "trainable": trainable,
            "total": frozen + trainable,
            "trainable_pct": round(100 * trainable / (frozen + trainable), 2),
        }

    def trainable_parameters(self):
        return [*self.lateral.parameters(), *self.decoder.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = []
        with torch.no_grad():
            out = x
            for index, layer in enumerate(self.features):
                out = layer(out)
                if index in self.STAGE_OUTPUTS:
                    pyramid.append(out)

        projected = [lateral(f) for lateral, f in zip(self.lateral, pyramid, strict=True)]
        logits = self.decoder(projected)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
