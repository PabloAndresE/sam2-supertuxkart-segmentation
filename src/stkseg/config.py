"""Typed experiment configuration, loadable from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "seg_dataset"
    img_size: tuple[int, int] = (512, 512)
    batch_size: int = 4
    val_split: float = 0.2
    seed: int = 42
    oversample_factor: float = 5.0
    num_workers: int = 0


@dataclass
class ModelConfig:
    encoder: str = "facebook/sam2.1-hiera-tiny"
    fpn_channels: int = 256
    decoder_width: int = 128
    dropout: float = 0.1


@dataclass
class LossConfig:
    ce_weight: float = 0.3
    dice_weight: float = 0.7
    class_weights: list[float] = field(
        default_factory=lambda: [0.5, 0.8, 3.0, 5.0, 8.0, 10.0, 15.0]
    )


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 5e-4
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    out_dir: str = "runs/baseline"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
