"""Typed experiment configuration, loadable from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "seg_dataset"
    img_size: tuple[int, int] = (512, 512)
    batch_size: int = 4
    #: random | temporal_block | track_holdout - see stkseg.splits
    strategy: str = "temporal_block"
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    block_size: int = 25
    #: Draws the split. Kept named `seed` for backward compatibility with
    #: configs and run directories produced before the seeds were separated.
    seed: int = 42
    #: Seeds weight initialisation and shuffling. When None it falls back to
    #: `seed`, which is what older runs did -- and why their "seed sweep"
    #: varied the split as well as the initialisation, conflating the two.
    init_seed: int | None = None
    #: (val_track, test_track) for track_holdout; None draws them from `seed`.
    holdout_tracks: tuple[str, str] | None = None
    oversample_factor: float = 5.0
    num_workers: int = 0

    @property
    def effective_init_seed(self) -> int:
        return self.seed if self.init_seed is None else self.init_seed


@dataclass
class ModelConfig:
    name: str = "sam2"
    encoder: str = "facebook/sam2.1-hiera-tiny"
    fpn_channels: int = 256
    decoder_width: int = 128
    dropout: float = 0.1
    #: Subset of ("high", "mid", "low") - used by the FPN ablation.
    fpn_levels: tuple[str, ...] = ("high", "mid", "low")


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
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def with_overrides(self, **sections: dict[str, Any]) -> Config:
        """Return a copy with per-section field overrides applied.

        Used by the ablation and seed runners so a base config can be varied
        along one axis at a time without editing files on disk.
        """
        updated = {}
        for name in ("data", "model", "loss", "train"):
            section = getattr(self, name)
            updated[name] = replace(section, **sections.get(name, {}))
        return Config(**updated)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
