"""Check everything a run needs before queuing hours of GPU time.

Every failure here is one that would otherwise surface after the job starts:
a missing dataset, an encoder that cannot be downloaded on a compute node with
no internet, a batch size that does not fit, a split that cannot be built.

    python scripts/preflight.py --config configs/server.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from stkseg.config import Config
from stkseg.data import SuperTuxKartDataset
from stkseg.registry import build_model
from stkseg.splits import make_splits

CHECKS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, ok, detail))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"preflight: config={args.config} device={device}\n")

    print("environment")
    record("torch", True, torch.__version__)
    if device == "cuda":
        record("cuda available", torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            record("gpu", True, f"{props.name}, {props.total_memory / 1e9:.1f} GB")
    else:
        # Informational, not a failure: a CPU/MPS preflight is a valid dry run.
        print("  [--  ] cuda available - no; dry run only, real training needs a GPU")

    print("\ndata")
    root = Path(cfg.data.root)
    if not record("dataset directory", root.is_dir(), str(root.resolve())):
        print("\nhint: unzip the dataset archive into the repo root first.")
        return 1

    dataset = SuperTuxKartDataset(root, img_size=cfg.data.img_size)
    record("image/mask pairs", len(dataset) > 0, f"{len(dataset)} pairs")
    tracks = sorted({k.track for k in dataset.keys})
    record("tracks", len(tracks) >= 3, f"{len(tracks)}: {', '.join(tracks)}")

    splits = make_splits(
        dataset.keys,
        strategy=cfg.data.strategy,
        fractions=tuple(cfg.data.fractions),
        seed=cfg.data.seed,
        block_size=cfg.data.block_size,
    )
    sizes = {k: len(v) for k, v in splits.items()}
    record("splits build", all(v > 0 for v in sizes.values()), str(sizes))

    image, mask = dataset[0]
    record(
        "sample shapes",
        image.shape[1:] == tuple(cfg.data.img_size) and mask.shape == tuple(cfg.data.img_size),
        f"image {tuple(image.shape)} mask {tuple(mask.shape)}",
    )

    print("\nmodel")
    try:
        # On an offline compute node this is where a missing HF cache shows up.
        model = build_model(cfg.model).to(device)
        record("weights load", True, str(model.param_counts))
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        record("weights load", False, f"{type(exc).__name__}: {exc}")
        print("\nhint: pre-download the encoder on a login node, then export HF_HOME.")
        return 1

    print("\nthroughput")
    # Timed on the REAL dataloader, not a synthetic tensor: with num_workers=0 the
    # PNG decode and 512px resize dominate, and a synthetic-tensor estimate came
    # out 4x optimistic against a measured epoch.
    from stkseg.data import build_dataloaders
    from stkseg.train import autocast_context

    loaders = build_dataloaders(
        cfg.data.root,
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        strategy=cfg.data.strategy,
        fractions=tuple(cfg.data.fractions),
        seed=cfg.data.seed,
        block_size=cfg.data.block_size,
        num_workers=cfg.data.num_workers,
        verbose=False,
    )

    try:
        model.train()
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        warmup, timed = 2, 8
        durations = []
        for step, (images, masks) in enumerate(loaders["train"]):
            if step >= warmup + timed:
                break
            started = time.perf_counter()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast_context(device, args.amp):
                loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            if step >= warmup:
                durations.append(time.perf_counter() - started)

        per_step = sum(durations) / len(durations)
        steps = -(-sizes["train"] // cfg.data.batch_size)
        val_steps = -(-sizes["val"] // cfg.data.batch_size)
        epoch_s = per_step * (steps + val_steps * 0.4)
        record(
            f"train step at batch {cfg.data.batch_size} (real data)",
            True,
            f"{per_step * 1000:.0f} ms/step, ~{epoch_s:.0f} s/epoch "
            f"-> ~{epoch_s * cfg.train.epochs / 3600:.1f} h for {cfg.train.epochs} epochs",
        )
        if cfg.data.num_workers == 0:
            print(
                "  [note] num_workers=0 - data loading is serialised with compute; "
                "raise it to 8 on a server"
            )
        if device == "cuda":
            record("peak memory", True, f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    except torch.cuda.OutOfMemoryError:
        record("train step", False, f"OOM at batch_size {cfg.data.batch_size}; lower it")
        return 1

    failures = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failures:
        print(f"preflight FAILED: {', '.join(failures)}")
        return 1
    print("preflight passed - safe to queue the suite")
    print(
        "note: the epoch estimate is a lower bound - it warms the page cache and\n"
        "      approximates the validation pass. A measured M1 Pro epoch came in\n"
        "      1.6x above its own estimate. Size SLURM --time with headroom."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
