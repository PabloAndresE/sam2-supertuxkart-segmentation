"""Train a segmentation model. Selection happens on val; test is never read here."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from stkseg.config import Config
from stkseg.constants import DEVICE
from stkseg.data import build_dataloaders
from stkseg.losses import CombinedLoss
from stkseg.metrics import mean_iou
from stkseg.registry import build_model
from stkseg.viz import plot_history


def autocast_context(device: str, enabled: bool):
    """Mixed precision on CUDA only.

    MPS autocast is not reliable for this graph and CPU bf16 is slower than
    fp32 here, so AMP stays opt-in and CUDA-only rather than silently changing
    numerics on a laptop.
    """
    if enabled and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def evaluate_split(model, loader, criterion, device, amp: bool = False) -> tuple[float, float]:
    model.eval()
    total_loss, total_iou, batches = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with autocast_context(device, amp):
                logits = model(images)
                loss = criterion(logits, masks)
            total_loss += loss.item()
            total_iou += mean_iou(logits.argmax(dim=1).cpu(), masks.cpu())
            batches += 1
    return total_loss / max(batches, 1), total_iou / max(batches, 1)


def run(
    cfg: Config, device: str = DEVICE, quiet: bool = False, amp: bool = False
) -> dict:
    """Train one configuration and return its history plus the best val score."""
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    torch.manual_seed(cfg.data.seed)

    loaders = build_dataloaders(
        cfg.data.root,
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        strategy=cfg.data.strategy,
        fractions=tuple(cfg.data.fractions),
        seed=cfg.data.seed,
        block_size=cfg.data.block_size,
        oversample_factor=cfg.data.oversample_factor,
        num_workers=cfg.data.num_workers,
        verbose=not quiet,
    )

    model = build_model(cfg.model).to(device)
    if not quiet:
        print(f"model={cfg.model.name} params={model.param_counts} amp={amp and device == 'cuda'}")

    criterion = CombinedLoss(
        class_weights=torch.tensor(cfg.loss.class_weights, dtype=torch.float32, device=device),
        ce_weight=cfg.loss.ce_weight,
        dice_weight=cfg.loss.dice_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.train.scheduler_patience,
        factor=cfg.train.scheduler_factor,
    )

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "lr": []}
    best_iou, best_epoch = 0.0, 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running, started = 0.0, time.time()
        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast_context(device, amp):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / len(loaders["train"])

        val_loss, val_iou = evaluate_split(model, loaders["val"], criterion, device, amp)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        for key, value in zip(history, (train_loss, val_loss, val_iou, lr), strict=True):
            history[key].append(value)

        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            payload = {"config": cfg.to_dict(), "epoch": epoch, "val_iou": val_iou}
            state = (
                model.decoder.state_dict()
                if hasattr(model, "decoder")
                else model.state_dict()
            )
            torch.save({"decoder": state, **payload}, out_dir / "decoder_best.pt")

        if not quiet:
            print(
                f"epoch {epoch:3d}/{cfg.train.epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | mIoU {val_iou:.4f} | lr {lr:.2e} | "
                f"{time.time() - started:.0f}s"
            )

    summary = {"best_val_iou": best_iou, "best_epoch": best_epoch, "history": history}
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    plot_history(history, str(out_dir / "training_curves.png"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model", default=None, help="sam2 | convnext_frozen | unet")
    parser.add_argument("--strategy", default=None, help="random | temporal_block | track_holdout")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--amp", action="store_true", help="fp16 mixed precision (CUDA only, ignored elsewhere)"
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    data_over = {
        k: v
        for k, v in (
            ("root", args.data_root),
            ("strategy", args.strategy),
            ("seed", args.seed),
            ("batch_size", args.batch_size),
            ("num_workers", args.num_workers),
        )
        if v is not None
    }
    model_over = {"name": args.model} if args.model else {}
    train_over = {k: v for k, v in (("epochs", args.epochs), ("out_dir", args.out_dir))
                  if v is not None}
    cfg = cfg.with_overrides(data=data_over, model=model_over, train=train_over)

    summary = run(cfg, device=args.device, amp=args.amp)
    print(
        f"best validation mIoU {summary['best_val_iou']:.4f} "
        f"at epoch {summary['best_epoch']} -> {cfg.train.out_dir}/decoder_best.pt"
    )


if __name__ == "__main__":
    main()
