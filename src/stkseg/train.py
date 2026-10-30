"""Train the segmentation decoder on top of a frozen SAM 2 encoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from stkseg.config import Config
from stkseg.constants import DEVICE
from stkseg.data import create_dataloaders
from stkseg.losses import CombinedLoss
from stkseg.metrics import mean_iou
from stkseg.model import SAM2Segmenter
from stkseg.viz import plot_history


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_iou, batches = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            total_iou += mean_iou(logits.argmax(dim=1).cpu(), masks.cpu())
            batches += 1
    return total_loss / max(batches, 1), total_iou / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None, help="override data.root")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.out_dir:
        cfg.train.out_dir = args.out_dir

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    torch.manual_seed(cfg.data.seed)
    device = args.device

    train_loader, val_loader = create_dataloaders(
        cfg.data.root,
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        seed=cfg.data.seed,
        oversample_factor=cfg.data.oversample_factor,
        num_workers=cfg.data.num_workers,
    )

    model = SAM2Segmenter(
        encoder_name=cfg.model.encoder,
        fpn_channels=cfg.model.fpn_channels,
        width=cfg.model.decoder_width,
        dropout=cfg.model.dropout,
    ).to(device)
    print(f"parameters: {model.param_counts}")

    criterion = CombinedLoss(
        class_weights=torch.tensor(cfg.loss.class_weights, dtype=torch.float32, device=device),
        ce_weight=cfg.loss.ce_weight,
        dice_weight=cfg.loss.dice_weight,
    ).to(device)
    # Only decoder parameters reach the optimizer; the encoder never receives grads.
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.train.scheduler_patience,
        factor=cfg.train.scheduler_factor,
    )

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "lr": []}
    best_iou = 0.0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running, started = 0.0, time.time()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        for key, value in zip(
            history, (train_loss, val_loss, val_iou, lr), strict=True
        ):
            history[key].append(value)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"decoder": model.decoder.state_dict(), "config": cfg.to_dict()},
                       out_dir / "decoder_best.pt")

        print(
            f"epoch {epoch:3d}/{cfg.train.epochs} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | mIoU {val_iou:.4f} | lr {lr:.2e} | {time.time()-started:.0f}s"
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, str(out_dir / "training_curves.png"))
    print(f"best validation mIoU: {best_iou:.4f} -> {out_dir/'decoder_best.pt'}")


if __name__ == "__main__":
    main()
