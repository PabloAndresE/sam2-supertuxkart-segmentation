"""Strip the frozen encoder out of a full checkpoint.

The original ``model.th`` stores encoder + decoder (~117 MB), but the encoder is
byte-identical to the public SAM 2 weights. Shipping only the decoder drops the
artifact to ~8 MB, which fits comfortably in a GitHub release.

    python scripts/extract_decoder.py model.th runs/baseline/decoder_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="full state dict (encoder + decoder)")
    parser.add_argument("target", help="output decoder-only checkpoint")
    args = parser.parse_args()

    state = torch.load(args.source, map_location="cpu", weights_only=True)
    decoder = {
        key[len("decoder."):]: value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder:
        raise SystemExit("no keys prefixed with 'decoder.' found in the checkpoint")

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": decoder}, target)

    params = sum(v.numel() for v in decoder.values())
    print(f"{len(decoder)} tensors, {params:,} parameters")
    print(f"{args.source} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
