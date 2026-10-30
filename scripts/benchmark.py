"""Measure inference latency, throughput and memory for the segmenter.

Reported numbers are what the live demo can actually sustain, so they are taken
after warm-up iterations and include the full encoder + decoder + upsample path.

    python scripts/benchmark.py --device mps --sizes 512 384 256
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from stkseg.constants import DEVICE
from stkseg.model import SAM2Segmenter


def bench_one(model, size: int, device: str, warmup: int, iters: int) -> dict:
    x = torch.randn(1, 3, size, size, device=device)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()

        samples = []
        for _ in range(iters):
            started = time.perf_counter()
            model(x)
            sync()
            samples.append((time.perf_counter() - started) * 1000)

    samples.sort()
    return {
        "resolution": size,
        "ms_median": round(statistics.median(samples), 1),
        "ms_p90": round(samples[int(0.9 * len(samples)) - 1], 1),
        "fps": round(1000 / statistics.median(samples), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 384, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--half", action="store_true", help="run the encoder in float16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = SAM2Segmenter().to(args.device).eval()
    if args.half:
        model = model.half()

    counts = model.param_counts
    print(f"device={args.device} half={args.half} params={counts}")

    results = [bench_one(model, s, args.device, args.warmup, args.iters) for s in args.sizes]
    for row in results:
        print(f"  {row['resolution']:>4}px  {row['ms_median']:>7.1f} ms  "
              f"p90 {row['ms_p90']:>7.1f} ms  {row['fps']:>6.1f} fps")

    if args.out:
        payload = {"device": args.device, "half": args.half, "params": counts, "results": results}
        Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
