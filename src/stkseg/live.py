"""Run the segmenter over a frame sequence and render a demo clip.

Three sources, in order of how little they need from you:

``--frames``
    A contiguous run of dataset frames. Needs nothing beyond the repo, and the
    frames are real consecutive video, so the output shows what the model does
    over time rather than on one lucky still.
``--video``
    Any recording (gameplay capture, phone video). Decoded through ffmpeg.
``--capture``
    A live screen region, for segmenting the real game as you play. Needs
    ``mss`` and macOS screen-recording permission.

Every mode reports **temporal flicker**: the share of pixels whose predicted
class changed from the previous frame. On near-identical consecutive frames any
flicker is the model contradicting itself, which is the cost of treating video
as independent images.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stkseg.config import Config
from stkseg.constants import CLASS_NAMES, COLORS, DEVICE
from stkseg.data import FRAME_RE
from stkseg.predict import load_model, segment

PANEL_GAP = 8
HUD_HEIGHT = 28
LEGEND_HEIGHT = 46
SWATCH = 11


def ground_truth_masks(root: Path, track: str, start: int, count: int, size) -> list[np.ndarray]:
    """The engine's masks for the same range, resized exactly as the model's input."""
    masks = []
    for index in range(start, start + count):
        path = root / "masks" / f"{track}_mask_combined_{index:04d}.png"
        if not path.exists():
            continue
        masks.append(
            np.array(Image.open(path).convert("L").resize(tuple(size), Image.NEAREST))
        )
    return masks


def frame_to_frame_change(masks: list[np.ndarray]) -> float:
    """Mean share of pixels whose class differs from the previous frame."""
    if len(masks) < 2:
        return 0.0
    return float(np.mean([(a != b).mean() for a, b in zip(masks[:-1], masks[1:], strict=True)]))


def frames_from_dataset(root: Path, track: str, start: int, count: int) -> Iterator[Image.Image]:
    """Consecutive frames of one track, in race order."""
    paths = []
    for path in sorted((root / "images").glob("*.png")):
        matched = FRAME_RE.match(path.stem)
        if matched and matched["track"] == track:
            paths.append((int(matched["index"]), path))
    if not paths:
        raise SystemExit(f"no frames for track {track!r} under {root}")

    paths.sort()
    selected = [p for idx, p in paths if start <= idx < start + count]
    if not selected:
        raise SystemExit(f"no frames in range {start}..{start + count} for {track!r}")
    for path in selected:
        yield Image.open(path).convert("RGB")


def frames_from_video(path: Path, count: int, stride: int = 1) -> Iterator[Image.Image]:
    """Decode a video through ffmpeg without extra Python dependencies."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; install it or use --frames")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(",")[:2])

    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"select=not(mod(n\\,{stride}))", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = width * height * 3
    try:
        for _ in range(count):
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) < frame_bytes:
                break
            yield Image.frombytes("RGB", (width, height), buffer)
    finally:
        process.kill()


def frames_from_capture(region: tuple[int, int, int, int], count: int) -> Iterator[Image.Image]:
    """Grab a screen region repeatedly. Requires mss and OS screen permission."""
    try:
        import mss
    except ImportError as exc:
        raise SystemExit("pip install mss to use --capture") from exc

    left, top, width, height = region
    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        for _ in range(count):
            shot = sct.grab(monitor)
            yield Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def _draw_legend(draw: ImageDraw.ImageDraw, mask: np.ndarray, top: int, width: int) -> None:
    """One swatch per class with its share of this frame.

    Without this the demo is unreadable: `Background` is black in the palette,
    so tinting it only darkens the frame and looks identical to "nothing
    detected", and `Nitro`/`Bomb` occupy a few hundredths of a percent of the
    pixels -- present, but far too small to see.
    """
    total = mask.size
    x, y = 6, top
    for index, name in enumerate(CLASS_NAMES):
        share = 100 * float((mask == index).sum()) / total
        active = share > 0
        colour = tuple(int(v) for v in COLORS[index])
        if not active:
            colour = tuple(max(28, c // 4) for c in colour)

        draw.rectangle([x, y + 2, x + SWATCH, y + 2 + SWATCH], fill=colour,
                       outline=(90, 90, 90))
        text = f"{name} {share:.2f}%" if active else f"{name} -"
        draw.text((x + SWATCH + 5, y), text,
                  fill=(235, 235, 235) if active else (110, 110, 110))

        x += SWATCH + 10 + 8 * len(text)
        if x > width - 90:
            x, y = 6, y + 16


def colourise(rgb: np.ndarray, mask: np.ndarray, alpha: float, tint_background: bool):
    """Blend the class colours over the frame.

    ``Background`` is left untouched by default: its palette entry is black, so
    blending it darkens the image instead of marking it, which reads as a
    missed region. Untinted pixels are therefore "predicted background", and
    every tinted pixel is a non-background prediction.
    """
    blended = rgb.astype(np.float32).copy()
    coloured = COLORS[mask].astype(np.float32)
    region = np.ones(mask.shape, dtype=bool) if tint_background else (mask != 0)
    blended[region] = rgb[region] * (1 - alpha) + coloured[region] * alpha
    return blended.astype(np.uint8)


def compose(
    rgb: np.ndarray,
    mask: np.ndarray,
    hud: str,
    alpha: float,
    truth: np.ndarray | None = None,
    tint_background: bool = False,
) -> Image.Image:
    """Input, optional ground truth, and the prediction, over a legend strip."""
    height, width = rgb.shape[:2]
    panels = [(rgb.astype(np.uint8), "input")]
    if truth is not None:
        panels.append((colourise(rgb, truth, alpha, tint_background), "ground truth"))
    panels.append((colourise(rgb, mask, alpha, tint_background), "prediction"))

    total_width = width * len(panels) + PANEL_GAP * (len(panels) - 1)
    canvas = Image.new("RGB", (total_width, height + HUD_HEIGHT + LEGEND_HEIGHT), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    for position, (image, title) in enumerate(panels):
        offset = position * (width + PANEL_GAP)
        canvas.paste(Image.fromarray(image), (offset, 0))
        draw.text((offset + 6, 6), title, fill=(245, 245, 245))

    draw.text((6, height + 7), hud, fill=(235, 235, 235))
    _draw_legend(draw, mask, height + HUD_HEIGHT - 2, total_width)
    return canvas


def run(
    source: Iterator[Image.Image],
    model,
    img_size: tuple[int, int],
    device: str,
    alpha: float,
    label: str,
    truths: list[np.ndarray] | None = None,
    tint_background: bool = False,
) -> tuple[list[Image.Image], dict]:
    panels: list[Image.Image] = []
    previous: np.ndarray | None = None
    latencies: list[float] = []
    flickers: list[float] = []

    for index, frame in enumerate(source):
        started = time.perf_counter()
        mask, rgb = segment(model, frame, img_size, device)
        latencies.append((time.perf_counter() - started) * 1000)

        if previous is not None:
            flickers.append(float((mask != previous).mean()))
        previous = mask

        median_ms = float(np.median(latencies))
        hud = (
            f"{label}   frame {index + 1}   "
            f"{median_ms:.0f} ms ({1000 / median_ms:.1f} fps)"
        )
        if flickers:
            hud += f"   flicker {flickers[-1] * 100:.2f}%"
        truth = truths[index] if truths and index < len(truths) else None
        panels.append(compose(rgb, mask, hud, alpha, truth, tint_background))

    stats = {
        "frames": len(panels),
        "median_ms": float(np.median(latencies)) if latencies else 0.0,
        "fps": 1000 / float(np.median(latencies)) if latencies else 0.0,
        "mean_flicker_pct": 100 * float(np.mean(flickers)) if flickers else 0.0,
        "max_flicker_pct": 100 * float(np.max(flickers)) if flickers else 0.0,
    }
    return panels, stats


def write_gif(panels: list[Image.Image], path: Path, fps: int, scale: float) -> None:
    if scale != 1.0:
        size = (int(panels[0].width * scale), int(panels[0].height * scale))
        panels = [p.resize(size, Image.LANCZOS) for p in panels]
    # Palette quantisation keeps the file small enough to sit in a README.
    quantised = [p.quantize(colors=128, method=Image.MEDIANCUT) for p in panels]
    quantised[0].save(
        path, save_all=True, append_images=quantised[1:],
        duration=int(1000 / fps), loop=0, optimize=True,
    )


def write_mp4(panels: list[Image.Image], path: Path, fps: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")
    width, height = panels[0].size
    # H.264 needs even dimensions.
    width, height = width - width % 2, height - height % 2
    process = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE,
    )
    for panel in panels:
        process.stdin.write(np.asarray(panel.crop((0, 0, width, height))).tobytes())
    process.stdin.close()
    process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", metavar="DATASET_ROOT")
    source.add_argument("--video", metavar="FILE")
    source.add_argument("--capture", action="store_true")

    parser.add_argument("--track", default="abyss", help="--frames: which track")
    parser.add_argument("--start", type=int, default=100, help="--frames: first index")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1, help="--video: keep 1 of N frames")
    parser.add_argument("--region", default="0,0,1280,720", help="--capture: left,top,w,h")

    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/baseline/decoder_best.pt")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--size", type=int, default=None, help="override inference resolution")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument(
        "--with-truth", action="store_true",
        help="--frames: add the engine's mask as a middle panel",
    )
    parser.add_argument(
        "--tint-background", action="store_true",
        help="also tint predicted background (black in the palette, so it just darkens)",
    )
    parser.add_argument("--label", default="SAM 2 frozen + 2.1M decoder")

    parser.add_argument("--gif", default=None)
    parser.add_argument("--mp4", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.8)
    args = parser.parse_args()

    if not args.gif and not args.mp4:
        raise SystemExit("nothing to write: pass --gif and/or --mp4")

    cfg = Config.from_yaml(args.config)
    img_size = (args.size, args.size) if args.size else tuple(cfg.data.img_size)
    model = load_model(args.config, args.checkpoint, args.device)

    if args.frames:
        source_iter = frames_from_dataset(Path(args.frames), args.track, args.start, args.count)
    elif args.video:
        source_iter = frames_from_video(Path(args.video), args.count, args.stride)
    else:
        region = tuple(int(v) for v in args.region.split(","))
        source_iter = frames_from_capture(region, args.count)

    truths = None
    if args.frames and args.with_truth:
        truths = ground_truth_masks(
            Path(args.frames), args.track, args.start, args.count, img_size
        )

    panels, stats = run(
        source_iter, model, img_size, args.device, args.alpha, args.label,
        truths=truths, tint_background=args.tint_background,
    )
    if not panels:
        raise SystemExit("no frames processed")

    # Raw flicker conflates the camera moving with the model contradicting
    # itself. The engine's own masks change between frames too, so the part
    # attributable to the model is the excess over that floor.
    if args.frames:
        truth = ground_truth_masks(Path(args.frames), args.track, args.start, args.count, img_size)
        if len(truth) >= 2:
            scene = 100 * frame_to_frame_change(truth)
            stats["scene_change_pct"] = scene
            stats["excess_flicker_pct"] = stats["mean_flicker_pct"] - scene

    if args.gif:
        write_gif(panels, Path(args.gif), args.fps, args.scale)
        print(f"wrote {args.gif} ({Path(args.gif).stat().st_size / 1e6:.1f} MB)")
    if args.mp4:
        write_mp4(panels, Path(args.mp4), args.fps)
        print(f"wrote {args.mp4} ({Path(args.mp4).stat().st_size / 1e6:.1f} MB)")

    print(
        f"{stats['frames']} frames   {stats['median_ms']:.0f} ms median "
        f"({stats['fps']:.1f} fps)   flicker mean {stats['mean_flicker_pct']:.2f}% "
        f"max {stats['max_flicker_pct']:.2f}%"
    )
    if "scene_change_pct" in stats:
        print(
            f"   ground-truth change {stats['scene_change_pct']:.2f}%  ->  "
            f"excess attributable to the model {stats['excess_flicker_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
