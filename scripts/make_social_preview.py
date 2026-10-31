"""Render the GitHub social preview card (1280x640).

GitHub shows this image whenever the repository is linked on LinkedIn, X or
Slack. The default card is the generic owner/name banner, so a deliberate one
is the cheapest way to make every share look intentional.

    python scripts/make_social_preview.py --frame 30
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 640
BG = (13, 17, 23)        # GitHub dark canvas
FG = (230, 237, 243)
MUTED = (125, 133, 144)
ACCENT = (88, 166, 255)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def extract_frame(video: Path, index: int, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"select=eq(n\\,{index})", "-vframes", "1", str(out)],
        check=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="docs/assets/demo/abyss_live.mp4")
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--out", default="docs/assets/social_preview.png")
    args = parser.parse_args()

    tmp = Path(args.out).with_suffix(".frame.png")
    extract_frame(Path(args.video), args.frame, tmp)

    card = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(card)

    # Demo strip across the lower half, cropped to the panels and scaled to width.
    strip = Image.open(tmp).convert("RGB")
    strip = strip.crop((0, 0, strip.width, int(strip.height * 0.74)))
    scale = WIDTH / strip.width
    strip = strip.resize((WIDTH, int(strip.height * scale)), Image.LANCZOS)
    card.paste(strip, (0, HEIGHT - strip.height))

    # Fade the top of the strip into the background so the text sits cleanly.
    for offset in range(70):
        y = HEIGHT - strip.height + offset
        alpha = 1 - offset / 70
        draw.line([(0, y), (WIDTH, y)], fill=tuple(int(c * alpha + 0) for c in BG), width=1)

    draw.text((56, 54), "SuperTuxKart semantic segmentation", font=load_font(44, True), fill=FG)
    draw.text(
        (56, 112),
        "Frozen SAM 2 encoder + a 2.1M-parameter decoder — 7% of the network trains",
        font=load_font(23), fill=MUTED,
    )

    facts = [
        ("0.117", "the selection score overstated generalisation"),
        ("16x", "which track is unseen beats every modelling choice"),
        ("8 MB", "shipped checkpoint, encoder fetched at load time"),
    ]
    x = 56
    for value, caption in facts:
        draw.text((x, 176), value, font=load_font(38, True), fill=ACCENT)
        draw.text((x, 224), caption, font=load_font(17), fill=MUTED)
        x += 400

    card.save(args.out)
    tmp.unlink(missing_ok=True)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1000:.0f} KB, {WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
