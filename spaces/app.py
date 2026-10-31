"""Hugging Face Space entry point.

Self-contained: it pulls the stkseg package and the decoder checkpoint from the
GitHub release at startup, so the Space holds no model weights of its own.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/PabloAndresE/sam2-supertuxkart-segmentation"
CHECKPOINT_URL = f"{REPO}/releases/download/v0.1.0/decoder_best.pt"

SOURCE = HERE / "stkseg_src"
if not SOURCE.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", f"{REPO}.git", str(SOURCE)], check=True
    )
sys.path.insert(0, str(SOURCE / "src"))

CHECKPOINT = HERE / "decoder_best.pt"
if not CHECKPOINT.exists():
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)  # noqa: S310

CONFIG = SOURCE / "configs" / "default.yaml"
os.environ.setdefault("STKSEG_CONFIG", str(CONFIG))
os.environ.setdefault("STKSEG_CHECKPOINT", str(CHECKPOINT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stkseg.config import Config  # noqa: E402
from stkseg.constants import CLASS_NAMES, COLORS  # noqa: E402
from stkseg.predict import load_model, segment  # noqa: E402
from stkseg.viz import mask_to_color  # noqa: E402

DEVICE = "cpu"  # free Spaces have no GPU
CFG = Config.from_yaml(str(CONFIG))
MODEL = load_model(str(CONFIG), str(CHECKPOINT), DEVICE)


def run(image: Image.Image, alpha: float):
    if image is None:
        raise gr.Error("upload a frame first")
    mask, rgb = segment(MODEL, image, CFG.data.img_size, DEVICE)
    coloured = mask_to_color(mask)
    # Background is black in the palette, so tinting it would only darken the
    # frame and read as a missed region. Untinted pixels are background.
    blended = rgb.copy()
    region = mask != 0
    blended[region] = (rgb[region] * (1 - alpha) + coloured[region] * alpha).astype(np.uint8)

    total = mask.size
    shares = {
        CLASS_NAMES[c]: round(float((mask == c).sum()) / total, 4)
        for c in np.unique(mask)
    }
    return coloured, blended, shares


LEGEND = " &nbsp; ".join(
    f"<span style='background:rgb{tuple(int(v) for v in COLORS[i])};color:white;"
    f"padding:2px 8px;border-radius:4px'>{CLASS_NAMES[i]}</span>"
    for i in range(len(CLASS_NAMES))
)

with gr.Blocks(title="SuperTuxKart segmentation") as demo:
    gr.Markdown(
        "# SuperTuxKart semantic segmentation\n"
        "Frozen SAM 2 Hiera-Tiny encoder + a 2.1M-parameter decoder (7% trainable). "
        "Validation mIoU **0.8897**, held-out test **0.7731**.\n\n"
        "Untinted pixels are predicted *Background*."
    )
    gr.HTML(LEGEND)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="frame")
            alpha = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="overlay opacity")
            btn = gr.Button("Segment", variant="primary")
        with gr.Column():
            out_mask = gr.Image(label="segmentation")
            out_overlay = gr.Image(label="overlay")
            out_stats = gr.Label(label="pixel share per class")

    btn.click(run, inputs=[inp, alpha], outputs=[out_mask, out_overlay, out_stats])

    examples = sorted((SOURCE / "docs" / "assets" / "examples").glob("*.png"))
    if examples:
        gr.Examples([str(p) for p in examples], inputs=inp)

demo.launch()
