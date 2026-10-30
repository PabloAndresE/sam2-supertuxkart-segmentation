"""Gradio demo: upload a SuperTuxKart frame, get its segmentation map."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stkseg.config import Config  # noqa: E402
from stkseg.constants import CLASS_NAMES, COLORS, DEVICE  # noqa: E402
from stkseg.predict import load_model, segment  # noqa: E402
from stkseg.viz import mask_to_color  # noqa: E402

CONFIG_PATH = os.environ.get("STKSEG_CONFIG", "configs/default.yaml")
CHECKPOINT = os.environ.get("STKSEG_CHECKPOINT", "runs/baseline/decoder_best.pt")

_cfg = Config.from_yaml(CONFIG_PATH)
_model = None


def get_model():
    """Load lazily so the UI boots even before the checkpoint is mounted."""
    global _model
    if _model is None:
        if not Path(CHECKPOINT).exists():
            raise gr.Error(
                f"checkpoint not found at {CHECKPOINT} - see the README for the download link"
            )
        _model = load_model(CONFIG_PATH, CHECKPOINT, DEVICE)
    return _model


def run(image: Image.Image, alpha: float):
    if image is None:
        raise gr.Error("upload a frame first")
    pred, rgb = segment(get_model(), image, _cfg.data.img_size, DEVICE)
    colored = mask_to_color(pred)
    blended = (rgb * (1 - alpha) + colored * alpha).astype(np.uint8)

    total = pred.size
    present = {
        CLASS_NAMES[c]: round(float((pred == c).sum()) / total, 4)
        for c in np.unique(pred)
    }
    return colored, blended, present


LEGEND = " &nbsp; ".join(
    f"<span style='background:rgb{tuple(int(v) for v in COLORS[i])};"
    f"color:white;padding:2px 8px;border-radius:4px'>{CLASS_NAMES[i]}</span>"
    for i in range(len(CLASS_NAMES))
)

with gr.Blocks(title="SAM 2 - SuperTuxKart segmentation") as demo:
    gr.Markdown(
        "# SuperTuxKart semantic segmentation\n"
        "Frozen SAM 2 Hiera-Tiny encoder + a 2.1M-parameter decoder "
        "(7% of the network is trainable). Validation mIoU **0.8534**."
    )
    gr.HTML(LEGEND)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="frame")
            alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="overlay opacity")
            btn = gr.Button("Segment", variant="primary")
        with gr.Column():
            out_mask = gr.Image(label="segmentation")
            out_overlay = gr.Image(label="overlay")
            out_stats = gr.Label(label="pixel share per class")

    btn.click(run, inputs=[inp, alpha], outputs=[out_mask, out_overlay, out_stats])

    examples_dir = Path("docs/assets/examples")
    if examples_dir.is_dir():
        gr.Examples([str(p) for p in sorted(examples_dir.glob("*.png"))], inputs=inp)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
