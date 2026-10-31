# Deploying the Space

> **Hugging Face now requires a PRO subscription to host Gradio or Docker Spaces**, even
> on free `cpu-basic` hardware; `create_repo` returns `402 Payment Required`. Only static
> (HTML/JS) Spaces remain free. The free click-through demo for this project is therefore
> the [Colab notebook](../notebooks/demo_colab.ipynb). The instructions below still apply
> if you take a PRO plan, or as a template for another host.
>
> A free alternative that would work here: export the encoder and decoder to ONNX and run
> them in a **static** Space with ONNX Runtime Web. That is a sub-project, not a deploy
> step — the fp16 encoder alone is roughly 55 MB to download into the browser.

The Space holds no weights: it clones this repository and downloads the decoder from the
GitHub release on first boot. **The release must be publicly reachable**, so either make
the repository public or move `decoder_best.pt` to a public location first.

```bash
pip install huggingface_hub
huggingface-cli login

# create it once (choose your own name)
huggingface-cli repo create supertuxkart-segmentation --type space --space_sdk gradio

git clone https://huggingface.co/spaces/<your-user>/supertuxkart-segmentation hf-space
cp spaces/app.py spaces/requirements.txt spaces/README.md hf-space/
cd hf-space && git add -A && git commit -m "SuperTuxKart segmentation demo" && git push
```

Free CPU hardware runs SAM 2 Hiera-Tiny at roughly 1-2 s per 512px image, which is fine
for a click-through demo. The first boot takes a few minutes while it downloads torch
and the encoder.

Once it is live, add the badge to the main README:

```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/<your-user>/supertuxkart-segmentation)
```
