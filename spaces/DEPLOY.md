# Deploying the Space

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
