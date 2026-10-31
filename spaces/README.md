---
title: SuperTuxKart Segmentation (frozen SAM 2)
emoji: 🏎️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# SuperTuxKart semantic segmentation

Meta's SAM 2 Hiera-Tiny encoder is frozen; a 2.1M-parameter decoder (7% of the network)
maps its features onto 7 game classes. Upload a frame and it returns the segmentation.

Validation mIoU 0.8897 over six classes; held-out test 0.7731. Full protocol, ablations
and the hypotheses that did not survive:
[github.com/PabloAndresE/sam2-supertuxkart-segmentation](https://github.com/PabloAndresE/sam2-supertuxkart-segmentation)

Runs on free CPU hardware at roughly 1-2 s per image.
