# Cloud Run image for the Gradio demo.
#
# Measured peak RSS for CPU inference is ~1.03 GB, so this needs at least 2 GB
# of memory -- which is why the 256-512 MB free tiers of Render, Fly and Koyeb
# cannot host it, and Cloud Run can.
#
# Both the encoder and the decoder are baked in at build time. Downloading them
# on first request would add 30+ seconds to an already slow cold start, and
# Cloud Run containers have no persistent disk to cache them between instances.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

# CPU-only wheels: the CUDA build is several GB and would wreck cold starts.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install "transformers>=4.56" numpy pillow pyyaml "gradio>=4.44"

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY app ./app
COPY docs/assets/examples ./docs/assets/examples
RUN pip install --no-deps -e .

# Bake the frozen encoder into the image (HF_HUB_OFFLINE above enforces that no
# network call happens at runtime, so a missing cache fails the build, loudly,
# instead of failing the first user request, silently).
RUN HF_HUB_OFFLINE=0 python -c "\
from transformers import Sam2Model; \
Sam2Model.from_pretrained('facebook/sam2.1-hiera-tiny'); \
print('encoder cached')"

ARG DECODER_URL=https://github.com/PabloAndresE/sam2-supertuxkart-segmentation/releases/download/v0.1.0/decoder_best.pt
RUN mkdir -p runs/baseline \
 && curl -fsSL -o runs/baseline/decoder_best.pt "$DECODER_URL" \
 && python -c "\
import torch; s = torch.load('runs/baseline/decoder_best.pt', map_location='cpu', weights_only=True); \
d = s.get('decoder', s); \
print('decoder tensors:', len(d), 'params:', sum(v.numel() for v in d.values()))"

# Cloud Run injects PORT; default keeps `docker run -p 7860:7860` working locally.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "GRADIO_SERVER_PORT=$PORT python app/app.py"]
