FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# CPU wheels keep the image usable on a laptop; swap the index for CUDA builds.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY app ./app
RUN pip install --no-deps -e .

EXPOSE 7860
CMD ["python", "app/app.py"]
