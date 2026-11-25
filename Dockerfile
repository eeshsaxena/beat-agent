# CPU-only image: setup (packages, Chromium, model weights, stand-in pack) happens at build time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 BEAT_AGENT_HOST=0.0.0.0 TORCH_HOME=/root/.cache/torch
WORKDIR /app
# libatomic1: Playwright's bundled Node.js needs it and python:3.12-slim lacks it (found by the CI Docker build)
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libatomic1 make && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv .venv \
    && .venv/bin/pip install torch==2.15.0 torchaudio==2.11.1 --index-url https://download.pytorch.org/whl/cpu \
    && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install --no-deps basic-pitch==0.4.0 \
    && .venv/bin/python -m playwright install --with-deps chromium

COPY . .
RUN .venv/bin/python scripts/setup.py

EXPOSE 8000
CMD [".venv/bin/python", "-m", "beatagent.cli", "demo"]
