# syntax=docker/dockerfile:1.7
ARG VARIANT=cpu

# ── CPU base ──────────────────────────────────────────────────
FROM python:3.11-slim AS base-cpu

# ── GPU base ──────────────────────────────────────────────────
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python3.11 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# ── Final (branches on VARIANT arg) ───────────────────────────
FROM base-${VARIANT} AS final
ARG VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg: audio decode. libsndfile1: librosa/soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

WORKDIR /app

# Copy requirements first so pip layer is cached across code-only changes.
COPY requirements.txt ./

# CPU: strip the cu126 extra-index line, install from CPU wheel index.
# GPU: install as-is (cu126 line already in requirements.txt).
RUN if [ "${VARIANT}" = "cpu" ]; then \
      tail -n +2 requirements.txt > /tmp/req.txt && \
      python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req.txt; \
    else \
      python3 -m pip install -r requirements.txt; \
    fi

COPY app ./app

RUN mkdir -p /app/uploads /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
