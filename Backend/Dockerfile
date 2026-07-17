# syntax=docker/dockerfile:1.7
ARG VARIANT=cpu

FROM python:3.11-slim AS base-cpu
FROM python:3.11-slim AS base-rocm
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base-gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python3.11 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

FROM base-${VARIANT} AS final
ARG VARIANT=cpu
ARG ROLE=worker

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

WORKDIR /app
COPY requirements-api.txt requirements-worker.txt ./

RUN if [ "${ROLE}" = "api" ]; then \
      python3 -m pip install -r requirements-api.txt; \
    else \
      if [ "${VARIANT}" = "gpu" ]; then TORCH_INDEX=https://download.pytorch.org/whl/cu126; \
      elif [ "${VARIANT}" = "rocm" ]; then TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4; \
      else TORCH_INDEX=https://download.pytorch.org/whl/cpu; fi && \
      python3 -m pip install torch torchaudio --index-url "${TORCH_INDEX}" && \
      python3 -m pip install -r requirements-worker.txt; \
    fi

COPY app ./app
RUN mkdir -p /app/shared-storage /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
