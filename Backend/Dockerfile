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
    PIP_DISABLE_PIP_VERSION_CHECK=1

![1785920988055](image/Dockerfile/1785920988055.png)![1785920992363](image/Dockerfile/1785920992363.png)![1785920999888](image/Dockerfile/1785920999888.png)# Installed per role/variant, because ffmpeg's dependency closure is ~230 packages
# and neither it nor gcc is needed everywhere:
#   libsndfile1 - everywhere; soundfile loads it at import time, including in the API.
#   ffmpeg      - workers only; the API never decodes audio.
#   gcc         - accelerated workers only. Triton (shipped with torch) JIT-compiles
#                 CUDA/ROCm kernels on first use and shells out to a host C compiler
#                 to build the launcher stub, and the nvidia/cuda *runtime* base has
#                 no toolchain. Without it any Triton-backed op fails with "Failed to
#                 find C compiler". The CPU variant never reaches that path.
# The cache mounts keep downloaded .debs across builds, so editing anything above
# this line no longer re-fetches the whole closure. They also mean apt's package
# lists never land in the image layer, which is what the old `rm -rf` was for.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 \
      $(if [ "${ROLE}" != "api" ]; then echo ffmpeg; fi) \
      $(if [ "${VARIANT}" != "cpu" ]; then echo gcc; fi)

# Triton checks $CC first, then searches for cc/gcc/clang on PATH.
ENV CC=gcc

RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

WORKDIR /app
COPY requirements-api.txt requirements-worker.txt ./

# Cache mount persists downloaded wheels across builds — only new/changed
# packages hit the network on subsequent requirements file edits.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${ROLE}" = "api" ]; then \
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
