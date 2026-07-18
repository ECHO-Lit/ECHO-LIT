#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "The MPS worker must run natively on macOS." >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
backend_root="${project_root}/Backend"
python_bin="${ECHO_MPS_PYTHON:-${backend_root}/.venv-mps/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "MPS virtual environment not found at ${python_bin}." >&2
  echo "Create it with:" >&2
  echo "  cd ${backend_root}" >&2
  echo "  python3.12 -m venv .venv-mps" >&2
  echo "  .venv-mps/bin/pip install -r requirements-worker.txt" >&2
  exit 1
fi

if ! "${python_bin}" -c 'import torch, sys; sys.exit(0 if torch.backends.mps.is_available() else 1)'; then
  echo "PyTorch MPS is unavailable. Check the PyTorch build, macOS version, and Apple GPU." >&2
  exit 1
fi

export ENVIRONMENT="${ENVIRONMENT:-development}"
export EXECUTION_PROFILE=mps
export ALLOW_PAID_EXECUTION=false
export ML_DEVICE=mps
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export JOB_REDIS_URL="${JOB_REDIS_URL:-redis://localhost:6379/1}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/2}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/3}"
export STORAGE_BACKEND="${STORAGE_BACKEND:-local}"
export STORAGE_LOCAL_ROOT="${STORAGE_LOCAL_ROOT:-${backend_root}/shared-storage}"
export MODEL_REGISTRY_MAX_ENTRIES="${MODEL_REGISTRY_MAX_ENTRIES:-1}"
export MODEL_REGISTRY_IDLE_SECONDS="${MODEL_REGISTRY_IDLE_SECONDS:-600}"

cd "${backend_root}"
exec "${python_bin}" -m celery \
  -A app.core.celery_app:celery_app worker \
  --hostname='echo-mps@%h' \
  --loglevel=INFO \
  --queues=cpu,gpu-fast,gpu-large \
  --pool=solo \
  --concurrency=1 \
  --prefetch-multiplier=1
