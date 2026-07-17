"""Inference accelerator discovery and device-specific utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import platform
from typing import Any

import torch

from .settings import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceRuntime:
    device: torch.device
    backend: str
    name: str
    runtime_version: str | None = None

    @property
    def accelerated(self) -> bool:
        return self.device.type != "cpu"

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "backend": self.backend,
            "name": self.name,
            "accelerated": self.accelerated,
            "runtime_version": self.runtime_version,
            "torch_version": torch.__version__,
        }


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def _cuda_runtime(index: int = 0) -> InferenceRuntime:
    device = torch.device(f"cuda:{index}")
    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    runtime_version = (
        getattr(torch.version, "hip", None)
        if backend == "rocm"
        else getattr(torch.version, "cuda", None)
    )
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        name = "AMD GPU" if backend == "rocm" else "NVIDIA GPU"
    return InferenceRuntime(device, backend, name, runtime_version)


def _mps_runtime() -> InferenceRuntime:
    name = f"Metal GPU ({platform.machine()})"
    mps_backend = getattr(torch.backends, "mps", None)
    get_name = getattr(mps_backend, "get_name", None)
    if callable(get_name):
        try:
            name = get_name()
        except Exception:
            pass
    return InferenceRuntime(torch.device("mps"), "mps", name)


def detect_inference_runtime(preference: str | None = None) -> InferenceRuntime:
    """Select the requested accelerator, with CPU as the safe fallback.

    ROCm intentionally uses a ``cuda`` torch device because PyTorch exposes AMD
    HIP devices through the torch.cuda API.
    """
    requested = (preference or settings.ML_DEVICE or "auto").strip().lower()
    if requested in {"gpu", "accelerator"}:
        requested = "auto"

    if requested == "cpu":
        return InferenceRuntime(torch.device("cpu"), "cpu", platform.processor() or "CPU")

    cuda_available = torch.cuda.is_available()
    hip_version = getattr(torch.version, "hip", None)

    if requested == "auto":
        if cuda_available:
            return _cuda_runtime()
        if _mps_available():
            return _mps_runtime()
    elif requested == "mps":
        if _mps_available():
            return _mps_runtime()
    elif requested in {"cuda", "nvidia", "rocm", "amd"} or requested.startswith("cuda:"):
        wants_rocm = requested in {"rocm", "amd"}
        wants_nvidia = requested == "nvidia"
        uses_generic_cuda_name = requested == "cuda" or requested.startswith("cuda:")
        runtime_matches = (
            uses_generic_cuda_name
            or (wants_rocm and bool(hip_version))
            or (wants_nvidia and not hip_version)
        )
        if cuda_available and runtime_matches:
            try:
                index = int(requested.split(":", 1)[1]) if requested.startswith("cuda:") else 0
            except ValueError as exc:
                raise ValueError("ML_DEVICE CUDA index must be an integer") from exc
            if index < 0:
                raise ValueError("ML_DEVICE CUDA index must be zero or greater")
            if index < torch.cuda.device_count():
                return _cuda_runtime(index)
            logger.warning("ML_DEVICE=%s requested a missing GPU index", requested)
    else:
        raise ValueError(
            "ML_DEVICE must be one of: auto, cpu, mps, cuda, cuda:<index>, nvidia, rocm, amd"
        )

    logger.warning("ML_DEVICE=%s is unavailable; falling back to CPU", requested)
    return InferenceRuntime(torch.device("cpu"), "cpu", platform.processor() or "CPU")


INFERENCE_RUNTIME = detect_inference_runtime()
INFERENCE_DEVICE = INFERENCE_RUNTIME.device


def inference_dtype(allow_half: bool = False) -> torch.dtype:
    """Use fp16 where it is mature for these models; otherwise retain fp32."""
    if allow_half and INFERENCE_RUNTIME.backend in {"cuda", "rocm"}:
        return torch.float16
    return torch.float32


def clear_accelerator_cache(device: torch.device | str | None = None) -> None:
    selected = torch.device(device or INFERENCE_DEVICE)
    if selected.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif selected.type == "mps" and _mps_available():
        empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def accelerator_memory_allocated_mb(device: torch.device | str | None = None) -> float | None:
    selected = torch.device(device or INFERENCE_DEVICE)
    if selected.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated(selected) / 1024**2
    if selected.type == "mps" and _mps_available():
        current_allocated = getattr(getattr(torch, "mps", None), "current_allocated_memory", None)
        if callable(current_allocated):
            return current_allocated() / 1024**2
    return None


logger.info(
    "Inference runtime selected: backend=%s device=%s name=%s",
    INFERENCE_RUNTIME.backend,
    INFERENCE_RUNTIME.device,
    INFERENCE_RUNTIME.name,
)
