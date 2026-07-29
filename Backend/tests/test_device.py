from unittest.mock import patch

import pytest
import torch

from app.core.device import InferenceRuntime, detect_inference_runtime


def test_auto_prefers_rocm_over_cpu():
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="AMD Radeon"),
        patch.object(torch.version, "hip", "6.4"),
    ):
        runtime = detect_inference_runtime("auto")

    assert runtime.device == torch.device("cuda:0")
    assert runtime.backend == "rocm"
    assert runtime.name == "AMD Radeon"


def test_generic_cuda_override_accepts_rocm_device():
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "device_count", return_value=1),
        patch.object(torch.cuda, "get_device_name", return_value="AMD Radeon"),
        patch.object(torch.version, "hip", "6.4"),
    ):
        runtime = detect_inference_runtime("cuda")

    assert runtime.device == torch.device("cuda:0")
    assert runtime.backend == "rocm"


def test_auto_detects_nvidia_cuda():
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA RTX"),
        patch.object(torch.version, "hip", None),
        patch.object(torch.version, "cuda", "12.6"),
    ):
        runtime = detect_inference_runtime("auto")

    assert runtime.device == torch.device("cuda:0")
    assert runtime.backend == "cuda"
    assert runtime.runtime_version == "12.6"


def test_auto_uses_mps_when_cuda_is_unavailable():
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch("app.core.device._mps_available", return_value=True),
        patch(
            "app.core.device._mps_runtime",
            return_value=InferenceRuntime(torch.device("mps"), "mps", "Apple GPU"),
        ),
    ):
        runtime = detect_inference_runtime("auto")

    assert runtime.device == torch.device("mps")
    assert runtime.backend == "mps"


def test_unavailable_explicit_device_falls_back_to_cpu():
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch("app.core.device._mps_available", return_value=False),
    ):
        runtime = detect_inference_runtime("nvidia")

    assert runtime.device == torch.device("cpu")
    assert runtime.backend == "cpu"


def test_invalid_device_fails_fast():
    with pytest.raises(ValueError, match="ML_DEVICE"):
        detect_inference_runtime("tpu")
