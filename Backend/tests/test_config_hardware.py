"""Configuration -- the accelerator matrix and the worker's resource bounds.

Test Plan Section 3.1.8.  See tests/plans/3.1.8-configuration-testing.md.

Module CD.  ECHO runs on NVIDIA CUDA, AMD ROCm, Apple MPS or plain CPU, chosen
by `ML_DEVICE`.  Every hardware configuration is produced by patching the
torch probes, as `test_device.py` does, since no single machine has them all.
`INFERENCE_RUNTIME` is fixed when `app.core.device` is imported, so cases
call `detect_inference_runtime` directly or patch the module globals.

SRS 2.4: "GPU acceleration (NVIDIA CUDA, AMD ROCm, or Apple MPS) is assumed
for heavy inference. The system degrades gracefully to CPU where no
accelerator is present."
SRS RE-3: "Where no GPU is available, the system shall fall back to CPU
execution rather than failing."
SRS PE-3: "The model registry shall evict idle model variants to cap memory."
"""
from __future__ import annotations

import sys
import types
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from app.core import device as device_module
from app.core.device import InferenceRuntime, detect_inference_runtime, inference_dtype
from app.core.model_catalog import ModelKind
from app.core.settings import settings
from app.schemas.jobs import RuntimeModelSpec
from tests import _config as cfg
from tests._fixtures import wav_bytes

pytestmark = [pytest.mark.configuration, pytest.mark.important]


@pytest.fixture(autouse=True)
def _clean_settings_env(monkeypatch):
    cfg.clear_settings_env(monkeypatch)


def _host(*, cuda: bool = False, hip: str | None = None, count: int = 1, mps: bool = False) -> ExitStack:
    """A hardware configuration, as the torch probes report it."""
    stack = ExitStack()
    stack.enter_context(patch.object(torch.cuda, "is_available", return_value=cuda))
    stack.enter_context(patch.object(torch.cuda, "device_count", return_value=count if cuda else 0))
    stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="Test GPU"))
    stack.enter_context(patch.object(torch.version, "hip", hip))
    stack.enter_context(patch.object(torch.version, "cuda", None if hip else "12.6"))
    stack.enter_context(patch("app.core.device._mps_available", return_value=mps))
    stack.enter_context(
        patch("app.core.device._mps_runtime", return_value=InferenceRuntime(torch.device("mps"), "mps", "Apple GPU"))
    )
    return stack


class TestDeviceMatrix:
    @pytest.mark.parametrize(
        ("preference", "host", "device", "backend"),
        [
            ("cpu", {"cuda": True}, "cpu", "cpu"),
            ("gpu", {"cuda": True}, "cuda:0", "cuda"),
            ("accelerator", {"mps": True}, "mps", "mps"),
            (" CUDA ", {"cuda": True}, "cuda:0", "cuda"),
            ("cuda:1", {"cuda": True, "count": 2}, "cuda:1", "cuda"),
            ("AMD", {"cuda": True, "hip": "6.4"}, "cuda:0", "rocm"),
            ("nvidia", {"cuda": True}, "cuda:0", "cuda"),
        ],
    )
    def test_preferences_resolve_as_documented(self, preference, host, device, backend):
        """CF-70: README's ML_DEVICE grammar, host by host."""
        with _host(**host):
            runtime = detect_inference_runtime(preference)
        assert runtime.device == torch.device(device)
        assert runtime.backend == backend

    @pytest.mark.parametrize(
        ("preference", "host"),
        [
            ("rocm", {"cuda": True}),
            ("nvidia", {"cuda": True, "hip": "6.4"}),
            ("cuda:3", {"cuda": True, "count": 2}),
            ("mps", {}),
            ("auto", {}),
        ],
    )
    def test_a_mismatched_or_absent_accelerator_falls_back_to_cpu(self, preference, host):
        """CF-71: RE-3 -- a worker without the requested accelerator still works."""
        with _host(**host):
            runtime = detect_inference_runtime(preference)
        assert runtime.device == torch.device("cpu")
        assert runtime.backend == "cpu"

    def test_the_configured_setting_applies_without_a_preference(self, monkeypatch):
        """CF-72: with no explicit preference, `settings.ML_DEVICE` decides."""
        monkeypatch.setattr(settings, "ML_DEVICE", "cpu")
        with _host(cuda=True):
            assert detect_inference_runtime().backend == "cpu"

    @pytest.mark.parametrize("preference", ["cuda:abc", "cuda:-1", "cuda:"])
    def test_a_malformed_device_is_refused_on_every_host(self, monkeypatch, preference):
        """CF-73: guards BUG-73.

        The index was only parsed when a GPU was present, so a typo passed
        every test on a CPU machine and crashed the GPU worker at import.
        """
        with _host(cuda=False), pytest.raises(ValueError, match="ML_DEVICE"):
            detect_inference_runtime(preference)
        with pytest.raises(ValidationError, match="ML_DEVICE"):
            cfg.load_settings(monkeypatch, {"ML_DEVICE": preference})


class TestPrecision:
    @pytest.mark.parametrize(
        ("backend", "device", "expected"),
        [("cpu", "cpu", torch.float32), ("cuda", "cuda:0", torch.float16), ("rocm", "cuda:0", torch.float16), ("mps", "mps", torch.float32)],
    )
    def test_half_precision_is_used_only_on_cuda_and_rocm(self, monkeypatch, backend, device, expected):
        """CF-74: fp16 only where it is mature for these models."""
        monkeypatch.setattr(device_module, "INFERENCE_RUNTIME", InferenceRuntime(torch.device(device), backend, "x"))
        assert inference_dtype(allow_half=True) == expected
        assert inference_dtype(allow_half=False) == torch.float32


class _Placed:
    """A model stand-in that records where it was put."""

    def __init__(self):
        self.devices: list[str] = []

    def to(self, target):
        self.devices.append(str(target))
        return self

    def eval(self):
        return self

    def tie_weights(self):
        pass


def _fake_transformers(model: _Placed) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model

    for name in ("AutoModelForAudioClassification", "AutoModelForCTC", "AutoModelForSpeechSeq2Seq", "AutoProcessor"):
        setattr(module, name, Loader)
    return module


def _custom_adapter():
    from app.worker.model_adapters import GenericHuggingFaceAdapter

    spec = RuntimeModelSpec(
        hf_repo="org/custom-asr", revision="abc123", kind=ModelKind.SEQ2SEQ_ASR, capabilities=["prediction", "attention"]
    )
    return GenericHuggingFaceAdapter("custom:abc", spec)


class TestAttentionPlacement:
    def test_custom_model_attention_honours_attention_force_cpu(self, monkeypatch):
        """CF-75: guards BUG-68.

        ATTENTION_FORCE_CPU exists because eager attention on the GPU reaches
        a Triton kernel that segfaults the worker (settings.py).  Built-in
        Whisper honoured it; a custom model's eager-attention variant was
        loaded straight onto the accelerator.
        """
        model = _Placed()
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(model))
        monkeypatch.setattr(device_module, "INFERENCE_DEVICE", torch.device("cuda:0"))
        monkeypatch.setattr(settings, "ATTENTION_FORCE_CPU", True)

        _custom_adapter().load_resource("eager-attention")
        assert model.devices == ["cpu"]

    @pytest.mark.parametrize(("variant", "force_cpu"), [("generic", True), ("eager-attention", False)])
    def test_other_custom_model_variants_stay_on_the_accelerator(self, monkeypatch, variant, force_cpu):
        """CF-76: the flag moves attention only, and only when it is on."""
        model = _Placed()
        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(model))
        monkeypatch.setattr(device_module, "INFERENCE_DEVICE", torch.device("cuda:0"))
        monkeypatch.setattr(settings, "ATTENTION_FORCE_CPU", force_cpu)

        _custom_adapter().load_resource(variant)
        assert model.devices == ["cuda:0"]

    @pytest.mark.parametrize(("force_cpu", "expected"), [(True, "cpu"), (False, "cuda:0")])
    def test_builtin_whisper_attention_honours_the_flag(self, monkeypatch, force_cpu, expected):
        """CF-77: the built-in path CF-75 brings custom models in line with."""
        from app.services import model_loader_service as loader

        model = _Placed()

        class Loader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return model

        monkeypatch.setattr(loader, "WhisperProcessor", Loader)
        monkeypatch.setattr(loader, "WhisperForConditionalGeneration", Loader)
        monkeypatch.setattr(loader, "INFERENCE_DEVICE", torch.device("cuda:0"))
        monkeypatch.setattr(loader, "_whisper_attention_processor_base", None)
        monkeypatch.setattr(loader, "_whisper_attention_model_base", None)
        monkeypatch.setattr(settings, "ATTENTION_FORCE_CPU", force_cpu)

        loader.get_whisper_attention_models("openai/whisper-base")
        assert model.devices == [expected]

    def test_custom_model_inputs_follow_the_model_device(self, tmp_path):
        """CF-78: why pinning the model is enough -- inputs go where the model is."""
        audio = tmp_path / "tone.wav"
        audio.write_bytes(wav_bytes(seconds=0.25))
        seen: list[str] = []

        class Processor:
            feature_extractor = types.SimpleNamespace(sampling_rate=16000)

            def __call__(self, waveform, sampling_rate, return_tensors):
                return {"input_features": torch.zeros(1, 80, 10)}

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["hello"]

        class Model:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def generate(self, **inputs):
                seen.extend(str(value.device) for value in inputs.values())
                return torch.zeros(1, 2, dtype=torch.long)

        result = _custom_adapter().execute("prediction", str(audio), {}, resource=(Processor(), Model()))
        assert result == {"text": "hello"}
        assert seen == ["cpu"]


class TestWorkerTunables:
    @pytest.mark.parametrize(
        "env",
        [
            {"MODEL_REGISTRY_MAX_ENTRIES": "-1"},
            {"MODEL_REGISTRY_MAX_ENTRIES": "0"},
            {"MODEL_REGISTRY_MAX_ENTRIES": "abc"},
            {"MODEL_REGISTRY_IDLE_SECONDS": "0"},
            {"MAX_SALIENCY_SECONDS": "-3"},
        ],
        ids=lambda env: ",".join(f"{k}={v}" for k, v in env.items()),
    )
    def test_registry_and_saliency_bounds_are_validated_at_startup(self, monkeypatch, env):
        """CF-79: guards BUG-72.

        These were read with `os.getenv`.  MODEL_REGISTRY_MAX_ENTRIES=-1
        emptied the registry and then called `popitem` on it, failing every
        model task with a KeyError; `abc` failed the first one.
        """
        with pytest.raises(ValidationError):
            cfg.load_settings(monkeypatch, env)

    def test_the_registry_takes_its_bounds_from_settings(self, monkeypatch):
        """CF-80: guards BUG-72 -- PE-3's registry bound is a validated setting."""
        from app.worker.model_registry import ModelRegistry

        monkeypatch.setattr(settings, "MODEL_REGISTRY_MAX_ENTRIES", 5)
        monkeypatch.setattr(settings, "MODEL_REGISTRY_IDLE_SECONDS", 60)
        registry = ModelRegistry()
        assert (registry.max_entries, registry.idle_seconds) == (5, 60)
