"""Registration, validation and inference for user-supplied Hugging Face models.

ECHO supports models by *task and architecture*, never by model name. A model
is admissible when it satisfies the constraints documented in
`SUPPORTED_CONSTRAINTS` below; everything ECHO can then do with it is derived
from its task via `CAPABILITIES_BY_TASK`. No model-specific user code is
involved at any point -- task, input tensor name, sampling rate, label map and
tokenizer are all read off the repo's config/processor.

Validation runs in two phases so that a repo which is obviously unusable is
rejected before any weights are downloaded:

  Phase 1 (`_check_metadata`)  config + processor only, no weights.
  Phase 2 (`_check_runtime`)   loads weights, runs one forward pass on
                               synthetic audio and asserts the model really
                               returns `logits` and honours
                               `output_hidden_states=True`.

A model that passes both is written to the session registry in Redis and
becomes selectable in the UI as `custom:<session_id>:<name>`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.models.auto import modeling_auto as _auto

from app.core.redis import redis
from app.core.settings import settings

logger = logging.getLogger(__name__)


# ── Tasks ────────────────────────────────────────────────────────────
# The three families named in the ECHO custom-model contract. The value is
# the wire format used by the API and the frontend.
TASK_SEQ2SEQ = "speech-seq2seq"
TASK_CTC = "ctc"
TASK_AUDIO_CLASSIFICATION = "audio-classification"

# Auto class used to instantiate each task. Constraint: "loadable using one of
# AutoModelForSpeechSeq2Seq, AutoModelForCTC, AutoModelForAudioClassification".
AUTO_CLASS_BY_TASK = {
    TASK_SEQ2SEQ: AutoModelForSpeechSeq2Seq,
    TASK_CTC: AutoModelForCTC,
    TASK_AUDIO_CLASSIFICATION: AutoModelForAudioClassification,
}

AUTO_CLASS_NAME_BY_TASK = {
    TASK_SEQ2SEQ: "AutoModelForSpeechSeq2Seq",
    TASK_CTC: "AutoModelForCTC",
    TASK_AUDIO_CLASSIFICATION: "AutoModelForAudioClassification",
}

# transformers keeps `model_type -> architecture class name` tables for each
# auto class. Inverting them to `architecture class name -> task` lets us key
# off `config.architectures`, which is unambiguous -- `model_type` alone is
# not, because e.g. "wav2vec2" appears in both the CTC and the audio
# classification table and "whisper" in both seq2seq and classification.
_ARCH_TO_TASK: Dict[str, str] = {}
_MODEL_TYPE_TO_TASKS: Dict[str, List[str]] = {}
for _task, _table in (
    (TASK_SEQ2SEQ, _auto.MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES),
    (TASK_CTC, _auto.MODEL_FOR_CTC_MAPPING_NAMES),
    (TASK_AUDIO_CLASSIFICATION, _auto.MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
):
    for _model_type, _arch in _table.items():
        # A table value is occasionally a tuple of class names.
        for _name in (_arch,) if isinstance(_arch, str) else tuple(_arch):
            _ARCH_TO_TASK[_name] = _task
        _MODEL_TYPE_TO_TASKS.setdefault(_model_type, []).append(_task)


# ── Capabilities ─────────────────────────────────────────────────────
# Which ECHO analyses each task supports. This is the single source of truth
# the frontend uses to enable/disable panels: a CTC model has no text decoder,
# so decoder activations and cross-attention are simply absent from its list
# rather than failing at request time.
CAPABILITIES_BY_TASK: Dict[str, List[str]] = {
    TASK_SEQ2SEQ: [
        "transcription",
        "encoder_activations",
        "decoder_activations",
        "cross_attention",
        "word_projections",
        "embeddings",
        "saliency",
        "activation_interventions",
        "perturbation_analysis",
    ],
    TASK_CTC: [
        "transcription",
        "frame_token_probabilities",
        "encoder_activations",
        "embeddings",
        "saliency",
        "perturbation_analysis",
    ],
    TASK_AUDIO_CLASSIFICATION: [
        "class_probabilities",
        "embeddings",
        "saliency",
        "perturbation_analysis",
        "label_interventions",
    ],
}

# Human-readable labels, surfaced verbatim in the UI capability chips.
CAPABILITY_LABELS = {
    "transcription": "Transcription",
    "encoder_activations": "Encoder activations",
    "decoder_activations": "Decoder activations",
    "cross_attention": "Cross-attention",
    "word_projections": "Internal word projections",
    "frame_token_probabilities": "Frame-level token probabilities",
    "class_probabilities": "Class probabilities",
    "embeddings": "Embeddings",
    "saliency": "Saliency",
    "activation_interventions": "Activation interventions",
    "label_interventions": "Label-level interventions",
    "perturbation_analysis": "Perturbation analysis",
}

TASK_LABELS = {
    TASK_SEQ2SEQ: "Sequence-to-sequence speech recognition",
    TASK_CTC: "CTC speech recognition",
    TASK_AUDIO_CLASSIFICATION: "Audio classification",
}

# The constraint list, echoed back to the client so the validation report is
# self-describing and the UI needs no hardcoded copy of it.
SUPPORTED_CONSTRAINTS: List[Dict[str, str]] = [
    {"id": "auto_class", "text": "Loadable with AutoModelForSpeechSeq2Seq, AutoModelForCTC or AutoModelForAudioClassification"},
    {"id": "processor", "text": "A compatible AutoProcessor, AutoFeatureExtractor or tokenizer is available"},
    {"id": "audio_input", "text": "The processor accepts audio input and specifies a sampling rate"},
    {"id": "pytorch", "text": "Implemented in PyTorch and inherits from PreTrainedModel"},
    {"id": "logits", "text": "Returns predictions through a standard logits output"},
    {"id": "hidden_states", "text": "Supports output_hidden_states=True"},
    {"id": "id2label", "text": "Classification models expose a valid id2label mapping"},
    {"id": "tokenizer", "text": "Speech-recognition models expose a compatible tokenizer"},
    {"id": "config_detectable", "text": "Input and output formats are detectable from the model configuration"},
    {"id": "resource_limits", "text": "Fits within ECHO's storage, memory, parameter-count and inference-time limits"},
]


# ── Validation report primitives ─────────────────────────────────────

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


@dataclass
class Check:
    id: str
    constraint: str
    status: str
    detail: str


class ValidationReport:
    """Accumulates per-constraint results; `ok` iff nothing hard-failed."""

    def __init__(self) -> None:
        self.checks: List[Check] = []
        self._texts = {c["id"]: c["text"] for c in SUPPORTED_CONSTRAINTS}

    def add(self, check_id: str, status: str, detail: str) -> None:
        self.checks.append(
            Check(id=check_id, constraint=self._texts.get(check_id, check_id), status=status, detail=detail)
        )

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


class ModelValidationError(Exception):
    """Raised when a model cannot be admitted. Carries the partial report."""

    def __init__(self, message: str, report: Optional[ValidationReport] = None):
        super().__init__(message)
        self.report = report


@dataclass
class CustomModelSpec:
    """Everything ECHO needs to run a registered model, all config-derived."""

    name: str
    model_id: str
    revision: Optional[str]
    task: str
    architecture: str
    model_type: str
    # Tensor key the feature extractor produces (`input_features` for Whisper,
    # `input_values` for wav2vec2-family). Read from `model_input_names`, so
    # the inference path never has to branch on the model.
    input_name: str
    sampling_rate: int
    num_parameters: int
    num_labels: Optional[int]
    id2label: Optional[Dict[str, str]]
    has_tokenizer: bool
    capabilities: List[str]
    registered_at: str
    probe_seconds: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["task_label"] = TASK_LABELS.get(self.task, self.task)
        d["auto_class"] = AUTO_CLASS_NAME_BY_TASK.get(self.task)
        d["capability_labels"] = [CAPABILITY_LABELS.get(c, c) for c in self.capabilities]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CustomModelSpec":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


# ── Task detection (config only) ─────────────────────────────────────

def detect_task(config) -> Tuple[Optional[str], str]:
    """Return `(task, detail)` for a loaded `AutoConfig`.

    Architecture name wins because it is unique across the three auto-class
    tables. `model_type` is the fallback and is only trusted when it maps to
    exactly one task.
    """
    architectures = list(getattr(config, "architectures", None) or [])
    for arch in architectures:
        task = _ARCH_TO_TASK.get(arch)
        if task:
            return task, f"architecture '{arch}' -> {AUTO_CLASS_NAME_BY_TASK[task]}"

    model_type = getattr(config, "model_type", None)
    candidates = _MODEL_TYPE_TO_TASKS.get(model_type, [])
    if len(candidates) == 1:
        return candidates[0], f"model_type '{model_type}' -> {AUTO_CLASS_NAME_BY_TASK[candidates[0]]}"
    if len(candidates) > 1:
        return None, (
            f"model_type '{model_type}' is ambiguous across "
            f"{', '.join(AUTO_CLASS_NAME_BY_TASK[c] for c in candidates)}; "
            "the repo's config must name a concrete architecture in `architectures`"
        )
    if architectures:
        return None, (
            f"architecture {architectures[0]!r} is not a supported audio task. ECHO supports "
            "sequence-to-sequence ASR, CTC ASR and audio classification."
        )
    return None, "config declares neither `architectures` nor a recognised audio `model_type`"


def _hub_kwargs(revision: Optional[str]) -> Dict[str, Any]:
    return {"revision": revision} if revision else {}


def _is_offline_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "offline" in text or "couldn't connect" in text or "connection error" in text


# ── Phase 1: metadata checks (no weights) ────────────────────────────

def _check_metadata(model_id: str, revision: Optional[str], report: ValidationReport) -> Dict[str, Any]:
    """Config + processor checks. Returns the facts phase 2 needs."""
    try:
        config = AutoConfig.from_pretrained(model_id, **_hub_kwargs(revision))
    except Exception as e:
        if _is_offline_error(e):
            raise ModelValidationError(
                f"Cannot reach the Hugging Face Hub to fetch '{model_id}'. The backend is running "
                "with HF_HUB_OFFLINE=1, so only already-cached repositories can be registered.",
                report,
            ) from e
        raise ModelValidationError(f"Could not load a config for '{model_id}': {e}", report) from e

    task, detail = detect_task(config)
    if task is None:
        report.add("auto_class", FAIL, detail)
        report.add("config_detectable", FAIL, "task could not be derived from the configuration")
        raise ModelValidationError(detail, report)
    report.add("auto_class", PASS, detail)

    architecture = (list(getattr(config, "architectures", None) or [None]))[0] or type(config).__name__

    # ── processor / feature extractor ──
    processor = None
    feature_extractor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, **_hub_kwargs(revision))
        feature_extractor = getattr(processor, "feature_extractor", None) or processor
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception as proc_err:
        # Audio-classification repos frequently ship only a feature extractor;
        # AutoProcessor raises for them, which is not a failure.
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **_hub_kwargs(revision))
        except Exception as fe_err:
            report.add("processor", FAIL, f"neither AutoProcessor nor AutoFeatureExtractor could be loaded ({fe_err})")
            raise ModelValidationError(
                f"'{model_id}' does not ship a usable processor or feature extractor.", report
            ) from fe_err
        logger.info("custom_model: %s has no AutoProcessor (%s); using AutoFeatureExtractor", model_id, proc_err)

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_hub_kwargs(revision))
        except Exception:
            tokenizer = None

    kind = "AutoProcessor" if processor is not None else "AutoFeatureExtractor"
    report.add("processor", PASS, f"{kind} loaded ({type(feature_extractor).__name__})")

    # ── audio input + sampling rate ──
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    input_names = list(getattr(feature_extractor, "model_input_names", None) or [])
    if sampling_rate is None:
        report.add("audio_input", FAIL, "the feature extractor does not declare a `sampling_rate`")
        raise ModelValidationError(
            f"'{model_id}': the processor does not specify the sampling rate it expects.", report
        )
    if not input_names:
        report.add("audio_input", FAIL, "the feature extractor declares no `model_input_names`")
        raise ModelValidationError(
            f"'{model_id}': the processor does not declare which input tensor the model consumes.", report
        )
    input_name = input_names[0]
    if input_name not in ("input_features", "input_values"):
        report.add(
            "audio_input", FAIL,
            f"first model input is '{input_name}', which is not a raw-audio input "
            "(expected 'input_features' or 'input_values')",
        )
        raise ModelValidationError(f"'{model_id}' does not appear to take audio input.", report)
    report.add("audio_input", PASS, f"accepts audio as `{input_name}` at {int(sampling_rate)} Hz")

    # ── config-detectable IO ──
    report.add(
        "config_detectable", PASS,
        f"task, input tensor (`{input_name}`) and sampling rate ({int(sampling_rate)} Hz) all read from the "
        "repository config -- no model-specific code required",
    )

    # ── id2label (classification only) ──
    id2label: Optional[Dict[str, str]] = None
    num_labels: Optional[int] = None
    if task == TASK_AUDIO_CLASSIFICATION:
        raw = getattr(config, "id2label", None)
        if not raw:
            report.add("id2label", FAIL, "classification model has no `id2label` mapping in its config")
            raise ModelValidationError(
                f"'{model_id}' is a classification model but exposes no id2label mapping, so output "
                "indices cannot be turned into readable class labels.",
                report,
            )
        id2label = {str(k): str(v) for k, v in raw.items()}
        # A config that was never given real labels still yields the
        # placeholder LABEL_0/LABEL_1/... map; accept it but say so, because
        # every ECHO class-probability view will be unreadable.
        placeholder = all(v.upper().startswith("LABEL_") for v in id2label.values())
        num_labels = len(id2label)
        if placeholder:
            report.add(
                "id2label", WARN,
                f"{num_labels} labels present but all are placeholders (LABEL_0 ...); "
                "class names will not be meaningful",
            )
        else:
            preview = ", ".join(list(id2label.values())[:5])
            report.add("id2label", PASS, f"{num_labels} labels: {preview}{' ...' if num_labels > 5 else ''}")
    else:
        report.add("id2label", SKIP, "not a classification model")

    # ── tokenizer (ASR only) ──
    has_tokenizer = tokenizer is not None
    if task in (TASK_SEQ2SEQ, TASK_CTC):
        if not has_tokenizer:
            report.add("tokenizer", FAIL, "no tokenizer in the repository; token IDs cannot be decoded to text")
            raise ModelValidationError(
                f"'{model_id}' is a speech-recognition model but ships no tokenizer, so ECHO cannot "
                "convert token IDs into text.",
                report,
            )
        report.add("tokenizer", PASS, f"{type(tokenizer).__name__}, vocab size {tokenizer.vocab_size}")
    else:
        report.add("tokenizer", SKIP, "not a speech-recognition model")

    return {
        "config": config,
        "task": task,
        "architecture": architecture,
        "model_type": getattr(config, "model_type", "") or "",
        "processor": processor,
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "input_name": input_name,
        "sampling_rate": int(sampling_rate),
        "id2label": id2label,
        "num_labels": num_labels,
        "has_tokenizer": has_tokenizer,
    }


# ── Phase 2: runtime checks (weights loaded, one forward pass) ───────

def _check_runtime(model_id: str, revision: Optional[str], facts: Dict[str, Any], report: ValidationReport):
    task = facts["task"]
    auto_cls = AUTO_CLASS_BY_TASK[task]

    load_started = time.perf_counter()
    try:
        model = _load_weights(model_id, revision, auto_cls)
    except Exception as e:
        if _is_offline_error(e):
            raise ModelValidationError(
                f"Weights for '{model_id}' are not in the local cache and the backend is offline "
                "(HF_HUB_OFFLINE=1).",
                report,
            ) from e
        report.add("pytorch", FAIL, f"{AUTO_CLASS_NAME_BY_TASK[task]}.from_pretrained failed: {e}")
        raise ModelValidationError(f"Could not load '{model_id}': {e}", report) from e

    # ── PyTorch + PreTrainedModel ──
    if not isinstance(model, PreTrainedModel):
        report.add("pytorch", FAIL, f"{type(model).__name__} does not inherit from PreTrainedModel")
        raise ModelValidationError(f"'{model_id}' is not a PyTorch PreTrainedModel.", report)
    report.add(
        "pytorch", PASS,
        f"{type(model).__name__} (PreTrainedModel, torch {torch.__version__}), "
        f"loaded in {time.perf_counter() - load_started:.1f}s",
    )

    # ── parameter count / memory ──
    num_parameters = sum(p.numel() for p in model.parameters())
    limit = settings.CUSTOM_MODEL_MAX_PARAMS
    if num_parameters > limit:
        report.add(
            "resource_limits", FAIL,
            f"{num_parameters/1e6:.0f}M parameters exceeds the {limit/1e6:.0f}M limit",
        )
        raise ModelValidationError(
            f"'{model_id}' has {num_parameters/1e6:.0f}M parameters, above ECHO's "
            f"{limit/1e6:.0f}M limit.",
            report,
        )

    # ── one real forward pass on synthetic audio ──
    sr = facts["sampling_rate"]
    # 2 s of low-amplitude noise. Silence is avoided deliberately: several ASR
    # models emit a zero-length generation for an all-zero waveform, which
    # would make the smoke test fail for a perfectly good model.
    rng = np.random.default_rng(0)
    probe_audio = (rng.standard_normal(int(sr * 2.0)) * 1e-3).astype(np.float32)

    probe_started = time.perf_counter()
    try:
        inputs = _extract_features(facts["feature_extractor"], probe_audio, sr, facts["input_name"])
        inputs = _add_decoder_inputs(model, task, inputs)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    except TypeError as e:
        report.add("hidden_states", FAIL, f"forward pass rejected output_hidden_states=True: {e}")
        raise ModelValidationError(
            f"'{model_id}' does not accept output_hidden_states=True, so embedding and activation "
            "analysis are impossible.",
            report,
        ) from e
    except Exception as e:
        report.add("logits", FAIL, f"forward pass raised {type(e).__name__}: {e}")
        raise ModelValidationError(f"'{model_id}' failed a smoke-test forward pass: {e}", report) from e
    probe_seconds = time.perf_counter() - probe_started

    logits = getattr(outputs, "logits", None)
    if logits is None or not isinstance(logits, torch.Tensor):
        report.add("logits", FAIL, "model output has no `logits` tensor")
        raise ModelValidationError(
            f"'{model_id}' does not return a standard `logits` output.", report
        )
    report.add("logits", PASS, f"logits tensor of shape {tuple(logits.shape)}")

    hidden = getattr(outputs, "hidden_states", None) or getattr(outputs, "encoder_hidden_states", None)
    if not hidden:
        report.add(
            "hidden_states", FAIL,
            "output_hidden_states=True was accepted but no hidden states were returned",
        )
        raise ModelValidationError(
            f"'{model_id}' does not expose hidden states, so embedding and activation analysis "
            "are impossible.",
            report,
        )
    report.add(
        "hidden_states", PASS,
        f"{len(hidden)} hidden-state layers, last of shape {tuple(hidden[-1].shape)}",
    )

    # ── remaining resource limits ──
    # Rough resident-memory estimate from parameter dtype sizes; the on-disk
    # figure is the same order and is what the storage limit is written against.
    storage_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    storage_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    max_storage = settings.CUSTOM_MODEL_MAX_STORAGE_BYTES
    if storage_bytes > max_storage:
        report.add(
            "resource_limits", FAIL,
            f"weights occupy {storage_bytes/1024**3:.2f} GiB, above the {max_storage/1024**3:.2f} GiB limit",
        )
        raise ModelValidationError(
            f"'{model_id}' needs {storage_bytes/1024**3:.2f} GiB, above ECHO's "
            f"{max_storage/1024**3:.2f} GiB limit.",
            report,
        )

    probe_limit = settings.CUSTOM_MODEL_PROBE_TIMEOUT_SECONDS
    if probe_seconds > probe_limit:
        report.add(
            "resource_limits", FAIL,
            f"a 2 s clip took {probe_seconds:.1f}s to process, above the {probe_limit:.0f}s probe budget",
        )
        raise ModelValidationError(
            f"'{model_id}' is too slow: {probe_seconds:.1f}s for a 2-second clip.", report
        )

    detail = (
        f"{num_parameters/1e6:.0f}M params (limit {limit/1e6:.0f}M), "
        f"{storage_bytes/1024**3:.2f} GiB (limit {max_storage/1024**3:.2f} GiB), "
        f"{probe_seconds:.2f}s for a 2 s clip (per-file limit "
        f"{settings.CUSTOM_MODEL_MAX_INFERENCE_SECONDS:.0f}s)"
    )
    # Extrapolating the 2 s probe, warn when a typical 30 s clip would come
    # close to the per-request ceiling.
    projected = probe_seconds * 15
    if projected > settings.CUSTOM_MODEL_MAX_INFERENCE_SECONDS * 0.5:
        report.add("resource_limits", WARN, detail + f"; a 30 s clip would take roughly {projected:.0f}s")
    else:
        report.add("resource_limits", PASS, detail)

    return model, num_parameters, storage_bytes, probe_seconds


def _load_weights(model_id: str, revision: Optional[str], auto_cls):
    """`from_pretrained` on the real device, with eager attention.

    Eager attention is requested so `output_attentions=True` works later --
    SDPA silently returns `None` attentions, which would break cross-attention
    analysis for seq2seq models. Models that do not implement an eager path
    fall back to their default.
    """
    kwargs = dict(_hub_kwargs(revision))
    try:
        model = auto_cls.from_pretrained(model_id, attn_implementation="eager", **kwargs)
    except (ValueError, TypeError) as e:
        logger.info("custom_model: %s rejected attn_implementation='eager' (%s); using default", model_id, e)
        model = auto_cls.from_pretrained(model_id, **kwargs)

    if hasattr(model, "tie_weights"):
        # Tied LM heads (Whisper's `proj_out`, most seq2seq decoders) are not
        # in the checkpoint and exist only via tying. Skipping this leaves the
        # head randomly initialised and the model emits fluent nonsense with
        # no error.
        model.tie_weights()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


def _extract_features(feature_extractor, audio: np.ndarray, sampling_rate: int, input_name: str) -> Dict[str, torch.Tensor]:
    """Run the repo's own feature extractor and keep only tensor inputs."""
    encoded = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    if input_name not in inputs:
        raise ValueError(
            f"feature extractor produced {sorted(inputs)} but the config declares `{input_name}`"
        )
    return inputs


def _add_decoder_inputs(model, task: str, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Give an encoder-decoder model a one-token decoder prompt.

    A bare `model(input_features=...)` on a seq2seq model has no decoder input,
    and transformers then builds `decoder_inputs_embeds` itself and immediately
    rejects the call for specifying both. Feeding the config's decoder start
    token makes the forward pass well-defined -- one decoder step is all the
    smoke test and any logits/hidden-state inspection need.
    """
    if task != TASK_SEQ2SEQ:
        return inputs
    config = model.config
    start_id = (
        getattr(config, "decoder_start_token_id", None)
        or getattr(config, "bos_token_id", None)
        or getattr(getattr(config, "decoder", None), "bos_token_id", None)
    )
    if start_id is None:
        raise ValueError(
            "encoder-decoder config declares neither `decoder_start_token_id` nor `bos_token_id`, "
            "so a decoder prompt cannot be derived from the configuration"
        )
    batch = next(iter(inputs.values())).shape[0]
    device = next(iter(inputs.values())).device
    inputs = dict(inputs)
    inputs["decoder_input_ids"] = torch.full((batch, 1), int(start_id), dtype=torch.long, device=device)
    return inputs


# ── Public validation entry point ────────────────────────────────────

def validate_model(model_id: str, revision: Optional[str] = None, deep: bool = True) -> Dict[str, Any]:
    """Check `model_id` against every ECHO constraint.

    `deep=False` stops after the config/processor phase, which is fast and
    downloads only a few KB -- used by the UI's "Check compatibility" button.
    `deep=True` additionally downloads weights and runs a smoke test; this is
    what registration requires.
    """
    report = ValidationReport()
    try:
        facts = _check_metadata(model_id, revision, report)
    except ModelValidationError as e:
        return {
            "model_id": model_id,
            "revision": revision,
            "compatible": False,
            "deep": False,
            "error": str(e),
            **report.as_dict(),
        }

    result: Dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "task": facts["task"],
        "task_label": TASK_LABELS[facts["task"]],
        "auto_class": AUTO_CLASS_NAME_BY_TASK[facts["task"]],
        "architecture": facts["architecture"],
        "model_type": facts["model_type"],
        "input_name": facts["input_name"],
        "sampling_rate": facts["sampling_rate"],
        "id2label": facts["id2label"],
        "num_labels": facts["num_labels"],
        "capabilities": CAPABILITIES_BY_TASK[facts["task"]],
        "capability_labels": [CAPABILITY_LABELS[c] for c in CAPABILITIES_BY_TASK[facts["task"]]],
        "deep": deep,
    }

    if not deep:
        for check_id in ("pytorch", "logits", "hidden_states", "resource_limits"):
            report.add(check_id, SKIP, "requires downloading weights; run a full validation to verify")
        result.update(compatible=report.ok, **report.as_dict())
        return result

    try:
        model, num_parameters, storage_bytes, probe_seconds = _check_runtime(model_id, revision, facts, report)
    except ModelValidationError as e:
        result.update(compatible=False, error=str(e), **report.as_dict())
        return result

    result.update(
        compatible=report.ok,
        num_parameters=num_parameters,
        storage_bytes=storage_bytes,
        probe_seconds=round(probe_seconds, 3),
        **report.as_dict(),
    )
    result["_model"] = model  # consumed by register_model; never serialised
    result["_facts"] = facts
    return result


# ── Session registry (Redis) ─────────────────────────────────────────

def k_models(session_id: str) -> str:
    return f"sess:{session_id}:custom_models"


def format_custom_model_name(session_id: str, name: str) -> str:
    return f"custom:{session_id}:{name}"


def is_custom_model(model: str) -> bool:
    return isinstance(model, str) and model.startswith("custom:")


def parse_custom_model_name(model: str) -> Tuple[str, str]:
    """Split `custom:<session_id>:<name>` into its parts."""
    if not is_custom_model(model):
        raise ValueError(f"Not a custom model reference: {model}")
    parts = model.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError(f"Invalid custom model reference: {model}")
    return parts[1], parts[2]


async def register_model(
    session_id: str, name: str, model_id: str, revision: Optional[str] = None
) -> Dict[str, Any]:
    """Validate and, on success, persist `model_id` into the session registry."""
    name = name.strip()
    if not name:
        raise ModelValidationError("A display name is required.")
    if ":" in name or "/" in name:
        raise ModelValidationError("The display name cannot contain ':' or '/'.")

    existing = await list_models(session_id)
    if any(m["name"] == name for m in existing):
        raise ModelValidationError(f"A model named '{name}' is already registered in this session.")
    if len(existing) >= settings.CUSTOM_MODEL_MAX_PER_SESSION:
        raise ModelValidationError(
            f"This session already has {len(existing)} custom models "
            f"(limit {settings.CUSTOM_MODEL_MAX_PER_SESSION}). Delete one first."
        )
    if not settings.CUSTOM_MODEL_ALLOW_DOWNLOAD:
        raise ModelValidationError("Custom model registration is disabled on this deployment.")

    result = validate_model(model_id, revision, deep=True)
    if not result.get("compatible"):
        raise ModelValidationError(
            result.get("error") or f"'{model_id}' does not satisfy ECHO's custom model constraints.",
            _report_from_result(result),
        )

    facts = result.pop("_facts")
    model = result.pop("_model")

    spec = CustomModelSpec(
        name=name,
        model_id=model_id,
        revision=revision,
        task=facts["task"],
        architecture=facts["architecture"],
        model_type=facts["model_type"],
        input_name=facts["input_name"],
        sampling_rate=facts["sampling_rate"],
        num_parameters=result["num_parameters"],
        num_labels=facts["num_labels"],
        id2label=facts["id2label"],
        has_tokenizer=facts["has_tokenizer"],
        capabilities=CAPABILITIES_BY_TASK[facts["task"]],
        registered_at=datetime.now(timezone.utc).isoformat(),
        probe_seconds=result["probe_seconds"],
        warnings=[c["detail"] for c in result["checks"] if c["status"] == WARN],
    )

    await redis.hset(k_models(session_id), name, json.dumps(spec.to_dict()))
    await redis.expire(k_models(session_id), settings.SESSION_TTL_SECONDS)

    # The validation load is already warm; seed the cache so the first real
    # inference does not pay for the download and load again.
    _CACHE.put(_cache_key(spec), (model, facts["processor"], facts["feature_extractor"], facts["tokenizer"]))

    logger.info(
        "custom_model: registered '%s' (%s, %s) for session %s", name, model_id, facts["task"], session_id
    )
    return {
        "model": format_custom_model_name(session_id, name),
        "spec": spec.to_dict(),
        "validation": {k: v for k, v in result.items() if not k.startswith("_")},
    }


def _report_from_result(result: Dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    for c in result.get("checks", []):
        report.add(c["id"], c["status"], c["detail"])
    return report


async def list_models(session_id: str) -> List[Dict[str, Any]]:
    raw = await redis.hgetall(k_models(session_id))
    models: List[Dict[str, Any]] = []
    for name, payload in (raw or {}).items():
        try:
            spec = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("custom_model: dropping unreadable registry entry '%s'", name)
            continue
        spec["formatted_name"] = format_custom_model_name(session_id, spec.get("name", name))
        models.append(spec)
    models.sort(key=lambda m: m.get("registered_at", ""))
    return models


async def get_model_spec(session_id: str, name: str) -> Optional[CustomModelSpec]:
    payload = await redis.hget(k_models(session_id), name)
    if not payload:
        return None
    return CustomModelSpec.from_dict(json.loads(payload))


async def delete_model(session_id: str, name: str) -> bool:
    removed = await redis.hdel(k_models(session_id), name)
    if removed:
        _CACHE.evict_matching(lambda key: key[0] == name)
        logger.info("custom_model: deleted '%s' from session %s", name, session_id)
    return bool(removed)


async def resolve_model(model_ref: str) -> CustomModelSpec:
    """Turn a `custom:<sid>:<name>` reference into its spec."""
    session_id, name = parse_custom_model_name(model_ref)
    spec = await get_model_spec(session_id, name)
    if spec is None:
        raise ModelValidationError(
            f"Custom model '{name}' is not registered in this session. It may have expired -- "
            "re-register it from the Custom Models dialog."
        )
    return spec


# ── Loaded-model cache ───────────────────────────────────────────────

class _ModelCache:
    """LRU cache of loaded models, safe to use from the inference threadpool.

    The lock is not just for the dict: concurrent `from_pretrained` calls in
    transformers 5.x race with each other and leave weights stranded on the
    meta device, so loads are serialised too.
    """

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._entries: "OrderedDict[Tuple[str, str, str], Any]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or_load(self, key, loader):
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            value = loader()
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                evicted, _ = self._entries.popitem(last=False)
                logger.info("custom_model: evicted %s from the model cache", evicted[1])
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                evicted, _ = self._entries.popitem(last=False)
                logger.info("custom_model: evicted %s from the model cache", evicted[1])

    def evict_matching(self, predicate) -> None:
        with self._lock:
            for key in [k for k in self._entries if predicate(k)]:
                self._entries.pop(key, None)


_CACHE = _ModelCache(settings.CUSTOM_MODEL_CACHE_SIZE)


def _cache_key(spec: CustomModelSpec) -> Tuple[str, str, str]:
    return (spec.name, spec.model_id, spec.revision or "")


def load_for_inference(spec: CustomModelSpec):
    """Return `(model, processor, feature_extractor, tokenizer)` for `spec`."""

    def loader():
        auto_cls = AUTO_CLASS_BY_TASK[spec.task]
        model = _load_weights(spec.model_id, spec.revision, auto_cls)
        processor = None
        tokenizer = None
        try:
            processor = AutoProcessor.from_pretrained(spec.model_id, **_hub_kwargs(spec.revision))
            feature_extractor = getattr(processor, "feature_extractor", None) or processor
            tokenizer = getattr(processor, "tokenizer", None)
        except Exception:
            feature_extractor = AutoFeatureExtractor.from_pretrained(spec.model_id, **_hub_kwargs(spec.revision))
        if tokenizer is None and spec.has_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(spec.model_id, **_hub_kwargs(spec.revision))
        return model, processor, feature_extractor, tokenizer

    return _CACHE.get_or_load(_cache_key(spec), loader)


# ── Inference ────────────────────────────────────────────────────────
# One entry point per task family. All three read their configuration off
# `spec`, so adding a new architecture within a supported task needs no code.

def _load_audio(spec: CustomModelSpec, audio_file_path: str) -> np.ndarray:
    """Decode to the sampling rate the model's own processor declared."""
    import librosa  # imported lazily; librosa costs ~1 s to import

    audio, _ = librosa.load(audio_file_path, sr=spec.sampling_rate)
    audio = audio.astype(np.float32)
    if audio.size == 0:
        raise ValueError(f"Loaded audio is empty: {audio_file_path}")
    # NaN/inf in the waveform propagate through the encoder and can trigger a
    # CUDA device-side assert that kills the context for the whole process.
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def _decode_ctc(tokenizer, processor, predicted_ids: torch.Tensor) -> str:
    """Collapse CTC frame predictions to text using the repo's tokenizer."""
    # Wav2Vec2-style processors implement the CTC collapse (blank removal +
    # de-duplication) in `batch_decode`; plain tokenizers do not, so fall back
    # to the tokenizer's own CTC-aware decode when the processor lacks it.
    decoder = processor if hasattr(processor, "batch_decode") else tokenizer
    text = decoder.batch_decode(predicted_ids)[0]
    return (text or "").strip()


def run_custom_inference(spec: CustomModelSpec, audio_file_path: str) -> Dict[str, Any]:
    """Run `spec` over one audio file and return a task-shaped result."""
    started = time.perf_counter()
    model, processor, feature_extractor, tokenizer = load_for_inference(spec)
    audio = _load_audio(spec, audio_file_path)
    inputs = _extract_features(feature_extractor, audio, spec.sampling_rate, spec.input_name)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    result: Dict[str, Any] = {
        "model_name": spec.name,
        "model_id": spec.model_id,
        "task": spec.task,
        "task_label": TASK_LABELS[spec.task],
        "capabilities": spec.capabilities,
    }

    if spec.task == TASK_SEQ2SEQ:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=440)
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        result["text"] = (text or "").strip()

    elif spec.task == TASK_CTC:
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        result["text"] = _decode_ctc(tokenizer, processor, predicted_ids)
        result["frame_probabilities"] = _summarise_frame_probabilities(logits, tokenizer, audio, spec)

    elif spec.task == TASK_AUDIO_CLASSIFICATION:
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = spec.id2label or {}
        probabilities = {
            id2label.get(str(i), f"LABEL_{i}"): float(probs[i].item()) for i in range(probs.shape[0])
        }
        top = int(torch.argmax(probs).item())
        result["predicted_label"] = id2label.get(str(top), f"LABEL_{top}")
        result["probabilities"] = probabilities
        result["confidence"] = float(probs[top].item())

    else:  # unreachable: a spec is only persisted with a supported task
        raise ValueError(f"Unsupported task on registered model: {spec.task}")

    elapsed = time.perf_counter() - started
    result["inference_seconds"] = round(elapsed, 3)
    if elapsed > settings.CUSTOM_MODEL_MAX_INFERENCE_SECONDS:
        # Reported rather than raised: the work is already done and the result
        # is valid. The caller surfaces it so a slow model is visible.
        result["limit_exceeded"] = (
            f"inference took {elapsed:.1f}s, above the "
            f"{settings.CUSTOM_MODEL_MAX_INFERENCE_SECONDS:.0f}s per-file limit"
        )
        logger.warning("custom_model: %s took %.1fs on %s", spec.name, elapsed, audio_file_path)
    return result


def _summarise_frame_probabilities(
    logits: torch.Tensor, tokenizer, audio: np.ndarray, spec: CustomModelSpec, max_frames: int = 400
) -> Dict[str, Any]:
    """Per-frame top token + probability for a CTC model.

    Full `[frames, vocab]` probabilities are far too large to ship to the
    browser, so only the argmax token and its probability are kept per frame,
    plus the frame's timestamp derived from the audio/frame ratio.
    """
    probs = torch.softmax(logits[0].float(), dim=-1)
    n_frames = probs.shape[0]
    duration = len(audio) / spec.sampling_rate
    stride = max(1, n_frames // max_frames)

    top_probs, top_ids = torch.max(probs, dim=-1)
    frames = []
    for i in range(0, n_frames, stride):
        token_id = int(top_ids[i].item())
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
        except Exception:
            token = str(token_id)
        frames.append(
            {
                "time": round(duration * i / n_frames, 4),
                "token": token,
                "probability": round(float(top_probs[i].item()), 4),
            }
        )
    return {"frames": frames, "total_frames": n_frames, "duration": round(duration, 3)}


def extract_custom_embedding(spec: CustomModelSpec, audio_file_path: str) -> np.ndarray:
    """Mean-pooled last hidden state -- the embedding every task supports."""
    model, _processor, feature_extractor, _tokenizer = load_for_inference(spec)
    audio = _load_audio(spec, audio_file_path)
    inputs = _extract_features(feature_extractor, audio, spec.sampling_rate, spec.input_name)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        if spec.task == TASK_SEQ2SEQ:
            # Pool the encoder: decoder states depend on generated tokens and
            # would make the embedding a function of the transcript.
            encoder = model.get_encoder()
            hidden = encoder(**inputs, output_hidden_states=True).last_hidden_state
        else:
            outputs = model(**inputs, output_hidden_states=True)
            states = getattr(outputs, "hidden_states", None)
            hidden = states[-1] if states else outputs.logits
    return hidden.mean(dim=1)[0].float().cpu().numpy()
