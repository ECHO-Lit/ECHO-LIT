"""Model definitions and capabilities shared by the API and worker layers.

The catalogue is intentionally dependency-free: the FastAPI control plane can
validate a requested operation without importing PyTorch or Transformers.  A
future custom-model registry can provide the same ``ModelDefinition`` shape
after validating a Hugging Face repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelKind(StrEnum):
    SEQ2SEQ_ASR = "seq2seq_asr"
    AUDIO_CLASSIFICATION = "audio_classification"


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    revision: str
    kind: ModelKind
    capabilities: frozenset[str]

    def supports(self, operation: str) -> bool:
        return operation in self.capabilities


# These built-ins preserve the existing public model IDs.  Do not scatter this
# mapping through routes and workers: custom models will be represented by the
# same definition once registration is implemented.
MODEL_DEFINITIONS: dict[str, ModelDefinition] = {
    "whisper-base": ModelDefinition(
        model_id="whisper-base",
        revision="openai/whisper-base",
        kind=ModelKind.SEQ2SEQ_ASR,
        capabilities=frozenset({
            "prediction",
            "embedding",
            "attention",
            "saliency",
            "hidden_states",
            "decoder_activations",
        }),
    ),
    "whisper-large": ModelDefinition(
        model_id="whisper-large",
        revision="openai/whisper-large-v3",
        kind=ModelKind.SEQ2SEQ_ASR,
        capabilities=frozenset({
            "prediction",
            "embedding",
            "attention",
            "saliency",
            "hidden_states",
            "decoder_activations",
        }),
    ),
    "wav2vec2": ModelDefinition(
        model_id="wav2vec2",
        revision="r-f/wav2vec-english-speech-emotion-recognition",
        kind=ModelKind.AUDIO_CLASSIFICATION,
        capabilities=frozenset({
            "prediction",
            "embedding",
            "saliency",
            "hidden_states",
        }),
    ),
}

# Kept as a mapping for result-cache compatibility while callers migrate to
# ``get_model_definition(...).revision``.
MODEL_REVISIONS = {model_id: definition.revision for model_id, definition in MODEL_DEFINITIONS.items()}


def get_model_definition(model_id: str) -> ModelDefinition:
    try:
        return MODEL_DEFINITIONS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_id}") from exc


def is_supported_model(model_id: str | None) -> bool:
    return model_id in MODEL_DEFINITIONS

