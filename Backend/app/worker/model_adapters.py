"""Worker-side adapters for the supported audio-model architectures.

Each adapter translates a generic job operation into the legacy model-service
functions.  This keeps the job executor architecture-agnostic and provides one
place to add a future Hugging Face custom-model adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.model_catalog import ModelDefinition, ModelKind, get_model_definition


class AudioModelAdapter(ABC):
    def __init__(self, definition: ModelDefinition) -> None:
        self.definition = definition

    @property
    def model_id(self) -> str:
        return self.definition.model_id

    @property
    def revision(self) -> str:
        return self.definition.revision

    @property
    def kind(self) -> ModelKind:
        return self.definition.kind

    def supports(self, operation: str) -> bool:
        return self.definition.supports(operation)

    @abstractmethod
    def resource_variant(self, operation: str) -> str:
        """Return the registry variant required by an operation."""

    @abstractmethod
    def load_resource(self, variant: str) -> Any:
        """Load the requested model resource through the existing service."""

    @abstractmethod
    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any]) -> Any:
        """Run a supported generic operation for one local audio file."""


class WhisperAdapter(AudioModelAdapter):
    def resource_variant(self, operation: str) -> str:
        if operation == "attention":
            return "eager-attention"
        if operation == "saliency":
            return "gradient"
        if operation in {"embedding", "hidden_states"}:
            return "encoder"
        return "generation"

    def load_resource(self, variant: str) -> Any:
        from app.services import model_loader_service as models

        if variant == "encoder":
            return (
                models.get_whisper_large_models()
                if self.model_id == "whisper-large"
                else models.get_whisper_base_models()
            )
        if variant == "gradient":
            return models.get_whisper_saliency_models(self.revision)
        if variant == "eager-attention":
            return models.get_whisper_attention_models(self.revision)
        return models.get_whisper_gen_model(self.revision)

    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any]) -> Any:
        if not self.supports(operation):
            raise ValueError(f"{self.model_id} does not support {operation}")
        if operation == "prediction":
            from app.services.model_loader_service import transcribe_whisper_base, transcribe_whisper_large

            return (
                transcribe_whisper_large(audio_path)
                if self.model_id == "whisper-large"
                else transcribe_whisper_base(audio_path)
            )
        if operation == "saliency":
            from app.services.saliency_service import generate_saliency

            return generate_saliency(audio_path, self.model_id, parameters.get("method", "gradcam"))
        if operation == "attention":
            from app.services.model_loader_service import extract_whisper_attention_pairs

            result = extract_whisper_attention_pairs(
                audio_path,
                model_size="large" if self.model_id == "whisper-large" else "base",
                layer_idx=int(parameters.get("layer_idx", 6)),
                head_idx=int(parameters.get("head_idx", 0)),
            )
            if result.get("error"):
                raise RuntimeError(f"Attention extraction failed: {result['error']}")
            return result
        if operation == "embedding":
            from app.services.model_loader_service import extract_whisper_embeddings

            return extract_whisper_embeddings(
                audio_path, "large" if self.model_id == "whisper-large" else "base"
            )
        raise ValueError(f"{self.model_id} does not yet implement {operation}")


class Wav2Vec2ClassificationAdapter(AudioModelAdapter):
    def resource_variant(self, operation: str) -> str:
        return "eager"

    def load_resource(self, variant: str) -> Any:
        del variant
        from app.services import model_loader_service as models

        return models.get_emotion_models()

    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any]) -> Any:
        if not self.supports(operation):
            raise ValueError(f"{self.model_id} does not support {operation}")
        if operation == "prediction":
            from app.services.model_loader_service import predict_emotion_wave2vec

            return predict_emotion_wave2vec(audio_path)
        if operation == "saliency":
            from app.services.saliency_service import generate_saliency

            return generate_saliency(audio_path, self.model_id, parameters.get("method", "gradcam"))
        if operation == "embedding":
            from app.services.model_loader_service import extract_wav2vec2_embeddings

            return extract_wav2vec2_embeddings(audio_path)
        raise ValueError(f"{self.model_id} does not yet implement {operation}")


def get_model_adapter(model_id: str) -> AudioModelAdapter:
    definition = get_model_definition(model_id)
    if definition.kind == ModelKind.SEQ2SEQ_ASR:
        return WhisperAdapter(definition)
    if definition.kind == ModelKind.AUDIO_CLASSIFICATION:
        return Wav2Vec2ClassificationAdapter(definition)
    raise ValueError(f"No adapter registered for model type: {definition.kind}")
