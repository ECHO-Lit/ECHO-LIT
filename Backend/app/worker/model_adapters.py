"""Worker-side adapters for the supported audio-model architectures.

Each adapter translates a generic job operation into the legacy model-service
functions.  This keeps the job executor architecture-agnostic and provides one
place to add a future Hugging Face custom-model adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.model_catalog import ModelDefinition, ModelKind, get_model_definition
from app.schemas.jobs import RuntimeModelSpec


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
    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any], resource: Any = None) -> Any:
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

    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any], resource: Any = None) -> Any:
        del resource
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

            return generate_saliency(
                audio_path,
                self.model_id,
                parameters.get("method", "gradcam"),
                full_audio=parameters.get("full_audio", False),
            )
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

    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any], resource: Any = None) -> Any:
        del resource
        if not self.supports(operation):
            raise ValueError(f"{self.model_id} does not support {operation}")
        if operation == "prediction":
            from app.services.model_loader_service import predict_emotion_wave2vec

            return predict_emotion_wave2vec(audio_path)
        if operation == "saliency":
            from app.services.saliency_service import generate_saliency

            return generate_saliency(
                audio_path,
                self.model_id,
                parameters.get("method", "gradcam"),
                full_audio=parameters.get("full_audio", False),
            )
        if operation == "embedding":
            from app.services.model_loader_service import extract_wav2vec2_embeddings

            return extract_wav2vec2_embeddings(audio_path)
        raise ValueError(f"{self.model_id} does not yet implement {operation}")


class GenericHuggingFaceAdapter(AudioModelAdapter):
    """Standard-transformers adapter for a worker-validated custom model."""

    def __init__(self, model_id: str, spec: RuntimeModelSpec) -> None:
        definition = ModelDefinition(model_id, spec.hf_repo, spec.kind, frozenset(spec.capabilities))
        super().__init__(definition)
        self.hf_revision = spec.revision

    def resource_variant(self, operation: str) -> str:
        return "eager-attention" if operation == "attention" else "generic"

    def load_resource(self, variant: str) -> Any:
        from transformers import AutoModelForAudioClassification, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor
        from app.core.device import INFERENCE_DEVICE
        options = {"trust_remote_code": False, "low_cpu_mem_usage": True}
        if self.hf_revision:
            options["revision"] = self.hf_revision
        if variant == "eager-attention":
            # SDPA does not return attention weights; standard Transformers
            # seq2seq models provide them with the eager implementation.
            options["attn_implementation"] = "eager"
        cls = {
            ModelKind.SEQ2SEQ_ASR: AutoModelForSpeechSeq2Seq,
            ModelKind.CTC_ASR: AutoModelForCTC,
            ModelKind.AUDIO_CLASSIFICATION: AutoModelForAudioClassification,
        }[self.kind]
        processor = AutoProcessor.from_pretrained(self.revision, **options)
        model = cls.from_pretrained(self.revision, **options).to(INFERENCE_DEVICE).eval()
        return processor, model

    def execute(self, operation: str, audio_path: str, parameters: dict[str, Any], resource: Any = None) -> Any:
        if operation not in {"prediction", "embedding", "saliency", "attention"}:
            raise ValueError(f"{self.model_id} does not yet implement {operation}")
        if resource is None:
            resource = self.load_resource("generic")
        processor, model = resource
        import librosa
        import numpy as np
        import torch

        sample_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        if not np.all(np.isfinite(audio)):
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
        if operation == "saliency":
            return self._generate_saliency(model, inputs, audio, sample_rate, parameters)
        if operation == "attention":
            return self._extract_attention(processor, model, inputs, audio, sample_rate, parameters)
        with torch.no_grad():
            if operation == "embedding":
                output = model(**inputs, output_hidden_states=True, return_dict=True)
                states = output.hidden_states[-1] if output.hidden_states else None
                if states is None:
                    raise ValueError("Model did not return hidden states")
                return states.mean(dim=1).squeeze().cpu().numpy()
            if self.kind == ModelKind.SEQ2SEQ_ASR:
                ids = model.generate(**inputs)
                return {"text": processor.batch_decode(ids, skip_special_tokens=True)[0]}
            output = model(**inputs, return_dict=True)
            if self.kind == ModelKind.CTC_ASR:
                ids = output.logits.argmax(dim=-1)
                return {"text": processor.batch_decode(ids)[0]}
            probabilities = torch.softmax(output.logits[0], dim=-1)
            index = int(probabilities.argmax())
            labels = getattr(model.config, "id2label", {})
            scores = {str(labels.get(i, i)): float(score) for i, score in enumerate(probabilities.cpu())}
            return {"predicted_label": str(labels.get(index, index)), "confidence": float(probabilities[index]), "scores": scores}

    def _extract_attention(self, processor: Any, model: Any, inputs: dict[str, Any], audio: Any, sample_rate: int, parameters: dict[str, Any]) -> dict[str, Any]:
        """Extract decoder self-attention for a standard seq2seq speech model."""
        import numpy as np
        import torch

        if self.kind != ModelKind.SEQ2SEQ_ASR:
            raise ValueError("Attention is available only for custom seq2seq speech models")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            outputs = model(
                **inputs,
                decoder_input_ids=generated_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        layers = getattr(outputs, "decoder_attentions", None)
        if not layers:
            raise ValueError("Model did not return decoder attention weights")
        layer_index = min(int(parameters.get("layer_idx", 0)), len(layers) - 1)
        if layer_index < 0:
            layer_index = 0
        layer = layers[layer_index]
        if layer is None or layer.ndim != 4:
            raise ValueError("Model returned an unsupported decoder-attention shape")
        head_index = min(int(parameters.get("head_idx", 0)), layer.shape[1] - 1)
        if head_index < 0:
            head_index = 0

        matrix = layer[0, head_index].float().cpu().numpy()
        token_ids = generated_ids[0].tolist()
        tokenizer = getattr(processor, "tokenizer", processor)
        convert_ids = getattr(tokenizer, "convert_ids_to_tokens", None)
        raw_tokens = convert_ids(token_ids) if callable(convert_ids) else [str(token) for token in token_ids]
        count = min(len(raw_tokens), matrix.shape[0], matrix.shape[1], 64)
        if count == 0:
            raise ValueError("Model generated no tokens for attention analysis")
        matrix = matrix[:count, :count]
        tokens = [self._display_token(token, index) for index, token in enumerate(raw_tokens[:count])]
        total_duration = float(len(audio)) / float(sample_rate) if len(audio) else 0.0
        token_duration = total_duration / count if count else 0.0
        chunks = [
            {"text": token, "timestamp": [index * token_duration, (index + 1) * token_duration]}
            for index, token in enumerate(tokens)
        ]
        pairs = [
            {
                "from_word": tokens[row],
                "to_word": tokens[column],
                "from_time": chunks[row]["timestamp"],
                "to_time": chunks[column]["timestamp"],
                "attention_weight": float(matrix[row, column]),
                "from_index": row,
                "to_index": column,
            }
            for row in range(count)
            for column in range(count)
        ]
        return {
            "model": self.model_id,
            "attention_type": "decoder_self",
            "layer": layer_index,
            "head": head_index,
            "attention_pairs": pairs,
            "timestamp_attention": [
                {
                    "time": (index + 0.5) * token_duration,
                    "attention": float(np.max(matrix[index])),
                    "frame_index": index,
                    "max_outgoing": float(np.max(matrix[index])),
                    "avg_incoming": float(np.mean(matrix[:, index])),
                    "self_attention": float(matrix[index, index]),
                }
                for index in range(count)
            ],
            "total_duration": total_duration,
            "sequence_length": count,
            "word_chunks": chunks,
        }

    @staticmethod
    def _display_token(token: Any, index: int) -> str:
        text = str(token).replace("Ġ", " ").replace("▁", " ").strip()
        return text if text else f"token_{index + 1}"

    def _generate_saliency(self, model: Any, inputs: dict[str, Any], audio: Any, sample_rate: int, parameters: dict[str, Any]) -> dict[str, Any]:
        """Produce input-gradient saliency for a standard Transformers audio model.

        Unlike the built-in Whisper and Wav2Vec2 integrations, custom models do
        not share internal module names.  Input-gradient attribution is stable
        across the supported Transformers audio architectures and makes no
        assumptions about a repository's implementation details.
        """
        import numpy as np
        import torch

        input_key = next(
            (key for key in ("input_values", "input_features") if key in inputs),
            None,
        )
        if input_key is None:
            raise ValueError("Model processor did not provide input_values or input_features")
        primary_input = inputs[input_key].detach().requires_grad_(True)
        inputs[input_key] = primary_input
        model.zero_grad(set_to_none=True)

        if self.kind == ModelKind.SEQ2SEQ_ASR:
            encoder = model.get_encoder()
            try:
                encoded = encoder(**inputs)
            except TypeError:
                # Some encoder implementations only accept their primary tensor.
                encoded = encoder(primary_input)
            hidden_states = getattr(encoded, "last_hidden_state", None)
            if hidden_states is None:
                hidden_states = encoded[0]
            target = hidden_states.pow(2).mean()
        else:
            output = model(**inputs, return_dict=True)
            logits = output.logits
            if self.kind == ModelKind.CTC_ASR:
                target_index = int(logits[0].detach().mean(dim=0).argmax())
                target = logits[0, :, target_index].mean()
            else:
                target = logits[0].max()

        gradients = torch.autograd.grad(target, primary_input, allow_unused=False)[0]
        attribution = (primary_input * gradients).detach().abs().squeeze(0)
        series = self._normalise_timeline(attribution)
        total_duration = float(len(audio)) / float(sample_rate) if len(audio) else 0.0
        return {
            "model": self.model_id,
            "method": parameters.get("method", "gradcam"),
            "segments": self._time_segments(series, total_duration),
            "total_duration": total_duration,
            "series": series.tolist(),
        }

    @staticmethod
    def _normalise_timeline(attribution: Any) -> Any:
        """Reduce arbitrary model input shapes to a normalized 1D timeline."""
        import numpy as np

        values = attribution.float()
        if values.ndim == 0:
            values = values.unsqueeze(0)
        elif values.ndim > 1:
            dimensions = list(values.shape)
            feature_axis = next(
                (index for index, size in enumerate(dimensions) if size in {32, 64, 80, 96, 128}),
                None,
            )
            time_axis = max(range(values.ndim), key=lambda index: dimensions[index])
            if feature_axis is not None and feature_axis != time_axis:
                time_axis = next(index for index in range(values.ndim) if index != feature_axis)
            values = values.mean(dim=tuple(index for index in range(values.ndim) if index != time_axis))

        series = values.cpu().numpy().astype(np.float32, copy=False)
        if series.size == 0:
            return np.zeros(1, dtype=np.float32)
        maximum = float(series.max())
        if maximum > 0:
            series = series / maximum
        if series.size > 2:
            width = max(3, min(31, (series.size // 64) * 2 + 1))
            series = np.convolve(series, np.ones(width, dtype=np.float32) / width, mode="same")
            peak = float(np.percentile(series, 95))
            if peak > 0:
                series = np.clip(series / peak, 0, 1)
        return series

    @staticmethod
    def _time_segments(series: Any, total_duration: float) -> list[dict[str, Any]]:
        import numpy as np

        if total_duration <= 0 or len(series) == 0:
            return []
        count = max(8, min(32, int(total_duration * 2)))
        segments = []
        for index in range(count):
            start = int(index * len(series) / count)
            end = max(start + 1, int((index + 1) * len(series) / count))
            value = float(np.mean(series[start:end]))
            segments.append({
                "start_time": index * total_duration / count,
                "end_time": (index + 1) * total_duration / count,
                "word": f"segment_{index + 1}",
                "saliency": value,
                "intensity": max(0.1, value),
            })
        return segments


def get_model_adapter(model_id: str, model_spec: RuntimeModelSpec | None = None) -> AudioModelAdapter:
    if model_spec:
        return GenericHuggingFaceAdapter(model_id, model_spec)
    definition = get_model_definition(model_id)
    if definition.kind == ModelKind.SEQ2SEQ_ASR:
        return WhisperAdapter(definition)
    if definition.kind == ModelKind.AUDIO_CLASSIFICATION:
        return Wav2Vec2ClassificationAdapter(definition)
    raise ValueError(f"No adapter registered for model type: {definition.kind}")
