"""Worker-only validation of user-supplied Hugging Face audio models."""

from __future__ import annotations

from app.core.model_catalog import ModelKind, custom_model_capabilities
from app.repositories.models import CustomModelRepository
from app.schemas.models import CustomModelStatus


def _kind_and_capabilities(config) -> tuple[ModelKind, list[str]]:
    architectures = " ".join(getattr(config, "architectures", None) or [])
    if bool(getattr(config, "is_encoder_decoder", False)):
        return ModelKind.SEQ2SEQ_ASR, custom_model_capabilities(ModelKind.SEQ2SEQ_ASR)
    if "ForCTC" in architectures:
        return ModelKind.CTC_ASR, custom_model_capabilities(ModelKind.CTC_ASR)
    if "ForAudioClassification" in architectures or int(getattr(config, "num_labels", 0) or 0) > 1:
        return ModelKind.AUDIO_CLASSIFICATION, custom_model_capabilities(ModelKind.AUDIO_CLASSIFICATION)
    raise ValueError(
        "Model must be a standard speech Seq2Seq, CTC, or audio-classification architecture"
    )


async def validate_custom_model(model_id: str) -> None:
    """Preflight config and processor without loading arbitrary remote code."""
    repository = CustomModelRepository()
    record = await repository.get(model_id)
    if not record:
        return
    try:
        from transformers import AutoConfig, AutoProcessor

        options = {"revision": record.revision, "trust_remote_code": False}
        if record.revision is None:
            options.pop("revision")
        config = AutoConfig.from_pretrained(record.hf_repo, **options)
        processor = AutoProcessor.from_pretrained(record.hf_repo, **options)
        kind, capabilities = _kind_and_capabilities(config)
        if not hasattr(processor, "__call__"):
            raise ValueError("The model repository does not provide an audio processor")
        if kind == ModelKind.AUDIO_CLASSIFICATION and not getattr(config, "id2label", None):
            raise ValueError("Classification models must define id2label")
        record.kind = kind
        record.capabilities = capabilities
        record.processor_type = type(processor).__name__
        record.status = CustomModelStatus.READY
        record.error = None
    except Exception as exc:  # Surface a safe, bounded error to the owner only.
        record.status = CustomModelStatus.FAILED
        record.error = str(exc)[:500]
    await repository.save(record)
