from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobOperation(str, Enum):
    prediction = "prediction"
    saliency = "saliency"
    attention = "attention"
    embedding = "embedding"
    perturbation = "perturbation"
    audio_features = "audio_features"


class JobStatus(str, Enum):
    queued = "queued"
    started = "started"
    processing = "processing"
    success = "success"
    failure = "failure"
    cancelled = "cancelled"


SUPPORTED_MODELS = {"whisper-base", "whisper-large", "wav2vec2"}
MODEL_REQUIRED_OPERATIONS = {
    JobOperation.prediction,
    JobOperation.saliency,
    JobOperation.attention,
    JobOperation.embedding,
}
SINGLE_AUDIO_OPERATIONS = {
    JobOperation.saliency,
    JobOperation.attention,
    JobOperation.perturbation,
}


class OperationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionParameters(OperationParameters):
    pass


class SaliencyParameters(OperationParameters):
    method: Literal["gradcam", "lime", "shap"] = "gradcam"


class AttentionParameters(OperationParameters):
    layer_idx: int = Field(default=6, ge=0, le=31)
    head_idx: int = Field(default=0, ge=0, le=31)


class EmbeddingParameters(OperationParameters):
    reduction: Literal["pca", "tsne", "umap"] = "pca"
    n_components: int = Field(default=2, ge=2, le=3)


class PerturbationSpec(OperationParameters):
    type: Literal["noise", "time_masking", "pitch_shift", "time_stretch"]
    params: dict[str, float | int] = Field(default_factory=dict)


class PerturbationParameters(OperationParameters):
    perturbations: list[PerturbationSpec] = Field(min_length=1, max_length=10)


class AudioFeatureParameters(OperationParameters):
    pass


PARAMETER_MODELS: dict[JobOperation, type[OperationParameters]] = {
    JobOperation.prediction: PredictionParameters,
    JobOperation.saliency: SaliencyParameters,
    JobOperation.attention: AttentionParameters,
    JobOperation.embedding: EmbeddingParameters,
    JobOperation.perturbation: PerturbationParameters,
    JobOperation.audio_features: AudioFeatureParameters,
}


class JobCreateRequest(BaseModel):
    operation: JobOperation
    audio_ids: list[str] = Field(min_length=1, max_length=200)
    model: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_operation(self) -> "JobCreateRequest":
        if self.operation in MODEL_REQUIRED_OPERATIONS and self.model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of: {', '.join(sorted(SUPPORTED_MODELS))}")
        if self.operation in SINGLE_AUDIO_OPERATIONS and len(self.audio_ids) != 1:
            raise ValueError(f"{self.operation.value} requires exactly one audio_id")
        if self.operation == JobOperation.attention and not (self.model or "").startswith("whisper"):
            raise ValueError("attention currently supports Whisper models only")
        if self.operation not in MODEL_REQUIRED_OPERATIONS and self.model is not None:
            raise ValueError(f"{self.operation.value} does not accept a model")
        parameter_model = PARAMETER_MODELS[self.operation].model_validate(self.parameters)
        self.parameters = parameter_model.model_dump(mode="json", exclude_none=True)
        return self


class AudioAsset(BaseModel):
    audio_id: str
    session_id: str
    object_key: str
    filename: str
    media_type: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int | None = None
    channels: int | None = None
    sha256: str
    created_at: datetime


class TaskAudio(BaseModel):
    audio_id: str
    object_key: str
    filename: str
    media_type: str
    sha256: str


class TaskEnvelope(BaseModel):
    job_id: str
    session_id: str
    operation: JobOperation
    model: str | None = None
    audio: list[TaskAudio]
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_schema_version: str
    code_version: str


class JobProgress(BaseModel):
    current: int = 0
    total: int = 1
    message: str = "Queued"


class JobError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class JobRecord(BaseModel):
    job_id: str
    session_id: str
    operation: JobOperation
    model: str | None = None
    audio_ids: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.queued
    progress: JobProgress = Field(default_factory=JobProgress)
    created_at: datetime
    updated_at: datetime
    task_id: str | None = None
    child_task_ids: list[str] = Field(default_factory=list)
    result_key: str | None = None
    cache_hit: bool = False
    error: JobError | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    cache_hit: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    operation: JobOperation
    model: str | None
    status: JobStatus
    progress: JobProgress
    created_at: datetime
    updated_at: datetime
    result_url: str | None = None
    error: JobError | None = None
    cache_hit: bool = False
