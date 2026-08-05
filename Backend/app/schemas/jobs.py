from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.model_catalog import MODEL_DEFINITIONS, ModelKind, is_supported_model


class JobOperation(str, Enum):
    prediction = "prediction"
    saliency = "saliency"
    attention = "attention"
    embedding = "embedding"
    perturbation = "perturbation"
    audio_features = "audio_features"
    linguistic_acoustic = "linguistic_acoustic"


class JobStatus(str, Enum):
    queued = "queued"
    started = "started"
    processing = "processing"
    success = "success"
    failure = "failure"
    cancelled = "cancelled"


SUPPORTED_MODELS = set(MODEL_DEFINITIONS)
MODEL_REQUIRED_OPERATIONS = {
    JobOperation.prediction,
    JobOperation.saliency,
    JobOperation.attention,
    JobOperation.embedding,
    JobOperation.linguistic_acoustic,
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
    cluster: bool = False
    min_cluster_size: int = Field(default=5, ge=2, le=50)


class PerturbationSpec(OperationParameters):
    type: Literal["noise", "time_masking", "pitch_shift", "time_stretch"]
    params: dict[str, float | int] = Field(default_factory=dict)


class PerturbationParameters(OperationParameters):
    perturbations: list[PerturbationSpec] = Field(min_length=1, max_length=10)


class AudioFeatureParameters(OperationParameters):
    pass


class LinguisticAcousticParameters(OperationParameters):
    task: Literal["transcription", "classification"]
    sweeps: list[dict[str, Any]]
    reference_transcript: str | None = None
    language: str | None = "en"
    normalize_loudness: bool = True
    include_lexical_control: bool = True


PARAMETER_MODELS: dict[JobOperation, type[OperationParameters]] = {
    JobOperation.prediction: PredictionParameters,
    JobOperation.saliency: SaliencyParameters,
    JobOperation.attention: AttentionParameters,
    JobOperation.embedding: EmbeddingParameters,
    JobOperation.perturbation: PerturbationParameters,
    JobOperation.audio_features: AudioFeatureParameters,
    JobOperation.linguistic_acoustic: LinguisticAcousticParameters,
}


class JobCreateRequest(BaseModel):
    operation: JobOperation
    audio_ids: list[str] = Field(min_length=1, max_length=200)
    model: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_operation(self) -> "JobCreateRequest":
        if self.operation == JobOperation.linguistic_acoustic:
            # Dispatched through its own render->infer->aggregate chord
            # (fr7_orchestrate), not the single-shot execute_job/orchestrate_batch
            # pipeline this endpoint drives -- see POST /analyses/linguistic-vs-acoustic.
            raise ValueError(
                "linguistic_acoustic must be submitted via POST /analyses/linguistic-vs-acoustic"
            )
        if self.operation in MODEL_REQUIRED_OPERATIONS and not self.model:
            raise ValueError(f"model must be one of: {', '.join(sorted(SUPPORTED_MODELS))}")
        if self.operation in SINGLE_AUDIO_OPERATIONS and len(self.audio_ids) != 1:
            raise ValueError(f"{self.operation.value} requires exactly one audio_id")
        if self.model in MODEL_DEFINITIONS and not MODEL_DEFINITIONS[self.model].supports(self.operation.value):
            raise ValueError(f"{self.model} does not support {self.operation.value}")
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


class RuntimeModelSpec(BaseModel):
    hf_repo: str
    revision: str | None = None
    kind: ModelKind
    capabilities: list[str]


class TaskEnvelope(BaseModel):
    job_id: str
    session_id: str
    operation: JobOperation
    model: str | None = None
    model_spec: RuntimeModelSpec | None = None
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
