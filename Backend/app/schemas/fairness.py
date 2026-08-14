"""FR-10 request/response schemas for POST /api/v1/analyses/fairness and
GET /api/v1/analyses/fairness/groupable (docs/FR10plan.md Part 1 S5.2).

Pydantic-only: safe to import from the FastAPI control plane.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.jobs import JobStatus

MetricId = Literal[
    "wer", "cer", "accuracy", "macro_f1", "ece",
    "silhouette", "leakage", "grounding_lift", "attribution_entropy",
]

# Cheap, control-plane-only budget guard. The real cost is unknown until the
# dataset is partitioned on a worker (PE-1: submission returns <500ms
# regardless of computation size) -- this only rejects obviously oversized
# requests before they are queued.
MAX_TOTAL_ITEMS = 20_000


class FairnessThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    absolute_error_gap: float = Field(default=0.05, ge=0.0, le=1.0)
    disparity_ratio: float = Field(default=1.25, ge=1.0, le=10.0)
    grounding_gap: float = Field(default=0.15, ge=0.0, le=5.0)
    alpha: float = Field(default=0.05, gt=0.0, le=0.25)


class FairnessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, max_length=200)
    grouping_key: list[str] = Field(min_length=1, max_length=2)
    model: str
    task: Literal["transcription", "classification", "auto"] = "auto"

    reference_group: str | None = None
    filters: dict[str, list[str]] | None = None
    language: str | None = "en"

    metrics: list[MetricId] = Field(default_factory=lambda: ["wer", "cer"])
    include_representation: bool = True
    include_explanations: bool = True
    include_faithfulness: bool = False
    saliency_method: Literal["gradcam", "lime", "shap"] = "gradcam"

    min_group_size: int = Field(default=8, ge=2, le=1000)
    min_speakers_per_group: int = Field(default=2, ge=1, le=100)
    max_items_per_group: int | None = Field(default=None, ge=2, le=2000)
    shard_size: int = Field(default=8, ge=1, le=32)
    explain_per_group: int = Field(default=12, ge=2, le=64)
    explain_shard_size: int = Field(default=4, ge=1, le=16)

    n_bootstrap: int = Field(default=2000, ge=200, le=20000)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    thresholds: FairnessThresholds = Field(default_factory=FairnessThresholds)

    @model_validator(mode="after")
    def validate_request(self) -> "FairnessRequest":
        if len(set(self.grouping_key)) != len(self.grouping_key):
            raise ValueError("grouping_key contains a duplicate column")
        if self.include_faithfulness and not self.include_explanations:
            raise ValueError("include_faithfulness requires include_explanations")
        if self.filters:
            for column in self.filters:
                if column in self.grouping_key:
                    raise ValueError(f"'{column}' cannot be both a filter and a grouping column")
        if self.max_items_per_group and self.max_items_per_group * 8 > MAX_TOTAL_ITEMS:
            raise ValueError(f"Requested budget exceeds {MAX_TOTAL_ITEMS} items across groups.")
        return self


class GroupableColumn(BaseModel):
    column: str
    n_values: int
    n_missing: int
    counts: dict[str, int]
    is_speaker_column: bool
    is_stratum_only: bool


class GroupableColumnsResponse(BaseModel):
    dataset: str
    n_rows: int
    speaker_column: str | None
    content_column: str | None
    columns: list[GroupableColumn]


class FairnessAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    status_url: str
    result_url: str
    dataset: str
    grouping_key: list[str]
    poll_after_ms: int = 1500
    notes: list[str] = Field(default_factory=list)
