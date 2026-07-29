"""Session-scoped custom Hugging Face model records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.core.model_catalog import ModelKind


class CustomModelStatus(StrEnum):
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"


class CustomModelCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_repo: str = Field(
        min_length=3,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    revision: str | None = Field(default=None, min_length=1, max_length=128)


class CustomModelRecord(BaseModel):
    model_id: str
    session_id: str
    hf_repo: str
    revision: str | None = None
    status: CustomModelStatus = CustomModelStatus.VALIDATING
    kind: ModelKind | None = None
    capabilities: list[str] = Field(default_factory=list)
    processor_type: str | None = None
    task_id: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class CustomModelCreateResponse(BaseModel):
    model_id: str
    status: CustomModelStatus
    status_url: str
