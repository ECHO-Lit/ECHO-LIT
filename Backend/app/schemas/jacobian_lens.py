"""Persistent metadata for fitted, session-owned Jacobian lenses."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class JacobianLensStatus(StrEnum):
    FITTING = "fitting"
    READY = "ready"
    FAILED = "failed"


class JacobianLensRecord(BaseModel):
    lens_id: str
    session_id: str
    model_id: str
    model_revision: str
    architecture: str | None = None
    status: JacobianLensStatus = JacobianLensStatus.FITTING
    artifact_key: str | None = None
    metadata_key: str | None = None
    layer_count: int | None = None
    fit_job_id: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    sample_count: int = Field(ge=0)
