"""FR-7: linguistic-vs-acoustic influence analysis (docs/FR7plan.md Part 1 S5.3).

Dispatches straight to the fr7_orchestrate Celery chord and returns 202; the
client polls the existing FR-14 /jobs/{id} and /jobs/{id}/result endpoints --
this endpoint deliberately does not introduce a parallel status API.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, HTTPException, Request

from app.core.celery_app import celery_app
from app.core.model_catalog import MODEL_DEFINITIONS, ModelKind, custom_model_capabilities
from app.core.settings import settings
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.repositories.models import CustomModelRepository
from app.schemas.analyses import LinguisticAcousticAccepted, LinguisticAcousticRequest
from app.schemas.jobs import (
    JobError,
    JobOperation,
    JobProgress,
    JobRecord,
    JobStatus,
    RuntimeModelSpec,
    TaskAudio,
    TaskEnvelope,
)
from app.schemas.models import CustomModelStatus
from app.services.fr7_planning import estimate_cost, resolve_task

router = APIRouter(prefix="/analyses")


@router.post(
    "/linguistic-vs-acoustic", response_model=LinguisticAcousticAccepted, status_code=202
)
async def create_linguistic_acoustic_analysis(payload: LinguisticAcousticRequest, request: Request):
    """FR-7. Returns 202 immediately; poll `status_url` (FR-14) for completion.

    The control plane never touches PyTorch: model capability is checked
    against the dependency-free catalogue, and all DSP/inference happens on
    workers.
    """
    session_id = request.state.sid

    model_spec: RuntimeModelSpec | None = None
    if payload.model in MODEL_DEFINITIONS:
        definition = MODEL_DEFINITIONS[payload.model]
        if not definition.supports("prediction"):
            raise HTTPException(400, f"{payload.model} cannot run prediction")
        kind = definition.kind
    else:
        custom = await CustomModelRepository().get_owned(payload.model, session_id)
        if not custom:
            raise HTTPException(404, "Custom model not found")
        if custom.status != CustomModelStatus.READY or not custom.kind:
            raise HTTPException(409, "Custom model is not ready")
        if "prediction" not in custom_model_capabilities(custom.kind):
            raise HTTPException(400, "Custom model does not support prediction")
        kind = custom.kind
        model_spec = RuntimeModelSpec(
            hf_repo=custom.hf_repo, revision=custom.revision,
            kind=custom.kind, capabilities=custom.capabilities,
        )

    task = resolve_task(payload.task, kind)
    if task == "transcription" and kind == ModelKind.AUDIO_CLASSIFICATION:
        raise HTTPException(400, "A classification model cannot run a transcription analysis")
    if task == "classification" and kind != ModelKind.AUDIO_CLASSIFICATION:
        raise HTTPException(400, "A transcription model cannot run a classification analysis")

    audio_repository = AudioRepository()
    assets = []
    for audio_id in payload.audio_ids:
        asset = await audio_repository.get_owned(audio_id, session_id)
        if not asset:
            raise HTTPException(404, f"Audio not found: {audio_id}")
        if asset.duration_seconds < 0.5:
            raise HTTPException(422, f"{asset.filename} is too short for perturbation analysis")
        assets.append(asset)

    now = datetime.now(timezone.utc)
    job_id = uuid.uuid4().hex
    parameters = {
        "task": task,
        "sweeps": [sweep.model_dump(mode="json") for sweep in payload.sweeps],
        "reference_transcript": payload.reference_transcript,
        "language": payload.language,
        "normalize_loudness": payload.normalize_loudness,
        "include_lexical_control": payload.include_lexical_control,
    }
    variants, seconds = estimate_cost(parameters, len(assets), payload.model)

    jobs = JobRepository()
    await jobs.create(JobRecord(
        job_id=job_id, session_id=session_id,
        operation=JobOperation.linguistic_acoustic, model=payload.model,
        audio_ids=payload.audio_ids, parameters=parameters,
        progress=JobProgress(current=0, total=2 * variants + 1, message="Queued"),
        created_at=now, updated_at=now,
    ))

    envelope = TaskEnvelope(
        job_id=job_id, session_id=session_id,
        operation=JobOperation.linguistic_acoustic, model=payload.model, model_spec=model_spec,
        audio=[
            TaskAudio(
                audio_id=asset.audio_id, object_key=asset.object_key, filename=asset.filename,
                media_type=asset.media_type, sha256=asset.sha256,
            )
            for asset in assets
        ],
        parameters=parameters,
        result_schema_version=settings.RESULT_SCHEMA_VERSION,
        code_version=settings.CODE_VERSION,
    )
    try:
        task_handle = celery_app.send_task(
            "app.worker.tasks.fr7_orchestrate",
            args=[envelope.model_dump(mode="json")], queue="cpu",
        )
        await jobs.update(job_id, task_id=task_handle.id)
    except Exception as exc:
        await jobs.update(
            job_id, status=JobStatus.failure,
            error=JobError(code="broker_unavailable", message="Job broker is unavailable", retryable=True),
        )
        raise HTTPException(503, "Job broker is unavailable") from exc

    return LinguisticAcousticAccepted(
        job_id=job_id, status=JobStatus.queued,
        status_url=f"/jobs/{job_id}", result_url=f"/jobs/{job_id}/result",
        estimated_variants=variants, estimated_seconds=seconds,
    )
