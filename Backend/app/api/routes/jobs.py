from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from app.core.celery_app import celery_app, queue_for
from app.core.model_catalog import MODEL_REVISIONS, custom_model_capabilities
from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository, TERMINAL_STATES
from app.repositories.models import CustomModelRepository
from app.repositories.jacobian_lenses import JacobianLensRepository
from app.schemas.jacobian_lens import JacobianLensRecord, JacobianLensStatus
from app.schemas.models import CustomModelStatus
from app.schemas.jobs import (
    JobCreateRequest,
    JobCreateResponse,
    JobError,
    JobProgress,
    JobRecord,
    JobStatus,
    JobStatusResponse,
    TaskAudio,
    TaskEnvelope,
    RuntimeModelSpec,
)


router = APIRouter(prefix="/jobs")


def _status_response(record: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=record.job_id,
        operation=record.operation,
        model=record.model,
        status=record.status,
        progress=record.progress,
        created_at=record.created_at,
        updated_at=record.updated_at,
        result_url=f"/jobs/{record.job_id}/result" if record.status == JobStatus.success else None,
        error=record.error,
        cache_hit=record.cache_hit,
    )


@router.post("", response_model=JobCreateResponse, status_code=202)
async def create_job(payload: JobCreateRequest, request: Request):
    model_spec = None
    if payload.model and payload.model not in {"whisper-base", "whisper-large", "wav2vec2"}:
        custom = await CustomModelRepository().get_owned(payload.model, request.state.sid)
        if not custom:
            raise HTTPException(status_code=404, detail="Custom model not found")
        if custom.status != CustomModelStatus.READY or not custom.kind:
            raise HTTPException(status_code=409, detail="Custom model is not ready")
        # Make records created before a capability expansion usable immediately.
        # The generic adapter owns these capabilities for every validated kind.
        current_capabilities = custom_model_capabilities(custom.kind)
        if custom.capabilities != current_capabilities:
            custom.capabilities = current_capabilities
            await CustomModelRepository().save(custom)
        if payload.operation.value not in custom.capabilities:
            raise HTTPException(status_code=400, detail=f"Custom model does not support {payload.operation.value}")
        model_spec = RuntimeModelSpec(
            hf_repo=custom.hf_repo,
            revision=custom.revision,
            kind=custom.kind,
            capabilities=custom.capabilities,
        )
    audio_repository = AudioRepository()
    assets = []
    for audio_id in payload.audio_ids:
        asset = await audio_repository.get_owned(audio_id, request.state.sid)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Audio not found: {audio_id}")
        assets.append(asset)

    now = datetime.now(timezone.utc)
    job_id = uuid.uuid4().hex
    lens_repository = JacobianLensRepository()
    if payload.operation.value == "jacobian_lens_fit":
        lens_id = f"jlens-{uuid.uuid4().hex}"
        revision = (
            model_spec.revision or model_spec.hf_repo
            if model_spec else MODEL_REVISIONS.get(payload.model or "", "")
        )
        await lens_repository.create(JacobianLensRecord(
            lens_id=lens_id,
            session_id=request.state.sid,
            model_id=payload.model or "",
            model_revision=revision,
            fit_job_id=job_id,
            created_at=now,
            updated_at=now,
            sample_count=len(payload.parameters["samples"]),
        ))
        payload.parameters = {**payload.parameters, "lens_id": lens_id}
    elif payload.operation.value == "jacobian_lens_apply":
        lens = await lens_repository.get_owned(payload.parameters["lens_id"], request.state.sid)
        if not lens:
            raise HTTPException(status_code=404, detail="Jacobian lens not found")
        if lens.status != JacobianLensStatus.READY:
            raise HTTPException(status_code=409, detail=f"Jacobian lens is {lens.status.value}")
        if lens.model_id != payload.model:
            raise HTTPException(status_code=400, detail="Jacobian lens belongs to a different model")
    record = JobRecord(
        job_id=job_id,
        session_id=request.state.sid,
        operation=payload.operation,
        model=payload.model,
        model_spec=model_spec,
        audio_ids=payload.audio_ids,
        parameters=payload.parameters,
        progress=JobProgress(current=0, total=len(assets), message="Queued"),
        created_at=now,
        updated_at=now,
    )
    jobs = JobRepository()
    await jobs.create(record)
    envelope = TaskEnvelope(
        job_id=job_id,
        session_id=request.state.sid,
        operation=payload.operation,
        model=payload.model,
        audio=[
            TaskAudio(
                audio_id=asset.audio_id,
                object_key=asset.object_key,
                filename=asset.filename,
                media_type=asset.media_type,
                sha256=asset.sha256,
            )
            for asset in assets
        ],
        parameters=payload.parameters,
        model_spec=model_spec,
        result_schema_version=settings.RESULT_SCHEMA_VERSION,
        code_version=settings.CODE_VERSION,
    )
    try:
        task_name = (
            "app.worker.tasks.orchestrate_batch"
            if len(assets) > 1 and payload.operation.value != "jacobian_lens_fit"
            else "app.worker.tasks.execute_job"
        )
        task = celery_app.send_task(
            task_name,
            args=[envelope.model_dump(mode="json")],
            queue=(
                queue_for(payload.operation.value, payload.model)
                if payload.operation.value == "jacobian_lens_fit"
                else "cpu" if len(assets) > 1 else queue_for(payload.operation.value, payload.model)
            ),
        )
        await jobs.update(job_id, task_id=task.id)
    except Exception as exc:
        await jobs.update(
            job_id,
            status=JobStatus.failure,
            error=JobError(code="broker_unavailable", message="Job broker is unavailable", retryable=True),
        )
        if payload.operation.value == "jacobian_lens_fit":
            lens = await lens_repository.get_owned(payload.parameters["lens_id"], request.state.sid)
            if lens:
                lens.status = JacobianLensStatus.FAILED
                lens.error = "Lens-fitting worker is unavailable"
                await lens_repository.save(lens)
        raise HTTPException(status_code=503, detail="Job broker is unavailable") from exc
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.queued,
        status_url=f"/jobs/{job_id}",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, request: Request):
    record = await JobRepository().get_owned(job_id, request.state.sid)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return _status_response(record)


@router.get("/{job_id}/result")
async def get_job_result(job_id: str, request: Request):
    record = await JobRepository().get_owned(job_id, request.state.sid)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status != JobStatus.success or not record.result_key:
        raise HTTPException(status_code=409, detail=f"Job is {record.status.value}")
    try:
        return await asyncio.to_thread(get_storage().get_json, record.result_key)
    except Exception as exc:
        raise HTTPException(status_code=410, detail="Job result has expired") from exc


@router.delete("/{job_id}", status_code=202)
async def cancel_or_delete_job(job_id: str, request: Request, response: Response):
    jobs = JobRepository()
    record = await jobs.get_owned(job_id, request.state.sid)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.status in TERMINAL_STATES:
        if record.result_key:
            try:
                result = await asyncio.to_thread(get_storage().get_json, record.result_key)
                for item in result.get("items", []):
                    generated_id = (item.get("result") or {}).get("audio_id")
                    if generated_id and generated_id not in record.audio_ids:
                        asset = await AudioRepository().delete(generated_id, request.state.sid)
                        if asset:
                            await asyncio.to_thread(get_storage().delete, asset.object_key)
            except Exception:
                pass
            await asyncio.to_thread(get_storage().delete, record.result_key)
        await jobs.delete(record)
        response.status_code = 204
        return None
    await jobs.request_cancel(job_id)
    if record.task_id:
        celery_app.control.revoke(record.task_id, terminate=False)
    for child_task_id in record.child_task_ids:
        celery_app.control.revoke(child_task_id, terminate=False)
    if record.status == JobStatus.queued:
        await jobs.update(
            job_id,
            status=JobStatus.cancelled,
            progress=JobProgress(current=0, total=record.progress.total, message="Cancelled"),
        )
    return {"job_id": job_id, "status": "cancellation_requested"}


@router.delete("/jacobian-lenses/{lens_id}", status_code=204)
async def delete_jacobian_lens(lens_id: str, request: Request, response: Response):
    storage = get_storage()
    deleted = await JacobianLensRepository().delete(lens_id, request.state.sid, storage=storage)
    if not deleted:
        raise HTTPException(status_code=404, detail="Jacobian lens not found")
    return None
