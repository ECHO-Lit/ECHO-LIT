from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from app.core.celery_app import celery_app, queue_for
from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository, TERMINAL_STATES
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
    audio_repository = AudioRepository()
    assets = []
    for audio_id in payload.audio_ids:
        asset = await audio_repository.get_owned(audio_id, request.state.sid)
        if not asset:
            raise HTTPException(status_code=404, detail=f"Audio not found: {audio_id}")
        assets.append(asset)

    now = datetime.now(timezone.utc)
    job_id = uuid.uuid4().hex
    record = JobRecord(
        job_id=job_id,
        session_id=request.state.sid,
        operation=payload.operation,
        model=payload.model,
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
        result_schema_version=settings.RESULT_SCHEMA_VERSION,
        code_version=settings.CODE_VERSION,
    )
    try:
        task_name = (
            "app.worker.tasks.orchestrate_batch"
            if len(assets) > 1
            else "app.worker.tasks.execute_job"
        )
        task = celery_app.send_task(
            task_name,
            args=[envelope.model_dump(mode="json")],
            queue="cpu" if len(assets) > 1 else queue_for(payload.operation.value, payload.model),
        )
        await jobs.update(job_id, task_id=task.id)
    except Exception as exc:
        await jobs.update(
            job_id,
            status=JobStatus.failure,
            error=JobError(code="broker_unavailable", message="Job broker is unavailable", retryable=True),
        )
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
