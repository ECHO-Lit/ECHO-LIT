from __future__ import annotations

from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from app.core.celery_app import celery_app
from app.repositories.models import CustomModelRepository
from app.schemas.models import (
    CustomModelCreateRequest,
    CustomModelCreateResponse,
    CustomModelRecord,
    CustomModelStatus,
)


router = APIRouter(prefix="/models")


@router.post("", response_model=CustomModelCreateResponse, status_code=202)
async def register_custom_model(payload: CustomModelCreateRequest, request: Request):
    """Validate a standard Hugging Face audio model on a worker.

    The API deliberately does not import Transformers or execute model code;
    validation runs in the isolated worker process with remote code disabled.
    """
    now = datetime.now(timezone.utc)
    record = CustomModelRecord(
        model_id=f"custom-{uuid.uuid4().hex}",
        session_id=request.state.sid,
        hf_repo=payload.hf_repo,
        revision=payload.revision,
        created_at=now,
        updated_at=now,
    )
    repository = CustomModelRepository()
    await repository.create(record)
    try:
        task = celery_app.send_task(
            "app.worker.tasks.validate_custom_model",
            args=[record.model_id],
            queue="cpu",
        )
        record.task_id = task.id
        await repository.save(record)
    except Exception as exc:
        record.status = CustomModelStatus.FAILED
        record.error = "Model-validation worker is unavailable"
        await repository.save(record)
        raise HTTPException(status_code=503, detail=record.error) from exc
    return CustomModelCreateResponse(
        model_id=record.model_id,
        status=record.status,
        status_url=f"/models/{record.model_id}",
    )


@router.get("", response_model=list[CustomModelRecord])
async def list_custom_models(request: Request):
    return await CustomModelRepository().list_owned(request.state.sid)


@router.get("/{model_id}", response_model=CustomModelRecord)
async def get_custom_model(model_id: str, request: Request):
    record = await CustomModelRepository().get_owned(model_id, request.state.sid)
    if not record:
        raise HTTPException(status_code=404, detail="Custom model not found")
    return record


@router.delete("/{model_id}", status_code=204)
async def delete_custom_model(model_id: str, request: Request, response: Response):
    record = await CustomModelRepository().get_owned(model_id, request.state.sid)
    if not record:
        raise HTTPException(status_code=404, detail="Custom model not found")
    if record.task_id and record.status == CustomModelStatus.VALIDATING:
        celery_app.control.revoke(record.task_id, terminate=False)
    await CustomModelRepository().delete(record)
    response.status_code = 204
    return None
