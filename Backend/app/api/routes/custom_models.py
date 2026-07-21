"""REST surface for user-supplied Hugging Face audio models.

Validation and registration both load models, which is slow and CPU-bound, so
every such call is pushed onto a worker thread rather than blocking the event
loop for the whole backend.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.settings import settings
from app.services.custom_model_service import (
    CAPABILITIES_BY_TASK,
    CAPABILITY_LABELS,
    SUPPORTED_CONSTRAINTS,
    TASK_LABELS,
    ModelValidationError,
    delete_model,
    get_model_spec,
    list_models,
    register_model,
    validate_model,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def get_session_id(request: Request) -> str:
    session_id = getattr(request.state, "sid", None)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID found")
    return session_id


@router.get("/custom-models/capabilities")
async def get_capabilities():
    """The support matrix: tasks, per-task analyses, constraints and limits.

    The frontend renders its compatibility documentation from this, so the
    contract is stated once, in the backend.
    """
    return {
        "constraints": SUPPORTED_CONSTRAINTS,
        "tasks": [
            {
                "task": task,
                "label": TASK_LABELS[task],
                "capabilities": [
                    {"id": c, "label": CAPABILITY_LABELS.get(c, c)} for c in capabilities
                ],
            }
            for task, capabilities in CAPABILITIES_BY_TASK.items()
        ],
        "limits": {
            "max_parameters": settings.CUSTOM_MODEL_MAX_PARAMS,
            "max_storage_bytes": settings.CUSTOM_MODEL_MAX_STORAGE_BYTES,
            "max_inference_seconds": settings.CUSTOM_MODEL_MAX_INFERENCE_SECONDS,
            "max_models_per_session": settings.CUSTOM_MODEL_MAX_PER_SESSION,
        },
    }


@router.post("/custom-models/validate")
async def validate_custom_model(
    request: Request,
    body: dict = Body(..., example={"model_id": "facebook/wav2vec2-base-960h", "deep": False}),
):
    """Check a repo against every constraint without registering it.

    `deep=false` (the default) inspects only the config and processor -- a few
    KB of download. `deep=true` additionally fetches weights and runs a smoke
    test, which is what registration does.
    """
    get_session_id(request)
    model_id = (body.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    revision: Optional[str] = (body.get("revision") or "").strip() or None
    deep = bool(body.get("deep", False))

    try:
        result = await asyncio.to_thread(validate_model, model_id, revision, deep)
    except Exception as e:
        logger.exception("custom_model: validation of %s crashed", model_id)
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

    # A deep validation keeps the loaded model on the result for registration
    # to reuse; it must never reach the wire.
    result.pop("_model", None)
    result.pop("_facts", None)
    return result


@router.post("/custom-models/register")
async def register_custom_model(
    request: Request,
    body: dict = Body(..., example={"name": "wav2vec2-ctc", "model_id": "facebook/wav2vec2-base-960h"}),
):
    """Validate a repo fully and add it to this session's model list."""
    session_id = get_session_id(request)
    name = (body.get("name") or "").strip()
    model_id = (body.get("model_id") or "").strip()
    revision: Optional[str] = (body.get("revision") or "").strip() or None

    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not name:
        # Default to the repo name so the common case needs one field.
        name = model_id.split("/")[-1]

    try:
        result = await register_model(session_id, name, model_id, revision)
    except ModelValidationError as e:
        # 422: the request was well-formed but the model is not admissible.
        # The report tells the user which constraint failed and why.
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(e),
                "report": e.report.as_dict() if e.report else None,
            },
        )
    except Exception as e:
        logger.exception("custom_model: registration of %s crashed", model_id)
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

    return JSONResponse(status_code=201, content=result)


@router.get("/custom-models/list")
async def list_custom_models(request: Request):
    session_id = get_session_id(request)
    models = await list_models(session_id)
    return {"session_id": session_id, "models": models, "total": len(models)}


@router.get("/custom-models/{name}")
async def get_custom_model(request: Request, name: str):
    session_id = get_session_id(request)
    spec = await get_model_spec(session_id, name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Custom model '{name}' not found in this session")
    payload = spec.to_dict()
    payload["formatted_name"] = f"custom:{session_id}:{name}"
    return payload


@router.delete("/custom-models/{name}")
async def delete_custom_model(request: Request, name: str):
    session_id = get_session_id(request)
    if not await delete_model(session_id, name):
        raise HTTPException(status_code=404, detail=f"Custom model '{name}' not found in this session")
    return {"message": f"Custom model '{name}' deleted", "name": name, "session_id": session_id}
