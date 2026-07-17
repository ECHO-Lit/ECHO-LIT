from fastapi import APIRouter, HTTPException, Body, Request
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel
from app.services.saliency_service import generate_saliency
from app.services.dataset_service import resolve_file
from app.core.redis import get_result, cache_result

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
SALIENCY_SCHEMA_VERSION = "v2"  # bump to bust stale caches after logic changes

class SaliencyRequest(BaseModel):
    model: str
    method: str = "gradcam"
    file_path: Optional[str] = None
    dataset: Optional[str] = None
    dataset_file: Optional[str] = None
    no_cache: Optional[bool] = False

class SaliencyResponse(BaseModel):
    model: str
    method: str
    segments: list
    total_duration: float
    emotion: Optional[str] = None
    series: Optional[list] = None

def get_session_id(request: Request) -> Optional[str]:
    """Extract session ID from request (optional for backwards compatibility)"""
    return getattr(request.state, 'sid', None)

@router.post("/saliency/generate", response_model=SaliencyResponse)
async def generate_saliency_endpoint(http_request: Request, request: SaliencyRequest):
    if not request.model:
        raise HTTPException(status_code=400, detail="Model is required")
    
    session_id = get_session_id(http_request)
    
    resolved_path = None
    if request.file_path:
        resolved_path = Path(request.file_path)
    elif request.dataset and request.dataset_file:
        try:
            resolved_path = resolve_file(request.dataset, request.dataset_file, session_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'."
        )
    
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")
    
    import hashlib
    # Include file size and modification time for better cache key uniqueness
    file_stat = resolved_path.stat()
    file_content_hash = hashlib.md5(
        f"{str(resolved_path)}_{file_stat.st_size}_{file_stat.st_mtime}".encode()
    ).hexdigest()
    cache_key = f"saliency_{SALIENCY_SCHEMA_VERSION}_{request.model}_{request.method}_{file_content_hash}"
    
    if not request.no_cache:
        cached_result = await get_result("saliency", cache_key)
        if cached_result is not None:
            logger.info(f"Returning cached saliency for {resolved_path}")
            return SaliencyResponse(**cached_result)
    
    # Check if we have existing prediction data to reuse
    prediction_cache_key = f"{request.model}_{file_content_hash}"
    existing_prediction = await get_result(request.model, prediction_cache_key)
    
    try:
        result = await asyncio.to_thread(
            generate_saliency, 
            str(resolved_path), 
            request.model, 
            request.method,
            existing_prediction
        )
        
        await cache_result("saliency", cache_key, result, ttl=6*60*60)
        logger.info(f"Cached saliency for {resolved_path}")
        
        return SaliencyResponse(**result)
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Error generating saliency for {resolved_path} (model={request.model}, method={request.method}): {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Saliency generation failed: {str(e)}")

@router.get("/saliency/{method}/{model}/{file_id}")
async def get_saliency(method: str, model: str, file_id: str):
    import hashlib
    cache_key = f"saliency_{model}_{method}_{file_id}"
    
    cached_result = await get_result("saliency", cache_key)
    if cached_result is None:
        raise HTTPException(status_code=404, detail="Saliency not found")
    
    return SaliencyResponse(**cached_result)
