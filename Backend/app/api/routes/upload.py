from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from app.core.settings import settings
from app.core.audio_probe import probe_audio
from app.core.storage import LocalObjectStorage, StorageError, get_storage
from app.repositories.audio import AudioRepository
from app.schemas.jobs import AudioAsset


router = APIRouter()
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


class MaterializeAudioRequest(BaseModel):
    dataset: str
    filename: str


def _asset_response(asset: AudioAsset) -> dict:
    return {
        "message": "File uploaded successfully",
        "audio_id": asset.audio_id,
        "file_id": asset.audio_id,
        "filename": asset.filename,
        "media_type": asset.media_type,
        "size": asset.size_bytes,
        "size_bytes": asset.size_bytes,
        "duration": asset.duration_seconds,
        "duration_seconds": asset.duration_seconds,
        "sample_rate": asset.sample_rate,
        "channels": asset.channels,
        "playback_url": f"/audio/{asset.audio_id}",
        "created_at": asset.created_at.isoformat(),
    }


@router.post("/upload", status_code=201)
async def upload_audio_file(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
):
    del model  # retained for one-release multipart compatibility; no inference occurs here
    original_name = Path(file.filename or "audio").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {extension or 'none'}")
    if file.content_type and not (
        file.content_type.startswith("audio/") or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")

    digest = hashlib.sha256()
    size = 0
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary:
            temp_path = Path(temporary.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Audio exceeds the 100 MB upload limit")
                digest.update(chunk)
                temporary.write(chunk)
        duration, sample_rate, channels = await asyncio.to_thread(probe_audio, temp_path)
        if duration > settings.MAX_AUDIO_DURATION_SECONDS:
            raise HTTPException(status_code=413, detail="Audio exceeds the 10 minute duration limit")

        audio_id = uuid.uuid4().hex
        object_key = f"uploads/{request.state.sid}/{audio_id}{extension}"
        storage = get_storage()
        await asyncio.to_thread(storage.put_file, object_key, temp_path, file.content_type)
        asset = AudioAsset(
            audio_id=audio_id,
            session_id=request.state.sid,
            object_key=object_key,
            filename=original_name,
            media_type=file.content_type or "application/octet-stream",
            size_bytes=size,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            sha256=digest.hexdigest(),
            created_at=datetime.now(timezone.utc),
        )
        try:
            await AudioRepository().create(asset)
        except Exception:
            await asyncio.to_thread(storage.delete, object_key)
            raise
        return _asset_response(asset)
    except HTTPException:
        raise
    except (StorageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
        await file.close()


@router.get("/upload/list")
async def list_uploaded_files(request: Request):
    assets = await AudioRepository().list_owned(request.state.sid)
    return {"files": [_asset_response(asset) for asset in assets]}


@router.post("/audio/materialize", status_code=201)
async def materialize_dataset_audio(payload: MaterializeAudioRequest, request: Request):
    from app.services.dataset_service import resolve_file

    try:
        source = resolve_file(payload.dataset, payload.filename, request.state.sid)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    extension = source.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported dataset audio type")
    duration, sample_rate, channels = await asyncio.to_thread(probe_audio, source)
    digest = await asyncio.to_thread(lambda: hashlib.sha256(source.read_bytes()).hexdigest())
    audio_id = uuid.uuid4().hex
    object_key = f"datasets/{request.state.sid}/{audio_id}{extension}"
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }
    storage = get_storage()
    await asyncio.to_thread(storage.put_file, object_key, source, media_types[extension])
    asset = AudioAsset(
        audio_id=audio_id,
        session_id=request.state.sid,
        object_key=object_key,
        filename=Path(payload.filename).name,
        media_type=media_types[extension],
        size_bytes=source.stat().st_size,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        sha256=digest,
        created_at=datetime.now(timezone.utc),
    )
    await AudioRepository().create(asset)
    return _asset_response(asset)


@router.get("/upload/metadata/{audio_id}")
async def get_audio_metadata(audio_id: str, request: Request):
    asset = await AudioRepository().get_owned(audio_id, request.state.sid)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio not found")
    return _asset_response(asset)


@router.delete("/upload/{audio_id}")
async def delete_uploaded_file(audio_id: str, request: Request):
    asset = await AudioRepository().delete(audio_id, request.state.sid)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio not found")
    await asyncio.to_thread(get_storage().delete, asset.object_key)
    return {"message": "File deleted successfully"}


async def _serve_audio(audio_id: str, request: Request):
    asset = await AudioRepository().get_owned(audio_id, request.state.sid)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio not found")
    storage = get_storage()
    headers = {
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": f'inline; filename="{asset.filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    if isinstance(storage, LocalObjectStorage):
        return FileResponse(storage.path_for(asset.object_key), media_type=asset.media_type, headers=headers)
    content = await asyncio.to_thread(storage.get_bytes, asset.object_key)
    return Response(content=content, media_type=asset.media_type, headers=headers)


@router.get("/audio/{audio_id}")
async def serve_audio(audio_id: str, request: Request):
    return await _serve_audio(audio_id, request)


@router.get("/upload/file/{audio_id}", deprecated=True)
async def serve_audio_compatibility(audio_id: str, request: Request):
    return await _serve_audio(audio_id, request)
