from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
import logging
from pathlib import Path
import json

from app.services.custom_dataset_service import (
    get_custom_dataset_manager,
    format_custom_dataset_name,
    cleanup_session_datasets
)
from app.services.dataset_labels_service import (
    available_patterns,
    derive_from_filenames,
    parse_labels_csv,
    preview_dataset,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# An uploaded answer key is text, and a large one is a mistake rather than a
# use case: one row per audio file, and the audio itself is capped elsewhere.
MAX_LABEL_CSV_BYTES = 4 * 1024 * 1024


def get_session_id(request: Request) -> str:
    """Extract session ID from request"""
    session_id = getattr(request.state, 'sid', None)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID found")
    return session_id


@router.post("/dataset/create")
async def create_custom_dataset(
    request: Request,
    dataset_name: str = Form(..., description="Name for the custom dataset")
):
    """Create a new custom dataset in the current session"""
    session_id = get_session_id(request)
    
    try:
        manager = get_custom_dataset_manager(session_id)
        metadata = manager.create_dataset(dataset_name)
        
        # Return the formatted dataset name that can be used in other APIs
        formatted_name = format_custom_dataset_name(session_id, dataset_name)
        
        return JSONResponse(
            status_code=201,
            content={
                "message": "Dataset created successfully",
                "dataset_name": formatted_name,
                "original_name": dataset_name,
                "session_id": session_id,
                "metadata": metadata
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")


@router.post("/dataset/{dataset_name}/files")
async def upload_files_to_dataset(
    request: Request,
    dataset_name: str,
    files: List[UploadFile] = File(..., description="Audio files to upload to the dataset")
):
    """Upload multiple audio files to an existing custom dataset"""
    session_id = get_session_id(request)
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    for file in files:
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type for {file.filename}. Only audio files are allowed."
            )
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file extension for {file.filename}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    try:
        manager = get_custom_dataset_manager(session_id)
        uploaded_files = []
        errors = []
        
        for file in files:
            try:
                # Read file data
                file_data = await file.read()
                
                # Add file to dataset
                file_metadata = manager.add_file_to_dataset(
                    dataset_name, 
                    file.filename, 
                    file_data
                )
                uploaded_files.append(file_metadata)
                
            except Exception as e:
                error_msg = f"Failed to upload {file.filename}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Get updated dataset metadata
        dataset_metadata = manager.get_dataset_metadata(dataset_name)
        formatted_name = format_custom_dataset_name(session_id, dataset_name)
        
        response_data = {
            "message": f"Uploaded {len(uploaded_files)} files successfully",
            "dataset_name": formatted_name,
            "uploaded_files": uploaded_files,
            "total_files": len(uploaded_files),
            "dataset_metadata": dataset_metadata
        }
        
        if errors:
            response_data["errors"] = errors
            response_data["message"] += f" ({len(errors)} errors occurred)"
        
        return JSONResponse(
            status_code=200 if not errors else 207,  # 207 = Multi-Status
            content=response_data
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading files to dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")


@router.get("/dataset/label-patterns")
async def list_label_patterns():
    """Filename patterns that can supply labels without a CSV.

    Many speech corpora encode everything in the filename -- SAVEE's `DC_a01.wav`
    is speaker DC, anger, take 1 -- so for those a user needs no answer key at
    all, only to say which corpus this is.
    """
    return JSONResponse(status_code=200, content={"patterns": available_patterns()})


def _label_response(manager, dataset_name: str, extra: dict | None = None) -> JSONResponse:
    """Stored labels plus the preview of what the probe will do with them.

    The preview is the point.  Extraction is the expensive step of a probe run
    and training is not, so a user should discover "two of your three classes are
    about to be dropped" here, in a second, rather than after a multi-minute job.
    """
    rows = manager.get_dataset_files_as_csv_format(dataset_name)
    stored = manager.get_labels(dataset_name) or {}
    body = {
        "dataset_name": dataset_name,
        "source": stored.get("source"),
        "columns": manager.get_label_columns(dataset_name),
        "warnings": stored.get("warnings", []),
        "updated_at": stored.get("updated_at"),
        "preview": preview_dataset(rows),
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=200, content=body)


@router.get("/dataset/{dataset_name}/labels")
async def get_dataset_labels(request: Request, dataset_name: str):
    """Current answer key for a dataset, with a per-property preview."""
    session_id = get_session_id(request)
    manager = get_custom_dataset_manager(session_id)
    if not manager.get_dataset_metadata(dataset_name):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    return _label_response(manager, dataset_name)


@router.post("/dataset/{dataset_name}/labels")
async def upload_dataset_labels(
    request: Request,
    dataset_name: str,
    file: UploadFile = File(..., description="CSV with a filename column plus label columns"),
):
    """Attach an answer key from an uploaded CSV, joined on `filename`."""
    session_id = get_session_id(request)
    manager = get_custom_dataset_manager(session_id)
    if not manager.get_dataset_metadata(dataset_name):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    raw = await file.read()
    if len(raw) > MAX_LABEL_CSV_BYTES:
        raise HTTPException(status_code=413, detail="Label CSV is too large (limit 4 MB)")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Label CSV must be UTF-8 encoded")

    try:
        table, warnings = parse_labels_csv(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # How many of *this dataset's* files the CSV actually reaches. A CSV that
    # parses cleanly but matches nothing is the most likely user error here, and
    # it would otherwise show up as an empty property list with no explanation.
    known = {row["filename"] for row in manager.get_dataset_files_as_csv_format(dataset_name)}
    matched = sum(1 for name in known if name in table or Path(name).name in table)
    if matched == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "The CSV parsed but none of its filenames match this dataset. "
                f"Dataset files look like: {', '.join(sorted(known)[:3])}"
            ),
        )
    if matched < len(known):
        warnings.append(f"{len(known) - matched} of {len(known)} files have no row in the CSV")

    record = manager.set_labels(dataset_name, table, source="csv", warnings=warnings)
    return _label_response(manager, dataset_name, {"matched_files": matched, "stored": record["columns"]})


@router.post("/dataset/{dataset_name}/labels/derive")
async def derive_dataset_labels(
    request: Request,
    dataset_name: str,
    pattern_id: str = Form(..., description="Filename pattern id, e.g. 'savee'"),
):
    """Attach an answer key parsed out of the filenames themselves."""
    session_id = get_session_id(request)
    manager = get_custom_dataset_manager(session_id)
    if not manager.get_dataset_metadata(dataset_name):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    filenames = [row["filename"] for row in manager.get_dataset_files_as_csv_format(dataset_name)]
    try:
        table, warnings = derive_from_filenames(filenames, pattern_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record = manager.set_labels(dataset_name, table, source=f"pattern:{pattern_id}", warnings=warnings)
    matched = sum(1 for name in filenames if name in table)
    return _label_response(manager, dataset_name, {"matched_files": matched, "stored": record["columns"]})


@router.delete("/dataset/{dataset_name}/labels")
async def delete_dataset_labels(request: Request, dataset_name: str):
    """Remove the answer key, leaving the audio and the derived bands in place."""
    session_id = get_session_id(request)
    manager = get_custom_dataset_manager(session_id)
    if not manager.get_dataset_metadata(dataset_name):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    manager.clear_labels(dataset_name)
    return _label_response(manager, dataset_name)


@router.get("/dataset/list")
async def list_custom_datasets(request: Request):
    """List all custom datasets in the current session"""
    session_id = get_session_id(request)
    
    try:
        manager = get_custom_dataset_manager(session_id)
        datasets = manager.list_datasets()
        
        # Add formatted names for each dataset
        for dataset in datasets:
            dataset["formatted_name"] = format_custom_dataset_name(
                session_id, 
                dataset["dataset_name"]
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "session_id": session_id,
                "datasets": datasets,
                "total_datasets": len(datasets)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing datasets for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/dataset/{dataset_name}/metadata")
async def get_dataset_metadata(request: Request, dataset_name: str):
    """Get metadata for a specific custom dataset"""
    session_id = get_session_id(request)
    
    try:
        manager = get_custom_dataset_manager(session_id)
        metadata = manager.get_dataset_metadata(dataset_name)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        # Add formatted name
        metadata["formatted_name"] = format_custom_dataset_name(session_id, dataset_name)
        
        return JSONResponse(
            status_code=200,
            content=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata for dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset metadata: {str(e)}")


@router.get("/dataset/{dataset_name}/files")
async def list_dataset_files(request: Request, dataset_name: str):
    """List all files in a specific custom dataset"""
    session_id = get_session_id(request)
    
    try:
        manager = get_custom_dataset_manager(session_id)
        metadata = manager.get_dataset_metadata(dataset_name)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        formatted_name = format_custom_dataset_name(session_id, dataset_name)
        
        return JSONResponse(
            status_code=200,
            content={
                "dataset_name": formatted_name,
                "original_name": dataset_name,
                "files": metadata["files"],
                "total_files": metadata["total_files"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files for dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list dataset files: {str(e)}")


@router.delete("/dataset/{dataset_name}")
async def delete_custom_dataset(request: Request, dataset_name: str):
    """Delete a custom dataset and all its files"""
    session_id = get_session_id(request)
    
    try:
        manager = get_custom_dataset_manager(session_id)
        success = manager.delete_dataset(dataset_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Dataset '{dataset_name}' deleted successfully",
                "dataset_name": dataset_name,
                "session_id": session_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")


@router.get("/dataset/{dataset_name}/files/{filename}")
async def serve_dataset_file(
    request: Request, 
    dataset_name: str, 
    filename: str
):
    """Serve an audio file from a custom dataset"""
    session_id = get_session_id(request)
    
    try:
        manager = get_custom_dataset_manager(session_id)
        file_path = manager.resolve_file_path(dataset_name, filename)
        
        # Determine the correct media type based on file extension
        file_extension = file_path.suffix.lower()
        media_type_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac'
        }
        media_type = media_type_map.get(file_extension, 'audio/*')
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            headers={
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'public, max-age=3600',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                'Access-Control-Allow-Headers': 'Range, Accept-Encoding',
                'Content-Disposition': f'inline; filename="{filename}"'
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in dataset '{dataset_name}'")
    except Exception as e:
        logger.error(f"Error serving file {filename} from dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


@router.post("/dataset/cleanup")
async def cleanup_session(request: Request):
    """Clean up all datasets for the current session (for testing/debugging)"""
    session_id = get_session_id(request)
    
    try:
        success = cleanup_session_datasets(session_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Session cleanup {'successful' if success else 'failed'}",
                "session_id": session_id,
                "success": success
            }
        )
        
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")