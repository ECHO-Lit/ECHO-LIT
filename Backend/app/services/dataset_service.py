from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
import csv
import logging
from app.core.audio_probe import probe_audio
from .custom_dataset_service import (
    get_custom_dataset_manager, 
    is_custom_dataset, 
    parse_custom_dataset_name
)

logger = logging.getLogger(__name__)

# Parsed rows for the bundled datasets, keyed by dataset name -> (csv mtime, rows).
#
# RAVDESS's CSV has no `duration` column, so every row triggers a probe_audio()
# call that opens the audio file. That is ~144 file reads per request, paid again
# by /{dataset}/metadata and /{dataset}/eda. The bundled datasets are read-only,
# so parse once and reuse; the mtime key means editing a CSV still invalidates.
# Custom datasets are per-session and mutable, and are deliberately NOT cached.
#
# Callers get the cached list itself, so they must treat it as read-only.
_metadata_cache: Dict[str, tuple[float, List[Dict[str, str]]]] = {}


# Resolve paths relative to repo structure: Backend/data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# CSV metadata files per dataset
DATASET_PATHS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv",
    "cv-valid-dev": DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv",
    "ravdess": DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv",
    "l2-arctic": DATA_DIR / "L2_ARCTIC_dataset" / "l2_metadata.csv",
    "saa": DATA_DIR / "SAA_dataset" / "saa_metadata.csv",
}

# Base directories for dataset audio files
DATASET_BASE_DIRS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev",
    "cv-valid-dev": DATA_DIR / "common_voice_valid_dev",
    "ravdess": DATA_DIR / "ravdess_subset",
    "l2-arctic": DATA_DIR / "L2_ARCTIC_dataset" / "audio",
    "saa": DATA_DIR / "SAA_dataset" / "audio",
}

# SAA's `filename` column has no extension (e.g. "arabic1"); the audio files on
# disk are .mp3. Datasets needing this normalization go here.
DATASETS_WITH_EXTENSIONLESS_FILENAMES: Dict[str, str] = {
    "saa": ".mp3",
}


def calculate_audio_duration(audio_path: Path) -> float:
    """Calculate duration of audio file in seconds"""
    try:
        duration, _, _ = probe_audio(audio_path)
        return round(duration, 2)
    except Exception as e:
        logger.warning(f"Could not calculate duration for {audio_path}: {e}")
        return 0.0

def load_metadata(dataset: str, session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Load metadata for both global and custom datasets"""
    
    # Handle custom datasets
    if is_custom_dataset(dataset):
        if not session_id:
            raise ValueError("session_id is required for custom datasets")
        
        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
        logger.info(f"Custom dataset metadata: session_id_from_name='{session_id_from_name}', current_session_id='{session_id}'")
        if session_id_from_name != session_id:
            logger.warning(f"Session ID mismatch in metadata: dataset has '{session_id_from_name}' but request has '{session_id}'")
            # Use the dataset's session ID instead
            manager = get_custom_dataset_manager(session_id_from_name)
            return manager.get_dataset_files_as_csv_format(dataset_name)
        
        manager = get_custom_dataset_manager(session_id)
        return manager.get_dataset_files_as_csv_format(dataset_name)
    
    # Handle global datasets (existing logic)
    ds = dataset.lower()
    if ds not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")

    csv_path = DATASET_PATHS[ds]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found for: {dataset}")

    csv_mtime = csv_path.stat().st_mtime
    cached = _metadata_cache.get(ds)
    if cached and cached[0] == csv_mtime:
        return cached[1]

    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys to lowercase; strip whitespace
                normalized = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

                ext = DATASETS_WITH_EXTENSIONLESS_FILENAMES.get(ds)
                if ext and normalized.get("filename") and "." not in normalized["filename"]:
                    normalized["filename"] += ext

                rows.append(normalized)
    except Exception:
        # Re-raise to let the route map to a 500
        raise

    # Datasets whose CSV lacks a duration column (RAVDESS) need every file probed.
    # Serially that is ~13s for 144 files; probe_audio is I/O-bound, so run the
    # batch on a thread pool instead.
    needs_duration = [r for r in rows if not r.get("duration")]
    if needs_duration:
        base_dir = DATASET_BASE_DIRS.get(ds)

        def _duration_for(row: Dict[str, str]) -> str:
            filename = row.get("filename", "")
            if not filename or base_dir is None:
                return "0.0"
            audio_path = base_dir / filename
            if not audio_path.exists():
                return "0.0"
            # calculate_audio_duration already swallows and logs probe failures.
            return str(calculate_audio_duration(audio_path))

        with ThreadPoolExecutor(max_workers=min(16, len(needs_duration))) as pool:
            for row, duration in zip(needs_duration, pool.map(_duration_for, needs_duration)):
                row["duration"] = duration

    _metadata_cache[ds] = (csv_mtime, rows)
    return rows


def resolve_file(dataset: str, file_path: str, session_id: Optional[str] = None) -> Path:
    """Resolve file path for both global and custom datasets"""
    
    # Handle custom datasets
    if is_custom_dataset(dataset):
        if not session_id:
            raise ValueError("session_id is required for custom datasets")
        
        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
        logger.info(f"Custom dataset: session_id_from_name='{session_id_from_name}', current_session_id='{session_id}'")
        if session_id_from_name != session_id:
            logger.warning(f"Session ID mismatch: dataset has '{session_id_from_name}' but request has '{session_id}'")
            # For debugging, let's check if the file exists with the dataset's session ID
            manager = get_custom_dataset_manager(session_id_from_name)
            try:
                return manager.resolve_file_path(dataset_name, file_path)
            except Exception as e:
                logger.error(f"Could not resolve file with dataset session ID: {e}")
                raise ValueError(f"Session ID mismatch for custom dataset. Dataset session: {session_id_from_name}, Request session: {session_id}")
        
        manager = get_custom_dataset_manager(session_id)
        return manager.resolve_file_path(dataset_name, file_path)
    
    # Handle global datasets (existing logic)
    ds = dataset.lower()
    if ds not in DATASET_BASE_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}")

    base_dir = DATASET_BASE_DIRS[ds]
    safe_name = Path(file_path).name
    audio_path = base_dir / safe_name
    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {safe_name}")

    return audio_path


def media_type_for(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    media_type_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }
    media_type = media_type_map.get(ext)
    if media_type is None:
        raise ValueError(f"Unsupported media type for extension: {ext}")
    return media_type
