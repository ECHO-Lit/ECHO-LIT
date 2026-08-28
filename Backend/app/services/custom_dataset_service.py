import json
import logging
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
from app.core.audio_probe import probe_audio

logger = logging.getLogger(__name__)

# Base directory for session-based custom datasets
SESSIONS_BASE_DIR = Path("uploads/sessions")
GLOBAL_DATASETS_DIR = Path("uploads/sessions/_global_datasets")
MANIFEST_FILENAME_FIELDS = ("filename", "file", "filepath", "path")
MANIFEST_TRANSCRIPT_FIELDS = ("transcript", "sentence", "text", "statement")

class CustomDatasetManager:
    """Manages session-based custom datasets"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = SESSIONS_BASE_DIR / session_id
        self.datasets_dir = self.session_dir / "datasets"
        
        # Ensure directories exist
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self, dataset_name: str) -> Dict:
        """Create a new custom dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        
        if dataset_dir.exists():
            raise ValueError(f"Dataset '{dataset_name}' already exists in this session")
        
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            "dataset_name": dataset_name,
            "created_at": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "files": [],
            "total_files": 0
        }
        
        metadata_file = dataset_dir / "dataset_metadata.json"
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created custom dataset '{dataset_name}' for session {self.session_id}")
        return metadata
    
    def add_file_to_dataset(self, dataset_name: str, filename: str, file_data: bytes) -> Dict:
        """Add a file to an existing custom dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"
        
        if not dataset_dir.exists() or not metadata_file.exists():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        
        # Load existing metadata
        with metadata_file.open("r") as f:
            metadata = json.load(f)
        
        # Generate unique filename to avoid conflicts
        file_path = dataset_dir / filename
        counter = 1
        original_filename = filename
        name, ext = Path(filename).stem, Path(filename).suffix
        
        while file_path.exists():
            filename = f"{name}_{counter}{ext}"
            file_path = dataset_dir / filename
            counter += 1
        
        # Save the file
        with file_path.open("wb") as f:
            f.write(file_data)
        
        # Calculate audio metadata
        try:
            duration, sample_rate, _ = probe_audio(file_path)
        except Exception as e:
            logger.warning(f"Could not extract audio metadata for {filename}: {e}")
            duration = 0.0
            sample_rate = 0
        
        # Add file metadata
        file_metadata = {
            "filename": filename,
            "original_filename": original_filename,
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "size": file_path.stat().st_size,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        metadata["files"].append(file_metadata)
        metadata["total_files"] = len(metadata["files"])
        if metadata.get("manifest") and metadata.get("transcripts"):
            stored_files = {item["filename"] for item in metadata["files"]}
            unmatched = sorted(name for name in metadata["transcripts"] if name not in stored_files)
            metadata["manifest"]["matched_audio_count"] = len(metadata["transcripts"]) - len(unmatched)
            metadata["manifest"]["unmatched_filenames"] = unmatched[:20]
        
        # Save updated metadata
        with metadata_file.open("w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Added file '{filename}' to dataset '{dataset_name}' in session {self.session_id}")
        return file_metadata

    def add_manifest_to_dataset(self, dataset_name: str, filename: str, manifest_data: bytes) -> Dict:
        """Attach a CSV manifest with filename-to-reference-transcript pairs.

        The mapping is retained even when audio is uploaded later, so users may
        upload either the audio batch or the manifest first.  The J-Lens
        metadata view only exposes pairs whose audio is currently present.
        """
        dataset_dir = self.datasets_dir / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"
        if not dataset_dir.exists() or not metadata_file.exists():
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        if Path(filename).suffix.lower() != ".csv":
            raise ValueError("Dataset manifest must be a .csv file")
        if len(manifest_data) > 10 * 1024 * 1024:
            raise ValueError("Dataset manifest must be 10 MB or smaller")
        try:
            content = manifest_data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Dataset manifest must be UTF-8 encoded") from exc

        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            raise ValueError("Dataset manifest must include a header row")
        field_map = {field.strip().lower(): field for field in reader.fieldnames if field}
        filename_field = next((field_map[key] for key in MANIFEST_FILENAME_FIELDS if key in field_map), None)
        transcript_field = next((field_map[key] for key in MANIFEST_TRANSCRIPT_FIELDS if key in field_map), None)
        if not filename_field or not transcript_field:
            raise ValueError(
                "Dataset manifest needs a filename column (filename, file, filepath, or path) "
                "and a transcript column (transcript, sentence, text, or statement)"
            )

        transcripts: Dict[str, str] = {}
        blank_rows = 0
        for row in reader:
            raw_filename = str(row.get(filename_field) or "").strip()
            transcript = str(row.get(transcript_field) or "").strip()
            if not raw_filename or not transcript:
                blank_rows += 1
                continue
            stored_name = Path(raw_filename.replace("\\", "/")).name
            if stored_name in transcripts:
                raise ValueError(f"Dataset manifest contains duplicate filename: {stored_name}")
            transcripts[stored_name] = transcript
        if not transcripts:
            raise ValueError("Dataset manifest contains no filename/transcript pairs")

        with metadata_file.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        stored_files = {file_info["filename"] for file_info in metadata.get("files", [])}
        matched = sum(name in stored_files for name in transcripts)
        unmatched = sorted(name for name in transcripts if name not in stored_files)
        metadata["transcripts"] = transcripts
        metadata["manifest"] = {
            "filename": Path(filename).name,
            "uploaded_at": datetime.utcnow().isoformat(),
            "filename_field": filename_field,
            "transcript_field": transcript_field,
            "pair_count": len(transcripts),
            "matched_audio_count": matched,
            "unmatched_filenames": unmatched[:20],
        }
        with metadata_file.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return {
            "filename": Path(filename).name,
            "pair_count": len(transcripts),
            "matched_audio_count": matched,
            "unmatched_audio_count": len(unmatched),
            "blank_row_count": blank_rows,
            "filename_field": filename_field,
            "transcript_field": transcript_field,
        }
    
    def list_datasets(self) -> List[Dict]:
        """List all custom datasets in the session."""
        if not self.datasets_dir.exists():
            return []

        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "dataset_metadata.json"
                if metadata_file.exists():
                    try:
                        with metadata_file.open("r") as f:
                            metadata = json.load(f)
                            datasets.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata for dataset {dataset_dir.name}: {e}")

        return datasets

    def list_global_datasets(self) -> List[Dict]:
        """List globally-shared custom datasets (visible to every session).

        These are provisioned by the startup script into
        ``uploads/sessions/_global_datasets`` and are read-only.
        """
        if not GLOBAL_DATASETS_DIR.exists():
            return []

        datasets = []
        for dataset_dir in GLOBAL_DATASETS_DIR.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "dataset_metadata.json"
                if metadata_file.exists():
                    try:
                        with metadata_file.open("r") as f:
                            metadata = json.load(f)
                            metadata["session_id"] = "__global__"
                            metadata["_global"] = True
                            datasets.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read global dataset metadata {dataset_dir.name}: {e}")
        return datasets

    def resolve_global_file(self, dataset_name: str, filename: str) -> Path:
        """Resolve an audio file inside a global dataset."""
        dataset_dir = GLOBAL_DATASETS_DIR / dataset_name
        file_path = dataset_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found in global dataset '{dataset_name}'")
        return file_path

    def get_global_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        dataset_dir = GLOBAL_DATASETS_DIR / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"
        if not metadata_file.exists():
            return None
        try:
            with metadata_file.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read global dataset metadata {dataset_name}: {e}")
            return None
    
    def get_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Get metadata for a specific dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        metadata_file = dataset_dir / "dataset_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with metadata_file.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read metadata for dataset {dataset_name}: {e}")
            return None
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a custom dataset and all its files"""
        dataset_dir = self.datasets_dir / dataset_name
        
        if not dataset_dir.exists():
            return False
        
        # Remove all files in the dataset
        try:
            import shutil
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted custom dataset '{dataset_name}' from session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Could not delete dataset {dataset_name}: {e}")
            return False
    
    def resolve_file_path(self, dataset_name: str, filename: str) -> Path:
        """Resolve the full path to a file in a custom dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        file_path = dataset_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found in dataset '{dataset_name}'")
        
        return file_path
    
    def get_dataset_files_as_csv_format(self, dataset_name: str) -> List[Dict[str, str]]:
        """Get dataset files in the same format as global datasets (for compatibility)"""
        metadata = self.get_dataset_metadata(dataset_name)
        if not metadata:
            return []
        
        csv_format_files = []
        transcripts = metadata.get("transcripts", {})
        for file_info in metadata["files"]:
            csv_format_files.append({
                "filename": file_info["filename"],
                "duration": str(file_info["duration"]),
                "sample_rate": str(file_info.get("sample_rate", 0)),
                "size": str(file_info["size"]),
                "uploaded_at": file_info["uploaded_at"],
                "transcript": transcripts.get(file_info["filename"], ""),
            })
        
        return csv_format_files

    def get_global_dataset_files_as_csv_format(self, dataset_name: str) -> List[Dict[str, str]]:
        """Get global dataset files in CSV format."""
        metadata = self.get_global_dataset_metadata(dataset_name)
        if not metadata:
            return []
        csv_format_files = []
        transcripts = metadata.get("transcripts", {})
        for file_info in metadata["files"]:
            csv_format_files.append({
                "filename": file_info["filename"],
                "duration": str(file_info["duration"]),
                "sample_rate": str(file_info.get("sample_rate", 0)),
                "size": str(file_info["size"]),
                "uploaded_at": file_info["uploaded_at"],
                "transcript": transcripts.get(file_info["filename"], ""),
            })
        return csv_format_files


def get_custom_dataset_manager(session_id: str) -> CustomDatasetManager:
    """Factory function to create a CustomDatasetManager for a session"""
    return CustomDatasetManager(session_id)


def cleanup_session_datasets(session_id: str) -> bool:
    """Clean up all datasets for a session (called when session expires)"""
    session_dir = SESSIONS_BASE_DIR / session_id
    
    if not session_dir.exists():
        return True
    
    try:
        import shutil
        shutil.rmtree(session_dir)
        logger.info(f"Cleaned up session datasets for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Could not cleanup session {session_id}: {e}")
        return False


def is_custom_dataset(dataset_name: str) -> bool:
    """Check if a dataset name indicates a custom dataset"""
    return dataset_name.startswith("custom:")


def parse_custom_dataset_name(dataset_name: str) -> tuple[str, str]:
    """Parse custom dataset name format 'custom:session_id:dataset_name'"""
    if not is_custom_dataset(dataset_name):
        raise ValueError(f"Not a custom dataset name: {dataset_name}")
    
    parts = dataset_name.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid custom dataset name format: {dataset_name}")
    
    return parts[1], parts[2]  # session_id, dataset_name


def format_custom_dataset_name(session_id: str, dataset_name: str) -> str:
    """Format a custom dataset name for external use"""
    return f"custom:{session_id}:{dataset_name}"
