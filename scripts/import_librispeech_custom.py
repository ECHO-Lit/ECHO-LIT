#!/usr/bin/env python3
"""Import the downloaded LibriSpeech 1,000 dataset into a session's
custom dataset so it appears in the Manage Datasets dialog and J-Lens
sample selector without re-uploading files through the browser."""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/app/data/librispeech_1000")
SESSIONS_BASE = Path("/app/uploads/sessions")

if not DATA_DIR.is_dir():
    logger.error("LibriSpeech data not found at %s (are you in the container?)", DATA_DIR)
    sys.exit(1)

csv_path = DATA_DIR / "librispeech_1000_metadata.csv"
if not csv_path.is_file():
    logger.error("Metadata CSV not found at %s", csv_path)
    sys.exit(1)

# Collect all .flac files
flac_files = sorted(DATA_DIR.glob("*.flac"))
if not flac_files:
    logger.error("No .flac files found in %s", DATA_DIR)
    sys.exit(1)

logger.info("Found %d .flac files and metadata.csv", len(flac_files))

# Create the custom dataset directory inside the session
# We need the session_id to create a dataset. Let SESSION_ID be passed
# or create a fresh one if none is provided.
session_id = sys.argv[1] if len(sys.argv) > 1 else None
dataset_name = sys.argv[2] if len(sys.argv) > 2 else "librispeech-1000-custom"

if session_id:
    dataset_dir = SESSIONS_BASE / session_id / "datasets" / dataset_name
else:
    # create a new session dir and dataset
    import uuid
    session_id = uuid.uuid4().hex
    dataset_dir = SESSIONS_BASE / session_id / "datasets" / dataset_name

dataset_dir.mkdir(parents=True, exist_ok=True)
logger.info("Session: %s", session_id)
logger.info("Dataset directory: %s", dataset_dir)

# Import audio files - copy all FLACs and manifest into the dataset dir
for src in flac_files:
    dst = dataset_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        logger.info("  copied %s", src.name)
    else:
        logger.info("  exists %s (skipped)", src.name)

manifest_dst = dataset_dir / csv_path.name
if not manifest_dst.exists():
    shutil.copy2(csv_path, manifest_dst)
    logger.info("  copied %s", csv_path.name)
else:
    logger.info("  exists %s (skipped)", csv_path.name)

# Build metadata.json
import csv
import hashlib
from datetime import datetime, timezone

transcripts: dict[str, str] = {}
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get("filename", "")
        text = row.get("text", "")
        if name and text:
            transcripts[name] = text.lower()

files_metadata = []
for fpath in sorted(dataset_dir.glob("*.flac")):
    files_metadata.append({
        "filename": fpath.name,
        "original_filename": fpath.name,
        "duration": 0.0,
        "sample_rate": 0,
        "size": fpath.stat().st_size,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })

metadata = {
    "dataset_name": dataset_name,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "session_id": session_id,
    "files": files_metadata,
    "total_files": len(files_metadata),
    "transcripts": transcripts,
    "manifest": {
        "filename": "librispeech_1000_metadata.csv",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "filename_field": "filename",
        "transcript_field": "text",
        "pair_count": len(transcripts),
        "matched_audio_count": len(files_metadata),
        "unmatched_filenames": [],
    },
}

metadata_file = dataset_dir / "dataset_metadata.json"
with metadata_file.open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

logger.info("")
logger.info("=== Done ===")
logger.info("Session ID: %s", session_id)
logger.info("Dataset name: %s", dataset_name)
logger.info("Formatted name: custom:%s:%s", session_id, dataset_name)
logger.info("Files imported: %d", len(files_metadata))
logger.info("Transcript pairs: %d", len(transcripts))