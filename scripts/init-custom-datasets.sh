#!/bin/bash
# Runs inside the echo-api container on every start.
# Idempotently imports the LibriSpeech 1,000 dataset as a custom dataset
# for any session that later loads the Manage Datasets dialog.

set -euo pipefail

SOURCE_DIR="/app/data/librispeech_1000"
DEST_PARENT="/app/uploads/sessions/_global_datasets"
DATASET_NAME="librispeech-1000-global"

# Only run if the source data exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "[init-custom-datasets] Source $SOURCE_DIR not found — skipping"
    exit 0
fi

# Create if needed
mkdir -p "$DEST_PARENT"

# Import the FLACs and write metadata.json directly.
# The backend gets this dataset via a sidecar loading path we'll add.
DST="$DEST_PARENT/$DATASET_NAME"
mkdir -p "$DST"

# Copy FLACs (only missing files for speed on subsequent boots)
for f in "$SOURCE_DIR"/*.flac; do
    name=$(basename "$f")
    if [ ! -f "$DST/$name" ]; then
        cp "$f" "$DST/$name"
    fi
done

# Copy the manifest
if [ ! -f "$DST/librispeech_1000_metadata.csv" ]; then
    cp "$SOURCE_DIR/librispeech_1000_metadata.csv" "$DST/"
fi

# Write metadata.json
python3 -c "
import csv, json, hashlib
from pathlib import Path
from datetime import datetime, timezone

src_csv = Path('/app/data/librispeech_1000/librispeech_1000_metadata.csv')
dst_dir = Path('$DST')

transcripts = {}
with src_csv.open('r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        name = row.get('filename', '')
        text = row.get('text', '')
        if name and text:
            transcripts[name] = text.lower()

flacs = sorted(dst_dir.glob('*.flac'))
files = []
for fp in flacs:
    files.append({
        'filename': fp.name,
        'original_filename': fp.name,
        'duration': 0.0,
        'sample_rate': 0,
        'size': fp.stat().st_size,
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
    })

meta = {
    'dataset_name': 'librispeech-1000-global',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'session_id': '_global',
    'files': files,
    'total_files': len(files),
    'transcripts': transcripts,
    'manifest': {
        'filename': 'librispeech_1000_metadata.csv',
        'uploaded_at': datetime.now(timezone.utc).isoformat(),
        'filename_field': 'filename',
        'transcript_field': 'text',
        'pair_count': len(transcripts),
        'matched_audio_count': len(files),
        'unmatched_filenames': [],
    },
}

(dst_dir / 'dataset_metadata.json').write_text(json.dumps(meta, indent=2))
print(f'[init-custom-datasets] Imported {len(files)} files, {len(transcripts)} transcript pairs')
print(f'[init-custom-datasets] Dataset ready at {dst_dir}')
"