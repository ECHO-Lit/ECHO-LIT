from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from app.services.dataset_service import load_metadata


def _basename(row: dict) -> Optional[str]:
    raw = row.get("path") or row.get("filepath") or row.get("file") or row.get("filename")
    return Path(str(raw)).name if raw else None


def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
        return f if f == f else None  # filter NaN
    except (TypeError, ValueError):
        return None


def _histogram(values: list[float], bins: int = 20) -> dict:
    if not values:
        return {"histogram": [], "bins": []}
    arr = np.array(values, dtype=float)
    hist, edges = np.histogram(arr, bins=min(bins, max(1, len(arr))))
    return {"histogram": hist.tolist(), "bins": edges.tolist()}


def compute_metadata_eda(dataset: str, session_id: Optional[str] = None) -> dict:
    """Compute dataset-wide EDA from metadata rows only (no audio decoding)."""
    rows = load_metadata(dataset, session_id)

    durations = [d for d in (_to_float(r.get("duration")) for r in rows) if d is not None and d > 0]

    labels = [r.get("emotion") or r.get("label") for r in rows]
    labels = [l for l in labels if l]

    sentences = [r.get("sentence") or r.get("transcript") or r.get("text") or "" for r in rows]
    word_counts = [len(s.split()) for s in sentences if s]

    sample_rates = [r.get("sample_rate") or r.get("samplerate") for r in rows]
    sample_rates = [str(sr) for sr in sample_rates if sr]

    # Keyed by basename so the frontend can join against acoustic-EDA results,
    # whose `individual_analyses[].filename` is always a plain basename
    # (materialized via /audio/materialize before the audio_features job runs).
    labels_by_file = {
        base: (r.get("emotion") or r.get("label"))
        for r in rows
        if (base := _basename(r)) and (r.get("emotion") or r.get("label"))
    }
    durations_by_file = {
        base: d
        for r in rows
        if (base := _basename(r)) and (d := _to_float(r.get("duration"))) is not None and d > 0
    }

    return {
        "dataset": dataset,
        "summary": {
            "total_files": len(rows),
            "total_hours": round(sum(durations) / 3600, 3) if durations else 0.0,
            "mean_duration": round(float(np.mean(durations)), 3) if durations else 0.0,
            "median_duration": round(float(np.median(durations)), 3) if durations else 0.0,
            "num_classes": len(set(labels)),
        },
        "duration_histogram": _histogram(durations),
        "class_balance": dict(Counter(labels)),
        "transcript_length_histogram": _histogram(word_counts),
        "sample_rate_breakdown": dict(Counter(sample_rates)),
        "labels_by_file": labels_by_file,
        "durations_by_file": durations_by_file,
    }
