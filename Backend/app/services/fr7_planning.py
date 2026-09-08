"""Dependency-free FR-7 planning helpers.

Safe to import from the API control plane, which does not install the
worker-only heavy dependencies (torch/librosa/jiwer/scikit-learn -- compare
requirements-api.txt to requirements-worker.txt, and the ROLE build arg in
Backend/Dockerfile). app/services/linguistic_acoustic_service.py imports
perturbation_service (torch/librosa) and sensitivity_metrics_service
(jiwer/scikit-learn) at module top, so schemas/analyses.py and
api/routes/analyses.py must import cost-estimation and task-resolution from
here instead, or importing them would crash the api-only container.
"""

from __future__ import annotations

from typing import Any

from app.core.model_catalog import ModelKind

MAX_GRID_VARIANTS = 60
STOCHASTIC_PROPERTIES = {"noise"}


def resolve_task(task: str, kind: ModelKind) -> str:
    if task != "auto":
        return task
    return "classification" if kind == ModelKind.AUDIO_CLASSIFICATION else "transcription"


def _sweep_variant_count(sweep: dict[str, Any]) -> int:
    repeats = int(sweep.get("repeats", 1)) if sweep["property"] in STOCHASTIC_PROPERTIES else 1
    return int(sweep["steps"]) * repeats


def estimate_cost(parameters: dict[str, Any], n_audio: int, model: str | None) -> tuple[int, float]:
    """Grid size and a rough wall-clock estimate, shown to the user before
    they submit (docs/FR7plan.md Part 1 S5.3 estimated_variants/estimated_seconds)."""
    per_audio = 1  # identity control
    if parameters.get("include_lexical_control", True):
        per_audio += 1
    for sweep in parameters.get("sweeps", []):
        per_audio += _sweep_variant_count(sweep)
    total = per_audio * max(n_audio, 1)
    per_variant_seconds = 3.5 if (model or "").startswith("whisper-large") else 1.5
    seconds = total * per_variant_seconds + per_audio * 0.3
    return total, round(seconds, 1)
