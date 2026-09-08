"""FR-7: linguistic-vs-acoustic sensitivity sweep -- grid expansion, variant
rendering/registration, per-variant re-inference and profile aggregation
(docs/FR7plan.md Part 1 S2.7, S3, S4).

Functions here are called from Celery tasks in app.worker.tasks; each one
re-validates the JSON envelope independently since it crosses a message
boundary on every stage of the render -> infer -> aggregate pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.core import redis as redis_module
from app.core.model_catalog import MODEL_REVISIONS
from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.schemas.jobs import AudioAsset, JobProgress, JobStatus, TaskEnvelope
from app.services.perturbation_service import (
    PROPERTY_UNITS,
    TARGET_LUFS,
    VariantSpec,
    render_variant,
)
from app.services.sensitivity_metrics_service import asr_metrics, classification_metrics
from app.services.fr7_planning import (  # noqa: F401 -- re-exported for existing callers/tests
    MAX_GRID_VARIANTS,
    STOCHASTIC_PROPERTIES,
    estimate_cost,
    resolve_task,
)

logger = logging.getLogger(__name__)

_LN2 = math.log(2)
_IDENTITY_REFERENCE = {"pitch": 0.0, "speed": 1.0, "time_mask": 0.0, "freq_mask": 0.0}


class JobCancelled(Exception):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


async def _check_cancel(job_id: str, jobs: JobRepository) -> None:
    if await jobs.cancellation_requested(job_id):
        raise JobCancelled()


# ---------------------------------------------------------------------------
# Grid expansion (Part 1 S2.7): a UNION of per-property axes, not a product --
# FR-7 varies exactly one property at a time (invariant I2).
# ---------------------------------------------------------------------------

def _vid(baseline_sha256: str, prop: str, theta_key: Any, repeat: int) -> str:
    material = f"{baseline_sha256}|{prop}|{theta_key}|{repeat}".encode()
    return hashlib.sha256(material).hexdigest()[:16]


def expand_grid(
    sweeps: list[dict[str, Any]], baseline_sha256: str, include_lexical_control: bool = True
) -> list[VariantSpec]:
    specs = [VariantSpec(
        variant_id=_vid(baseline_sha256, "identity", 0.0, 0),
        property="identity", theta=0.0, repeat=0, is_control=True,
    )]
    seen_ids = {specs[0].variant_id}

    for sweep in sweeps:
        prop = sweep["property"]
        steps = int(sweep["steps"])
        repeats = int(sweep.get("repeats", 1)) if prop in STOCHASTIC_PROPERTIES else 1

        if prop == "freq_mask":
            band_low = float(sweep.get("band_low_hz", 0.0))
            widths = np.linspace(sweep["start"], sweep["stop"], steps)
            grid_values: list[Any] = [(band_low, band_low + float(width)) for width in widths]
        else:
            grid_values = [float(theta) for theta in np.linspace(sweep["start"], sweep["stop"], steps)]

        for theta in grid_values:
            theta_key = f"{theta[0]:.3f}-{theta[1]:.3f}" if isinstance(theta, tuple) else theta
            for repeat in range(repeats):
                variant_id = _vid(baseline_sha256, prop, theta_key, repeat)
                if variant_id in seen_ids:
                    continue
                seen_ids.add(variant_id)
                specs.append(VariantSpec(
                    variant_id=variant_id, property=prop, theta=theta,
                    repeat=repeat, is_control=False,
                ))

    if include_lexical_control:
        variant_id = _vid(baseline_sha256, "time_mask", 30.0, 0)
        if variant_id not in seen_ids:
            specs.append(VariantSpec(
                variant_id=variant_id, property="time_mask", theta=30.0,
                repeat=0, is_control=True,
            ))
    return specs


def spec_to_dict(spec: VariantSpec) -> dict[str, Any]:
    return asdict(spec)


def spec_from_dict(data: dict[str, Any]) -> VariantSpec:
    theta = data["theta"]
    if isinstance(theta, list):
        theta = tuple(theta)
    return VariantSpec(
        variant_id=data["variant_id"], property=data["property"], theta=theta,
        repeat=data["repeat"], is_control=data["is_control"],
    )


# ---------------------------------------------------------------------------
# Stage A: orchestration entry (grid expansion + status transition)
# ---------------------------------------------------------------------------

def sweep_cache_key(envelope: TaskEnvelope) -> str:
    """Session-neutral: keyed only on audio content + parameters, never on
    session_id/job_id, so an identical sweep from ANY session can reuse it."""
    material = {
        "operation": "linguistic_acoustic", "model": envelope.model,
        "revision": MODEL_REVISIONS.get(envelope.model or ""),
        "audio": sorted(asset.sha256 for asset in envelope.audio),
        "parameters": envelope.parameters,
        "schema": envelope.result_schema_version, "code": envelope.code_version,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def complete_sweep_from_cache(envelope_data: dict[str, Any]) -> bool:
    envelope = TaskEnvelope.model_validate(envelope_data)
    digest = sweep_cache_key(envelope)
    cached_key = await redis_module.redis.get(f"analysis-cache:{digest}")
    storage = get_storage()
    if not cached_key or not storage.exists(cached_key):
        return False
    payload = storage.get_json(cached_key)
    payload["job_id"] = envelope.job_id
    payload.setdefault("metadata", {})["cache_hit"] = True
    job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
    storage.put_json(job_result_key, payload)
    total = int(payload.get("metadata", {}).get("variants_total", 1))
    await JobRepository().update(
        envelope.job_id, status=JobStatus.success,
        progress=JobProgress(current=2 * total + 1, total=2 * total + 1, message="Completed from cache"),
        result_key=job_result_key, cache_hit=True,
    )
    await redis_module.job_redis.hincrby("metrics:jobs", "fr7_cache_hits", 1)
    await redis_module.job_redis.hincrby("metrics:jobs", "fr7_success", 1)
    return True


async def prepare_sweep(envelope_data: dict[str, Any], celery_task_id: str) -> list[dict[str, Any]]:
    """Expand the grid across every audio asset, transition
    QUEUED -> STARTED -> PROCESSING, and return JSON-safe render-task payloads."""
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    await jobs.update(envelope.job_id, status=JobStatus.started, task_id=celery_task_id)

    parameters = envelope.parameters
    sweeps = parameters.get("sweeps", [])
    include_control = parameters.get("include_lexical_control", True)

    all_specs: list[dict[str, Any]] = []
    for asset in envelope.audio:
        for spec in expand_grid(sweeps, asset.sha256, include_control):
            payload = spec_to_dict(spec)
            payload.update(
                audio_id=asset.audio_id, object_key=asset.object_key,
                filename=asset.filename, baseline_sha256=asset.sha256,
            )
            all_specs.append(payload)

    total = len(all_specs)
    await jobs.update(
        envelope.job_id, status=JobStatus.processing,
        progress=JobProgress(current=0, total=2 * total + 1, message=f"Rendering {total} variants"),
    )
    return all_specs


# ---------------------------------------------------------------------------
# Stage B: per-variant DSP render + registration as a session-owned AudioAsset
# ---------------------------------------------------------------------------

async def _advance_stage(job_id: str, counter_name: str, member: str, stage: str, message: str) -> None:
    """Idempotent progress counter: a Redis SET (not INCR), so a Celery
    redelivery under acks_late can never double-count (mirrors the existing
    job:{id}:completed-items pattern in worker.executor)."""
    key = f"job:{job_id}:{counter_name}"
    pipe = redis_module.job_redis.pipeline()
    pipe.sadd(key, member)
    pipe.expire(key, settings.JOB_TTL_SECONDS)
    await pipe.execute()
    count = await redis_module.job_redis.scard(key)
    record = await JobRepository().get(job_id)
    if not record:
        return
    total = record.progress.total
    n = max((total - 1) // 2, 1)
    current = count if stage == "render" else n + count
    await JobRepository().update(
        job_id, status=JobStatus.processing,
        progress=JobProgress(current=min(current, total), total=total, message=message),
    )


async def render_and_register(envelope_data: dict[str, Any], spec_data: dict[str, Any]) -> dict[str, Any]:
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    try:
        await _check_cancel(envelope.job_id, jobs)
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        raise

    spec = spec_from_dict(spec_data)
    storage = get_storage()

    with tempfile.TemporaryDirectory(prefix=f"fr7-render-{envelope.job_id}-") as temp_dir:
        suffix = Path(spec_data["filename"]).suffix or ".wav"
        local_baseline = Path(temp_dir) / f"baseline{suffix}"
        storage.download_file(spec_data["object_key"], local_baseline)
        rendered = render_variant(str(local_baseline), spec, spec_data["baseline_sha256"], temp_dir)

        result: dict[str, Any] = {**rendered, "source_audio_id": spec_data["audio_id"]}
        if not result.get("applicable"):
            await _advance_stage(
                envelope.job_id, "rendered", spec.variant_id, "render",
                f"Rendered variant {spec.variant_id} (not applicable)",
            )
            return result

        variant_path = Path(result["path"])
        variant_audio_id = uuid.uuid4().hex
        object_key = f"generated/{envelope.session_id}/{variant_audio_id}.wav"
        storage.put_file(object_key, variant_path, "audio/wav")
        asset = AudioAsset(
            audio_id=variant_audio_id, session_id=envelope.session_id, object_key=object_key,
            filename=f"{spec.property}_{spec.repeat}_{variant_audio_id[:8]}.wav",
            media_type="audio/wav", size_bytes=variant_path.stat().st_size,
            duration_seconds=float(result["duration_seconds"]), sample_rate=int(result["sample_rate"]),
            channels=None, sha256=result["sha256"], created_at=datetime.now(timezone.utc),
        )
        await AudioRepository().create(asset)
        result.update(
            variant_audio_id=variant_audio_id, object_key=object_key,
            playback_url=f"/audio/{variant_audio_id}",
        )

    await _advance_stage(
        envelope.job_id, "rendered", spec.variant_id, "render", f"Rendered variant {spec.variant_id}",
    )
    return result


async def mark_render_complete(envelope_data: dict[str, Any], rendered: list[dict[str, Any]]) -> None:
    envelope = TaskEnvelope.model_validate(envelope_data)
    record = await JobRepository().get(envelope.job_id)
    if not record:
        return
    applicable = sum(1 for item in rendered if item.get("applicable"))
    await JobRepository().update(
        envelope.job_id, status=JobStatus.processing,
        progress=JobProgress(
            current=record.progress.current, total=record.progress.total,
            message=f"Rendered {applicable} of {len(rendered)} variants; starting inference",
        ),
    )


# ---------------------------------------------------------------------------
# Stage D: per-variant re-inference (item-level cache-aware, mirrors
# worker.executor's per-file cache so a repeated sweep skips GPU work
# entirely thanks to invariant I4's deterministic rendering)
# ---------------------------------------------------------------------------

def _variant_item_cache_key(envelope: TaskEnvelope, variant_sha256: str) -> str:
    material = {
        "operation": "prediction", "model": envelope.model,
        "revision": MODEL_REVISIONS.get(envelope.model or ""),
        "audio": variant_sha256, "parameters": {},
        "schema": envelope.result_schema_version, "code": envelope.code_version,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _run_prediction(model: str, model_spec: Any, audio_path: str) -> Any:
    from app.worker.model_adapters import get_model_adapter
    from app.worker.model_registry import model_registry

    adapter = get_model_adapter(model, model_spec)
    if not adapter.supports("prediction"):
        raise ValueError(f"{model} does not support prediction")
    resource = model_registry.prepare(adapter, "prediction")
    return adapter.execute("prediction", audio_path, {}, resource)


async def infer_variant(
    envelope_data: dict[str, Any], rendered: dict[str, Any], celery_task_id: str
) -> dict[str, Any]:
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    try:
        await _check_cancel(envelope.job_id, jobs)
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        raise
    storage = get_storage()

    digest = _variant_item_cache_key(envelope, rendered["sha256"])
    cache_pointer = f"analysis-item-cache:{digest}"
    cached_key = await redis_module.redis.get(cache_pointer)
    if cached_key and storage.exists(cached_key):
        output = storage.get_json(cached_key)["result"]
        cache_hit = True
    else:
        with tempfile.TemporaryDirectory(prefix=f"fr7-infer-{envelope.job_id}-") as temp_dir:
            local_path = Path(temp_dir) / f"{rendered['variant_audio_id']}.wav"
            storage.download_file(rendered["object_key"], local_path)
            output = _jsonable(_run_prediction(envelope.model, envelope.model_spec, str(local_path)))
        result_key = f"cache-items/{digest}.json"
        storage.put_json(result_key, {"result": output})
        await redis_module.redis.set(cache_pointer, result_key, ex=settings.JOB_TTL_SECONDS)
        cache_hit = False

    await _advance_stage(
        envelope.job_id, "inferred", rendered["variant_id"], "infer",
        f"Inferred variant {rendered['variant_id']}",
    )
    return {**rendered, "output": output, "cache_hit": cache_hit, "task_id": celery_task_id}


# ---------------------------------------------------------------------------
# Metric extraction + sensitivity-profile aggregation (Part 1 S4.4)
# ---------------------------------------------------------------------------

def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return output.get("text") or output.get("transcript") or ""
    return ""


def _extract_posterior(output: Any) -> dict[str, float]:
    if isinstance(output, dict):
        if isinstance(output.get("probabilities"), dict):
            return output["probabilities"]
        if isinstance(output.get("scores"), dict):
            return output["scores"]
    return {}


def _degradation(
    task: str, baseline_output: Any, variant_output: Any,
    reference_transcript: str | None, language: str | None,
) -> tuple[float, dict[str, Any]]:
    if task == "classification":
        baseline_posterior = _extract_posterior(baseline_output)
        variant_posterior = _extract_posterior(variant_output)
        metrics = classification_metrics(variant_posterior, baseline_posterior)
        degradation = min(max(metrics["js_divergence"] / _LN2, 0.0), 1.0)
        return degradation, metrics
    baseline_text = _extract_text(baseline_output)
    variant_text = _extract_text(variant_output)
    reference = reference_transcript or baseline_text
    metrics = asr_metrics(variant_text, reference, language)
    return metrics["wer"], {**metrics, "transcript": variant_text}


def _identity_index(theta_sorted: np.ndarray, prop: str) -> int:
    if prop == "noise":
        return int(np.argmax(theta_sorted))  # highest SNR = cleanest = closest to "no perturbation"
    reference = _IDENTITY_REFERENCE.get(prop, 0.0)
    return int(np.argmin(np.abs(theta_sorted - reference)))


def _central_difference(theta: np.ndarray, d: np.ndarray, i0: int) -> float:
    if len(theta) < 2:
        return 0.0
    if i0 == 0:
        denom = theta[1] - theta[0]
        return float((d[1] - d[0]) / denom) if denom != 0 else 0.0
    if i0 == len(theta) - 1:
        denom = theta[i0] - theta[i0 - 1]
        return float((d[i0] - d[i0 - 1]) / denom) if denom != 0 else 0.0
    denom = theta[i0 + 1] - theta[i0 - 1]
    return float((d[i0 + 1] - d[i0 - 1]) / denom) if denom != 0 else 0.0


def _first_crossing(theta: np.ndarray, d: np.ndarray, level: float = 0.5) -> float | None:
    for i in range(len(d) - 1):
        d0, d1 = float(d[i]), float(d[i + 1])
        if d0 == level:
            return float(theta[i])
        if (d0 - level) * (d1 - level) < 0:
            t0, t1 = float(theta[i]), float(theta[i + 1])
            frac = (level - d0) / (d1 - d0)
            return float(t0 + frac * (t1 - t0))
    if abs(float(d[-1]) - level) < 1e-9:
        return float(theta[-1])
    return None


def _asymmetry(d: np.ndarray, i0: int) -> float | None:
    left, right = d[:i0], d[i0 + 1:]
    if len(left) < 1 or len(right) < 1:
        return None
    return float(np.mean(right) - np.mean(left))


def build_property_profile(points: list[dict[str, Any]], prop: str) -> dict[str, Any]:
    """One property's response curve d(theta) reduced to a rankable scalar
    (sensitivity_index), a local-fragility slope, and a failure threshold."""
    usable = [pt for pt in points if pt.get("applicable")]
    if len(usable) < 2:
        return {
            "property": prop, "applicable": False,
            "reason": "Fewer than two applicable grid points",
            "unit": PROPERTY_UNITS.get(prop, ""),
        }

    theta = np.array([pt["theta"] for pt in usable], dtype=float)
    order = np.argsort(theta)
    theta = theta[order]
    usable = [usable[i] for i in order]
    d = np.clip(np.array([pt["degradation"] for pt in usable], dtype=float), 0.0, 1.0)

    span = float(theta.max() - theta.min()) or 1.0
    u = (theta - theta.min()) / span
    sensitivity = float(np.trapz(d, u))

    i0 = _identity_index(theta, prop)
    slope = _central_difference(theta, d, i0)
    breakdown = _first_crossing(theta, d, level=0.5)
    monotonic = bool(np.all(np.diff(d[i0:]) >= -0.02)) if i0 < len(d) - 1 else True

    return {
        "property": prop, "applicable": True,
        "curve": [
            {
                "theta": float(t), "degradation": float(v),
                "ci95": pt.get("ci95"), "raw": pt.get("raw"),
                "variant_audio_id": pt.get("variant_audio_id"), "playback_url": pt.get("playback_url"),
                "measured_snr_db": pt.get("measured_snr_db"),
            }
            for t, v, pt in zip(theta, d, usable)
        ],
        "sensitivity_index": sensitivity,
        "local_slope_at_identity": slope,
        "breakdown_theta": breakdown,
        "asymmetry": _asymmetry(d, i0),
        "monotonic": monotonic,
        "unit": PROPERTY_UNITS.get(prop, ""),
    }


def _evidence_strings(ranked: list[dict[str, Any]], controls: dict[str, Any], task: str) -> list[str]:
    metric = "self-WER" if task == "transcription" else "output divergence"
    lines: list[str] = []
    for profile in ranked[:3]:
        unit = profile.get("unit", "")
        if profile["breakdown_theta"] is not None:
            lines.append(
                f"{profile['property']} crosses 50% {metric} near {profile['breakdown_theta']:.2f} {unit}."
            )
        else:
            lines.append(
                f"{profile['property']} never reaches 50% {metric} within the swept range "
                f"(sensitivity {profile['sensitivity_index']:.2f})."
            )
    ceiling = controls.get("lexical_destruction", {}).get("degradation")
    if ceiling is not None:
        lines.append(f"Removing ~30% of the words directly costs {ceiling:.2f} {metric} (reference ceiling).")
    return lines


def build_sensitivity_profile(
    profiles: list[dict[str, Any]], controls: dict[str, Any], task: str
) -> dict[str, Any]:
    active = [p for p in profiles if p.get("applicable")]
    if not active:
        return {
            "verdict": "inconclusive", "reason": "No property could be isolated for this input",
            "acoustic_influence": 0.0, "linguistic_robustness": 1.0,
            "relative_to_lexical_destruction": 0.0, "dominant_property": None,
            "ranking": [], "evidence": [],
        }

    ranked = sorted(active, key=lambda p: p["sensitivity_index"], reverse=True)
    top = ranked[0]
    acoustic_influence = float(top["sensitivity_index"])
    ceiling = max(float(controls.get("lexical_destruction", {}).get("degradation", 1.0)), 1e-6)
    relative = min(acoustic_influence / ceiling, 1.0)

    if acoustic_influence < 0.10:
        verdict = "linguistically_driven"
    elif relative >= 0.60:
        verdict = "acoustically_dominated"
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "acoustic_influence": acoustic_influence,
        "linguistic_robustness": float(1.0 - acoustic_influence),
        "relative_to_lexical_destruction": relative,
        "dominant_property": top["property"],
        "ranking": [
            {"property": p["property"], "sensitivity_index": p["sensitivity_index"],
             "breakdown_theta": p["breakdown_theta"]}
            for p in ranked
        ],
        "evidence": _evidence_strings(ranked, controls, task),
    }


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    if len(values) < 2:
        single = float(values[0]) if len(values) else 0.0
        return [single, single]
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _aggregate_repeats(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse stochastic-operator repeats (same theta, different seed) into
    one point with a mean, std and bootstrap 95% CI."""
    grouped: dict[float, list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(round(float(point["theta"]), 6), []).append(point)
    aggregated = []
    for theta, group in sorted(grouped.items()):
        values = np.array([g["degradation"] for g in group], dtype=float)
        entry: dict[str, Any] = {
            "theta": theta, "degradation": float(np.mean(values)), "applicable": True,
            "raw": group[0]["raw"],
            "variant_audio_id": group[0].get("variant_audio_id"),
            "playback_url": group[0].get("playback_url"),
            "measured_snr_db": group[0].get("measured_snr_db"),
        }
        if len(group) > 1:
            entry["degradation_std"] = float(np.std(values))
            entry["ci95"] = _bootstrap_ci(values)
        aggregated.append(entry)
    return aggregated


async def aggregate_sweep(
    outputs: list[dict[str, Any]], rendered: list[dict[str, Any]], envelope_data: dict[str, Any]
) -> None:
    started_at = time.monotonic()
    envelope = TaskEnvelope.model_validate(envelope_data)
    jobs = JobRepository()
    try:
        await _check_cancel(envelope.job_id, jobs)
    except JobCancelled:
        await jobs.update(envelope.job_id, status=JobStatus.cancelled)
        return

    task = envelope.parameters.get("task", "transcription")
    language = envelope.parameters.get("language", "en")
    reference_transcript = envelope.parameters.get("reference_transcript")

    outputs_by_id = {item["variant_id"]: item for item in outputs}
    merged: list[dict[str, Any]] = []
    for item in rendered:
        merged_item = dict(item)
        if item.get("applicable") and item["variant_id"] in outputs_by_id:
            merged_item.update(outputs_by_id[item["variant_id"]])
        merged.append(merged_item)

    by_audio: dict[str, list[dict[str, Any]]] = {}
    for item in merged:
        by_audio.setdefault(item["source_audio_id"], []).append(item)

    per_property_points: dict[str, list[dict[str, Any]]] = {}
    control_degradations: list[float] = []
    baseline_summary: dict[str, Any] | None = None
    not_applicable: list[dict[str, Any]] = []

    for audio_id, items in by_audio.items():
        baseline_item = next((i for i in items if i["property"] == "identity"), None)
        if baseline_item is None or not baseline_item.get("applicable") or "output" not in baseline_item:
            continue  # identity control itself failed -- this audio can't anchor a profile
        baseline_output = baseline_item["output"]

        for item in items:
            if item["property"] == "identity":
                continue
            if not item.get("applicable"):
                not_applicable.append({
                    "property": item["property"], "theta": item["theta"], "reason": item.get("reason"),
                })
                continue
            if "output" not in item:
                continue  # inference failed for this variant; excluded from the curve
            degradation, raw = _degradation(
                task, baseline_output, item["output"], reference_transcript, language
            )
            theta = item["theta"]
            theta_scalar = (theta[1] - theta[0]) if isinstance(theta, (list, tuple)) else float(theta)
            point = {
                "theta": theta_scalar, "degradation": degradation, "applicable": True, "raw": raw,
                "variant_audio_id": item.get("variant_audio_id"), "playback_url": item.get("playback_url"),
                "measured_snr_db": item.get("measured_snr_db"),
            }
            if item.get("is_control") and item["property"] == "time_mask":
                control_degradations.append(degradation)
            else:
                per_property_points.setdefault(item["property"], []).append(point)

        if baseline_summary is None:
            baseline_summary = {
                "audio_id": audio_id,
                "transcript": _extract_text(baseline_output) if task == "transcription" else None,
                "reference_source": "user_supplied" if reference_transcript else "self_baseline",
                "stable": True,
            }

    profiles = [
        build_property_profile(_aggregate_repeats(points), prop)
        for prop, points in per_property_points.items()
    ]
    lexical_degradation = float(np.mean(control_degradations)) if control_degradations else 0.0
    controls = {"lexical_destruction": {"degradation": lexical_degradation, "theta": 30.0}}
    sensitivity = build_sensitivity_profile(profiles, controls, task)

    cached_count = sum(1 for item in outputs if item.get("cache_hit"))
    result: dict[str, Any] = {
        "job_id": envelope.job_id, "operation": "linguistic_acoustic",
        "model": envelope.model, "task": task,
        "baseline": baseline_summary or {
            "audio_id": None, "transcript": None, "reference_source": "self_baseline", "stable": False,
        },
        "profile": sensitivity,
        "properties": profiles,
        "controls": controls,
        "set_level": {
            "macro_f1": None,
            "macro_f1_unavailable": "requires per-item ground-truth labels, not collected by this request",
        },
        "metadata": {
            "result_schema_version": envelope.result_schema_version,
            "code_version": envelope.code_version,
            "model_revision": MODEL_REVISIONS.get(envelope.model or ""),
            "variants_total": len(rendered),
            "variants_cached": cached_count,
            "cache_hit_rate": (cached_count / len(outputs)) if outputs else 0.0,
            "queue_latency_seconds": 0.0,
            "execution_seconds": time.monotonic() - started_at,
            "normalizer": "EnglishTextNormalizer" if (language or "en").startswith("en") else "BasicTextNormalizer",
            "target_lufs": TARGET_LUFS,
            "not_applicable": not_applicable,
            "variant_playback_available": True,
            "baseline_unstable": False,
        },
    }

    storage = get_storage()
    job_result_key = f"results/{envelope.session_id}/{envelope.job_id}/result.json"
    storage.put_json(job_result_key, _jsonable(result))

    # Session-neutral cache copy: strip session-owned audio references so a
    # repeated identical sweep from ANY session can reuse the metrics/profile.
    cache_copy = json.loads(json.dumps(_jsonable(result)))
    cache_copy["baseline"]["audio_id"] = None
    for prop_profile in cache_copy["properties"]:
        for point in prop_profile.get("curve", []):
            point.pop("variant_audio_id", None)
            point.pop("playback_url", None)
    cache_copy["metadata"]["variant_playback_available"] = False
    cache_digest = sweep_cache_key(envelope)
    cache_result_key = f"cache/{cache_digest}/result.json"
    storage.put_json(cache_result_key, cache_copy)
    await redis_module.redis.set(f"analysis-cache:{cache_digest}", cache_result_key, ex=settings.JOB_TTL_SECONDS)

    total = len(rendered)
    await jobs.update(
        envelope.job_id, status=JobStatus.success,
        progress=JobProgress(current=2 * total + 1, total=2 * total + 1, message="Completed"),
        result_key=job_result_key,
    )
    await redis_module.job_redis.delete(f"job:{envelope.job_id}:rendered", f"job:{envelope.job_id}:inferred")
    await redis_module.job_redis.hincrby("metrics:jobs", "fr7_success", 1)
    logger.info(
        "fr7_completed job_id=%s execution_seconds=%.3f cache_hit_rate=%.2f",
        envelope.job_id, time.monotonic() - started_at, result["metadata"]["cache_hit_rate"],
    )
