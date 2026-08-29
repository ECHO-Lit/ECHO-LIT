"""FR-7 orchestration: job status machine, TTL, caching, cancellation
(docs/FR7plan.md Part 1 S3, S8).

Exercises the service-layer pipeline (prepare_sweep -> render_and_register ->
mark_render_complete -> infer_variant -> aggregate_sweep) directly rather than
through Celery's eager mode: tasks.py's persistent per-worker event loop
(`_loop`) is intentionally incompatible with pytest-asyncio's per-test loop
(see the comment in app/worker/tasks.py), and the Celery task wrappers around
these functions are two-line _run() pass-throughs, so calling the async
functions directly exercises the exact same logic the tasks invoke.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core.storage import LocalObjectStorage
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.schemas.jobs import AudioAsset, JobOperation, JobProgress, JobRecord, JobStatus, TaskAudio, TaskEnvelope
from app.services import linguistic_acoustic_service as las

SR = 16_000
SESSION_ID = "test-session"


def _voiced(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 140.0 + 10.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = sum(0.15 / k * np.sin(k * phase) for k in range(1, 6))
    return (signal / np.max(np.abs(signal))).astype(np.float32)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch) -> LocalObjectStorage:
    store = LocalObjectStorage(tmp_path / "storage")
    monkeypatch.setattr(las, "get_storage", lambda: store)
    return store


@pytest.fixture
def fake_predict(monkeypatch):
    """Deterministic stand-in for a model adapter: the predicted 'transcript'
    is derived from the variant's own audio bytes, so byte-identical variants
    (invariant I4) yield identical predictions and hit the item cache, while
    any real perturbation yields a different (WER=1.0) transcript. This tests
    the pipeline's plumbing, not DSP-vs-metric correctness (covered in
    test_fr7_dsp_invariants.py / test_fr7_metrics.py)."""
    calls = {"count": 0}

    def _predict(model, model_spec, audio_path):
        calls["count"] += 1
        digest = hashlib.sha256(Path(audio_path).read_bytes()).hexdigest()[:8]
        return {"text": f"transcript {digest}"}

    monkeypatch.setattr(las, "_run_prediction", _predict)
    return calls


async def _seed_audio(storage: LocalObjectStorage) -> AudioAsset:
    tmp_wav = Path(storage.root) / "_seed.wav"
    sf.write(tmp_wav, _voiced(), SR)
    digest = hashlib.sha256(tmp_wav.read_bytes()).hexdigest()
    object_key = f"uploads/{SESSION_ID}/baseline.wav"
    storage.put_file(object_key, tmp_wav, "audio/wav")
    asset = AudioAsset(
        audio_id="audio-1", session_id=SESSION_ID, object_key=object_key,
        filename="baseline.wav", media_type="audio/wav",
        size_bytes=tmp_wav.stat().st_size, duration_seconds=2.0, sample_rate=SR,
        channels=1, sha256=digest, created_at=datetime.now(timezone.utc),
    )
    await AudioRepository().create(asset)
    return asset


def _envelope(job_id: str, asset: AudioAsset, parameters: dict) -> dict:
    envelope = TaskEnvelope(
        job_id=job_id, session_id=SESSION_ID, operation=JobOperation.linguistic_acoustic,
        model="whisper-base", model_spec=None,
        audio=[TaskAudio(audio_id=asset.audio_id, object_key=asset.object_key,
                         filename=asset.filename, media_type=asset.media_type, sha256=asset.sha256)],
        parameters=parameters, result_schema_version="v1", code_version="test",
    )
    return envelope.model_dump(mode="json")


async def _create_job(job_id: str, asset: AudioAsset, parameters: dict) -> None:
    now = datetime.now(timezone.utc)
    await JobRepository().create(JobRecord(
        job_id=job_id, session_id=SESSION_ID, operation=JobOperation.linguistic_acoustic,
        model="whisper-base", audio_ids=[asset.audio_id], parameters=parameters,
        progress=JobProgress(current=0, total=1, message="Queued"),
        created_at=now, updated_at=now,
    ))


_PARAMETERS = {
    "task": "transcription",
    "sweeps": [{"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 3}],
    "reference_transcript": None,
    "language": "en",
    "normalize_loudness": True,
    "include_lexical_control": True,
}


async def _run_full_pipeline(job_id: str, asset: AudioAsset, parameters: dict) -> dict:
    envelope = _envelope(job_id, asset, parameters)
    specs = await las.prepare_sweep(envelope, "celery-task-1")
    rendered = [await las.render_and_register(envelope, spec) for spec in specs]
    await las.mark_render_complete(envelope, rendered)
    applicable = [item for item in rendered if item.get("applicable")]
    outputs = [await las.infer_variant(envelope, item, "celery-task-2") for item in applicable]
    await las.aggregate_sweep(outputs, rendered, envelope)
    return envelope


@pytest.mark.asyncio
async def test_status_transitions_and_ttl(storage, fake_predict):
    asset = await _seed_audio(storage)
    job_id = "job-status"
    await _create_job(job_id, asset, _PARAMETERS)

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.queued

    await _run_full_pipeline(job_id, asset, _PARAMETERS)

    final = await JobRepository().get(job_id)
    assert final.status == JobStatus.success
    assert final.result_key

    from app.core import redis as redis_module
    ttl = await redis_module.job_redis.ttl(f"job:{job_id}")
    assert 86_000 < ttl <= 86_400

    result = storage.get_json(final.result_key)
    assert result["operation"] == "linguistic_acoustic"
    assert "profile" in result and "verdict" in result["profile"]
    expected_variants = len(las.expand_grid(_PARAMETERS["sweeps"], asset.sha256, True))
    assert result["metadata"]["variants_total"] == expected_variants


@pytest.mark.asyncio
async def test_grid_always_includes_declared_controls(storage, fake_predict):
    asset = await _seed_audio(storage)
    envelope = _envelope("job-grid", asset, _PARAMETERS)
    specs = await las.prepare_sweep(envelope, "celery-task")
    properties = {spec["property"] for spec in specs}
    assert "identity" in properties
    assert "time_mask" in properties  # lexical-destruction control (I5)
    identity_specs = [s for s in specs if s["property"] == "identity"]
    assert len(identity_specs) == 1


@pytest.mark.asyncio
async def test_repeated_sweep_hits_sweep_cache(storage, fake_predict):
    asset = await _seed_audio(storage)
    job_id_a = "job-cache-a"
    await _create_job(job_id_a, asset, _PARAMETERS)
    await _run_full_pipeline(job_id_a, asset, _PARAMETERS)
    calls_after_first_run = fake_predict["count"]
    assert calls_after_first_run > 0

    job_id_b = "job-cache-b"
    await _create_job(job_id_b, asset, _PARAMETERS)
    envelope_b = _envelope(job_id_b, asset, _PARAMETERS)
    hit = await las.complete_sweep_from_cache(envelope_b)
    assert hit is True

    record_b = await JobRepository().get(job_id_b)
    assert record_b.status == JobStatus.success
    assert record_b.cache_hit is True
    # No new model calls: the whole sweep was served from the session-neutral cache.
    assert fake_predict["count"] == calls_after_first_run


@pytest.mark.asyncio
async def test_repeated_variant_render_hits_item_cache(storage, fake_predict):
    """Invariant I4: re-rendering the same (baseline_sha, prop, theta, repeat)
    produces byte-identical audio, so the second inference is served from the
    per-item cache rather than calling the model again."""
    asset = await _seed_audio(storage)
    envelope = _envelope("job-item-cache", asset, _PARAMETERS)
    specs = await las.prepare_sweep(envelope, "t")
    pitch_spec = next(s for s in specs if s["property"] == "pitch")

    rendered_a = await las.render_and_register(envelope, pitch_spec)
    rendered_b = await las.render_and_register(envelope, pitch_spec)
    assert rendered_a["sha256"] == rendered_b["sha256"]

    out_a = await las.infer_variant(envelope, rendered_a, "t1")
    assert out_a["cache_hit"] is False
    out_b = await las.infer_variant(envelope, rendered_b, "t2")
    assert out_b["cache_hit"] is True
    assert out_a["output"] == out_b["output"]


@pytest.mark.asyncio
async def test_cancellation_stops_pipeline(storage, fake_predict):
    asset = await _seed_audio(storage)
    job_id = "job-cancel"
    await _create_job(job_id, asset, _PARAMETERS)
    envelope = _envelope(job_id, asset, _PARAMETERS)
    specs = await las.prepare_sweep(envelope, "t")

    await JobRepository().request_cancel(job_id)

    with pytest.raises(las.JobCancelled):
        await las.render_and_register(envelope, specs[0])

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.cancelled


@pytest.mark.asyncio
async def test_not_applicable_variant_excluded_but_job_succeeds(storage, fake_predict):
    asset = await _seed_audio(storage)
    parameters = {
        **_PARAMETERS,
        "sweeps": [{"property": "speed", "start": 0.01, "stop": 0.02, "steps": 2}],
    }
    job_id = "job-not-applicable"
    await _create_job(job_id, asset, parameters)
    envelope = await _run_full_pipeline(job_id, asset, parameters)

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.success
    result = storage.get_json(record.result_key)
    assert len(result["metadata"]["not_applicable"]) >= 1
