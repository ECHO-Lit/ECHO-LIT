"""FR-7 end-to-end acceptance -- the SRS criterion, run against a real model.

Test Plan Section 3.1.2.  Closes the one item in docs/FR7plan.md Part 1 S8 that
its 12 planned assertions never covered (FR7plan.md:1547).

SRS FR-7 acceptance criterion: "Varying pitch/speed while words are unchanged
reveals whether the model's output is driven by acoustic properties, quantified
by the sensitivity profile."

Every other FR-7 test substitutes a deterministic stand-in for the model, which
is correct for testing the pipeline but cannot test this: the criterion is a
claim about a real ASR model's behaviour under real DSP.  So this one downloads
and runs whisper-base.

It is opt-in and skipped by default -- it needs torch, a model download and
roughly a minute of compute, none of which belong in a run that is expected to
finish in seconds:

    ECHO_MODEL_TESTS=1 pytest tests/test_fr7_acceptance.py

The guard is an environment variable rather than a network probe, because a
probe would make the decision to run depend on connectivity, which is precisely
the kind of nondeterminism a test suite should not have.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core.storage import LocalObjectStorage
from app.repositories.audio import AudioRepository
from app.repositories.jobs import JobRepository
from app.schemas.jobs import (
    AudioAsset,
    JobOperation,
    JobProgress,
    JobRecord,
    TaskAudio,
    TaskEnvelope,
)
from app.services import linguistic_acoustic_service as las

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv("ECHO_MODEL_TESTS"),
        reason="Real-model acceptance test; set ECHO_MODEL_TESTS=1 to run",
    ),
    pytest.mark.skipif(find_spec("torch") is None, reason="torch is not installed"),
]

SR = 16_000
SESSION_ID = "fr7-acceptance"


def _speech_like(duration: float = 3.0) -> np.ndarray:
    """A voiced, formant-shaped signal.

    Not real speech, but pitched and harmonically structured, so pitch and speed
    operators have something meaningful to act on and Whisper produces a stable
    (if nonsense) transcript for the unperturbed baseline.
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    f0 = 130.0 + 12.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    signal = sum((0.4 / k) * np.sin(k * phase) for k in range(1, 12))
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / duration, 0, 1)))
    signal = signal * (0.3 + 0.7 * envelope)
    return (signal / np.max(np.abs(signal))).astype(np.float32)


@pytest.fixture
def storage(tmp_path, monkeypatch) -> LocalObjectStorage:
    store = LocalObjectStorage(tmp_path / "storage")
    monkeypatch.setattr(las, "get_storage", lambda: store)
    return store


async def _seed_audio(store: LocalObjectStorage) -> AudioAsset:
    wav = Path(store.root) / "_baseline.wav"
    sf.write(wav, _speech_like(), SR)
    object_key = f"uploads/{SESSION_ID}/baseline.wav"
    store.put_file(object_key, wav, "audio/wav")
    asset = AudioAsset(
        audio_id="audio-1",
        session_id=SESSION_ID,
        object_key=object_key,
        filename="baseline.wav",
        media_type="audio/wav",
        size_bytes=wav.stat().st_size,
        duration_seconds=3.0,
        sample_rate=SR,
        channels=1,
        sha256=hashlib.sha256(wav.read_bytes()).hexdigest(),
        created_at=datetime.now(timezone.utc),
    )
    await AudioRepository().create(asset)
    return asset


PARAMETERS = {
    "task": "transcription",
    "sweeps": [
        {"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 5},
        {"property": "speed", "start": 0.7, "stop": 1.4, "steps": 5},
    ],
    "reference_transcript": None,
    "language": "en",
    "normalize_loudness": True,
    # The control is the whole point: it destroys the words while leaving the
    # acoustics comparable, giving an upper bound to measure against.
    "include_lexical_control": True,
}


async def test_pitch_and_speed_sensitivity_stay_below_the_lexical_control(storage):
    """FR-7 SRS acceptance criterion, end to end on whisper-base.

    The claim being verified: perturbing *how* something sounds, while leaving
    *what* was said intact, must degrade the transcript less than destroying the
    words themselves.  If pitch or speed sensitivity met or exceeded the lexical
    control, the sensitivity profile would be measuring noise rather than
    acoustic dependence, and every FR-7 verdict built on it would be unfounded.
    """
    asset = await _seed_audio(storage)
    job_id = "fr7-acceptance-job"
    now = datetime.now(timezone.utc)
    await JobRepository().create(JobRecord(
        job_id=job_id, session_id=SESSION_ID, operation=JobOperation.linguistic_acoustic,
        model="whisper-base", audio_ids=[asset.audio_id], parameters=PARAMETERS,
        progress=JobProgress(current=0, total=1, message="Queued"),
        created_at=now, updated_at=now,
    ))
    envelope = TaskEnvelope(
        job_id=job_id, session_id=SESSION_ID, operation=JobOperation.linguistic_acoustic,
        model="whisper-base", model_spec=None,
        audio=[TaskAudio(
            audio_id=asset.audio_id, object_key=asset.object_key, filename=asset.filename,
            media_type=asset.media_type, sha256=asset.sha256,
        )],
        parameters=PARAMETERS, result_schema_version="v1", code_version="acceptance",
    ).model_dump(mode="json")

    # The same service pipeline the Celery tasks drive, called directly: the task
    # wrappers are two-line pass-throughs and the worker's persistent event loop
    # is incompatible with pytest-asyncio's per-test loop.
    specs = await las.prepare_sweep(envelope, "celery-task-1")
    rendered = [await las.render_and_register(envelope, spec) for spec in specs]
    await las.mark_render_complete(envelope, rendered)
    applicable = [item for item in rendered if item.get("applicable")]
    outputs = [await las.infer_variant(envelope, item, "celery-task-2") for item in applicable]
    await las.aggregate_sweep(outputs, rendered, envelope)

    record = await JobRepository().get(job_id)
    result = storage.get_json(record.result_key)
    profile = result["sensitivity_profile"]
    controls = profile["controls"]

    lexical = controls["lexical_destruction"]["degradation"]
    assert lexical > 0, "the control must actually destroy the transcript"

    for prop in ("pitch", "speed"):
        measured = profile["properties"][prop]
        assert measured["applicable"], f"{prop} sweep produced no usable variants"
        assert measured["sensitivity_index"] < lexical, (
            f"{prop} sensitivity {measured['sensitivity_index']} is not below the "
            f"lexical-destruction control {lexical}; the profile is not separating "
            "acoustic sensitivity from loss of the words"
        )

    # The profile must reach a stated conclusion with evidence, not just numbers.
    assert profile["verdict"] in {
        "linguistically_driven", "acoustically_dominated", "mixed", "inconclusive"
    }
    assert profile["evidence"]
