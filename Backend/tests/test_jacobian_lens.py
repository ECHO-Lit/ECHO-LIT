from types import SimpleNamespace

import pytest


def test_jacobian_lens_job_contract_requires_paired_owned_audio():
    from pydantic import ValidationError
    from app.schemas.jobs import JobCreateRequest

    request = JobCreateRequest(
        operation="jacobian_lens_fit",
        model="whisper-base",
        audio_ids=["audio-a", "audio-b"],
        parameters={
            "samples": [
                {"audio_id": "audio-a", "transcript": "first sample"},
                {"audio_id": "audio-b", "transcript": "second sample"},
            ],
        },
    )
    assert request.parameters["probe_count"] == 4
    assert request.parameters["samples"][0]["transcript"] == "first sample"

    with pytest.raises(ValidationError):
        JobCreateRequest(
            operation="jacobian_lens_fit",
            model="whisper-base",
            audio_ids=["audio-a"],
            parameters={
                "samples": [
                    {"audio_id": "audio-a", "transcript": "one"},
                    {"audio_id": "audio-b", "transcript": "two"},
                ],
            },
        )


def test_speech_adapters_expose_jacobian_lens_architecture():
    from app.core.model_catalog import ModelKind
    from app.schemas.jobs import RuntimeModelSpec
    from app.worker.model_adapters import get_model_adapter

    assert get_model_adapter("whisper-base").jacobian_lens_architecture() == "seq2seq"
    assert get_model_adapter("wav2vec2").jacobian_lens_architecture() is None
    ctc = get_model_adapter(
        "custom-ctc",
        RuntimeModelSpec(
            hf_repo="org/ctc-model",
            kind=ModelKind.CTC_ASR,
            capabilities=["prediction", "jacobian_lens_fit", "jacobian_lens_apply"],
        ),
    )
    assert ctc.jacobian_lens_architecture() == "ctc"
    assert ctc.jacobian_lens_revision() == "org/ctc-model@main"


@pytest.mark.asyncio
async def test_jacobian_lens_repository_is_session_owned():
    from datetime import datetime, timezone

    from app.repositories.jacobian_lenses import JacobianLensRepository
    from app.schemas.jacobian_lens import JacobianLensRecord

    repository = JacobianLensRepository()
    now = datetime.now(timezone.utc)
    record = JacobianLensRecord(
        lens_id="jlens-test",
        session_id="session-a",
        model_id="whisper-base",
        model_revision="openai/whisper-base",
        fit_job_id="job-a",
        created_at=now,
        updated_at=now,
        sample_count=2,
    )
    await repository.create(record)
    assert await repository.get_owned("jlens-test", "session-a") == record
    assert await repository.get_owned("jlens-test", "session-b") is None


@pytest.mark.asyncio
async def test_fit_job_creates_session_owned_lens_and_uses_single_worker_job(client, sample_audio_file, monkeypatch):
    from app.api.routes import jobs as jobs_routes
    from app.repositories.jobs import JobRepository

    audio_ids = []
    for name in ("first.wav", "second.wav"):
        with sample_audio_file.open("rb") as handle:
            upload = await client.post("/upload", files={"file": (name, handle, "audio/wav")})
        assert upload.status_code == 201
        audio_ids.append(upload.json()["audio_id"])

    sent = []
    monkeypatch.setattr(
        jobs_routes.celery_app,
        "send_task",
        lambda *args, **kwargs: sent.append((args, kwargs)) or SimpleNamespace(id="fit-task"),
    )
    response = await client.post("/jobs", json={
        "operation": "jacobian_lens_fit",
        "model": "whisper-base",
        "audio_ids": audio_ids,
        "parameters": {"samples": [
            {"audio_id": audio_ids[0], "transcript": "first transcript"},
            {"audio_id": audio_ids[1], "transcript": "second transcript"},
        ]},
    })
    assert response.status_code == 202
    record = await JobRepository().get(response.json()["job_id"])
    assert record.parameters["lens_id"].startswith("jlens-")
    assert sent[0][0][0] == "app.worker.tasks.execute_job"
    assert sent[0][1]["queue"] == "gpu-large"
