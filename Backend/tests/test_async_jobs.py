from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from datetime import datetime, timezone
import hashlib
import sys

import pytest

from app.api.routes import jobs as jobs_routes
from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobStatus
from app.schemas.jobs import AudioAsset, JobProgress, JobRecord, TaskAudio, TaskEnvelope


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.mark.asyncio
async def test_upload_creates_owned_audio_without_prediction(client, sample_audio_file):
    with sample_audio_file.open("rb") as handle:
        response = await client.post(
            "/upload",
            files={"file": ("sample.wav", handle, "audio/wav")},
            data={"model": "whisper-base"},
        )
    assert response.status_code == 201
    payload = response.json()
    assert payload["audio_id"] == payload["file_id"]
    assert "file_path" not in payload
    assert "prediction" not in payload
    assert payload["playback_url"] == f"/audio/{payload['audio_id']}"

    playback = await client.get(payload["playback_url"])
    assert playback.status_code == 200
    assert playback.content


@pytest.mark.asyncio
async def test_job_lifecycle_and_session_ownership(client, sample_audio_file, monkeypatch):
    with sample_audio_file.open("rb") as handle:
        upload = await client.post(
            "/upload", files={"file": ("sample.wav", handle, "audio/wav")}
        )
    audio_id = upload.json()["audio_id"]
    monkeypatch.setattr(
        jobs_routes.celery_app,
        "send_task",
        lambda *args, **kwargs: SimpleNamespace(id="celery-task"),
    )
    created = await client.post(
        "/jobs",
        json={
            "operation": "prediction",
            "model": "whisper-base",
            "audio_ids": [audio_id],
            "parameters": {},
        },
    )
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    status = await client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "queued"
    assert status.json()["progress"]["total"] == 1

    repository = JobRepository()
    record = await repository.get(job_id)
    result_key = f"results/{record.session_id}/{job_id}/result.json"
    get_storage().put_json(result_key, {"items": [{"audio_id": audio_id, "result": "ok"}]})
    await repository.update(job_id, status=JobStatus.success, result_key=result_key)
    result = await client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["items"][0]["result"] == "ok"

    from httpx import AsyncClient
    from app.main import app

    async with AsyncClient(app=app, base_url="http://other") as other:
        forbidden = await other.get(f"/jobs/{job_id}")
    assert forbidden.status_code == 404


@pytest.mark.asyncio
async def test_pending_result_and_cancellation(client, sample_audio_file, monkeypatch):
    with sample_audio_file.open("rb") as handle:
        upload = await client.post(
            "/upload", files={"file": ("sample.wav", handle, "audio/wav")}
        )
    monkeypatch.setattr(
        jobs_routes.celery_app,
        "send_task",
        lambda *args, **kwargs: SimpleNamespace(id="celery-task"),
    )
    monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", lambda *args, **kwargs: None)
    created = await client.post(
        "/jobs",
        json={
            "operation": "prediction",
            "model": "whisper-base",
            "audio_ids": [upload.json()["audio_id"]],
        },
    )
    job_id = created.json()["job_id"]
    assert (await client.get(f"/jobs/{job_id}/result")).status_code == 409
    cancelled = await client.delete(f"/jobs/{job_id}")
    assert cancelled.status_code == 202
    assert (await client.get(f"/jobs/{job_id}")).json()["status"] == "cancelled"


def test_job_contract_rejects_invalid_combinations():
    from pydantic import ValidationError
    from app.schemas.jobs import JobCreateRequest

    with pytest.raises(ValidationError):
        JobCreateRequest(operation="saliency", model="whisper-base", audio_ids=["a", "b"])
    with pytest.raises(ValidationError):
        JobCreateRequest(operation="attention", model="wav2vec2", audio_ids=["a"])
    with pytest.raises(ValidationError):
        JobCreateRequest(
            operation="attention",
            model="whisper-base",
            audio_ids=["a"],
            parameters={"layer_idx": -1},
        )
    with pytest.raises(ValidationError):
        JobCreateRequest(
            operation="prediction",
            model="whisper-base",
            audio_ids=["a"],
            parameters={"storage_key": "uploads/another-session/secret.wav"},
        )
    with pytest.raises(ValidationError):
        JobCreateRequest(
            operation="audio_features", model="whisper-base", audio_ids=["a"]
        )


def test_queue_routing_matches_worker_classes():
    from app.core.celery_app import finalization_queue_for, queue_for

    assert queue_for("prediction", "whisper-base", "mock") == "mock"
    assert queue_for("attention", "whisper-large", "mock") == "mock"
    assert finalization_queue_for("mock") == "mock"
    assert queue_for("prediction", "whisper-base", "mps") == "gpu-fast"
    assert queue_for("embedding", "wav2vec2", "mps") == "gpu-fast"
    assert queue_for("prediction", "whisper-large", "mps") == "gpu-large"
    assert queue_for("attention", "whisper-base", "cloud-gpu") == "gpu-large"
    assert queue_for("saliency", "wav2vec2", "cloud-gpu") == "gpu-large"
    assert queue_for("perturbation", None, "mps") == "cpu"
    assert queue_for("audio_features", None, "cloud-gpu") == "cpu"
    assert finalization_queue_for("mps") == "cpu"


def test_model_registry_reuses_and_evicts_variants(monkeypatch):
    from app.worker.model_registry import ModelRegistry

    calls: list[tuple[str, str]] = []
    fake_models = ModuleType("app.services.model_loader_service")
    fake_models.get_whisper_base_models = lambda: object()
    fake_models.get_whisper_large_models = lambda: object()
    fake_models.get_whisper_saliency_models = lambda revision: ("gradient", revision)
    fake_models.get_whisper_attention_models = lambda revision: ("attention", revision)
    fake_models.get_whisper_gen_model = lambda revision: ("generation", revision)
    fake_models.get_emotion_models = lambda: object()
    fake_models.unload_model_resources = lambda model, purpose: calls.append((model, purpose))
    fake_device = ModuleType("app.core.device")
    fake_device.INFERENCE_DEVICE = "cpu"
    monkeypatch.setitem(sys.modules, "app.services.model_loader_service", fake_models)
    monkeypatch.setitem(sys.modules, "app.core.device", fake_device)

    registry = ModelRegistry()
    registry.max_entries = 1
    first = registry.prepare("whisper-base", "embedding")
    assert registry.prepare("whisper-base", "embedding") is first
    registry.prepare("whisper-large", "attention")
    assert calls == [("whisper-base", "encoder")]


@pytest.mark.asyncio
async def test_worker_execution_progress_result_and_cache(sample_audio_file, monkeypatch):
    from app.repositories.audio import AudioRepository
    from app.worker import executor

    session_id = "worker-session"
    audio_id = "audio-id"
    object_key = f"uploads/{session_id}/{audio_id}.wav"
    get_storage().put_file(object_key, sample_audio_file, "audio/wav")
    asset = AudioAsset(
        audio_id=audio_id,
        session_id=session_id,
        object_key=object_key,
        filename="sample.wav",
        media_type="audio/wav",
        size_bytes=sample_audio_file.stat().st_size,
        duration_seconds=5,
        sample_rate=16000,
        channels=1,
        sha256=hashlib.sha256(sample_audio_file.read_bytes()).hexdigest(),
        created_at=datetime.now(timezone.utc),
    )
    await AudioRepository().create(asset)
    monkeypatch.setattr(executor, "_execute_one", lambda *args, **kwargs: {"text": "hello"})

    async def run(job_id: str):
        now = datetime.now(timezone.utc)
        record = JobRecord(
            job_id=job_id,
            session_id=session_id,
            operation="prediction",
            model="whisper-base",
            audio_ids=[audio_id],
            progress=JobProgress(total=1),
            created_at=now,
            updated_at=now,
        )
        await JobRepository().create(record)
        envelope = TaskEnvelope(
            job_id=job_id,
            session_id=session_id,
            operation="prediction",
            model="whisper-base",
            audio=[TaskAudio(
                audio_id=audio_id,
                object_key=object_key,
                filename="sample.wav",
                media_type="audio/wav",
                sha256=asset.sha256,
            )],
            execution_profile="mock",
            result_schema_version="v1",
            code_version="test",
        )
        await executor.execute(envelope.model_dump(mode="json"), f"task-{job_id}")
        return await JobRepository().get(job_id)

    first = await run("first-job")
    assert first.status == JobStatus.success
    assert first.progress.current == 1
    assert get_storage().get_json(first.result_key)["items"][0]["result"]["text"] == "hello"

    second = await run("second-job")
    assert second.status == JobStatus.success
    assert second.cache_hit is True


def test_mock_outputs_are_deterministic_and_frontend_compatible(sample_audio_file):
    from app.worker.mock_executor import execute_one

    first = execute_one("prediction", "wav2vec2", sample_audio_file, {})
    second = execute_one("prediction", "wav2vec2", sample_audio_file, {})
    assert first == second
    assert first["predicted_emotion"] in first["probabilities"]
    assert sum(first["probabilities"].values()) == pytest.approx(1, abs=1e-5)

    saliency = execute_one(
        "saliency", "whisper-base", sample_audio_file, {"method": "gradcam"}
    )
    assert saliency["segments"]
    assert saliency["series"]
    attention = execute_one(
        "attention", "whisper-base", sample_audio_file, {"layer_idx": 6, "head_idx": 0}
    )
    assert attention["attention_pairs"]
    assert attention["timestamp_attention"]
    embedding = execute_one("embedding", "whisper-base", sample_audio_file, {})
    assert len(embedding) == 32


def test_execution_profile_safety_validation():
    from pydantic import ValidationError
    from app.core.settings import Settings

    with pytest.raises(ValidationError):
        Settings(
            ENVIRONMENT="development",
            EXECUTION_PROFILE="cloud-gpu",
            ALLOW_PAID_EXECUTION=True,
            _env_file=None,
        )
    with pytest.raises(ValidationError):
        Settings(
            ENVIRONMENT="production",
            EXECUTION_PROFILE="cloud-gpu",
            ALLOW_PAID_EXECUTION=False,
            _env_file=None,
        )
    production = Settings(
        ENVIRONMENT="production",
        EXECUTION_PROFILE="cloud-gpu",
        ALLOW_PAID_EXECUTION=True,
        _env_file=None,
    )
    assert production.EXECUTION_PROFILE == "cloud-gpu"
