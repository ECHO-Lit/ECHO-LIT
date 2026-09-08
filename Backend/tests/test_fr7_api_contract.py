"""FR-7 API contract: POST /analyses/linguistic-vs-acoustic
(docs/FR7plan.md Part 1 S5.3, S8)."""

from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.api.routes import analyses as analyses_routes
from app.core.settings import settings
from app.core.storage import get_storage
from app.main import app


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def fake_broker(monkeypatch):
    monkeypatch.setattr(
        analyses_routes.celery_app, "send_task",
        lambda *args, **kwargs: SimpleNamespace(id="celery-task"),
    )


async def _upload(client, sample_audio_file) -> str:
    with sample_audio_file.open("rb") as handle:
        response = await client.post("/upload", files={"file": ("sample.wav", handle, "audio/wav")})
    assert response.status_code == 201
    return response.json()["audio_id"]


_BASE_PAYLOAD = {
    "model": "whisper-base",
    "task": "transcription",
    "sweeps": [{"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 3}],
}


@pytest.mark.asyncio
async def test_202_contract(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={**_BASE_PAYLOAD, "audio_ids": [audio_id]},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["status_url"] == f"/jobs/{body['job_id']}"
    assert body["result_url"] == f"/jobs/{body['job_id']}/result"
    assert body["estimated_variants"] > 0
    assert body["estimated_seconds"] > 0

    status = await client.get(body["status_url"])
    assert status.status_code == 200
    assert status.json()["status"] == "queued"
    assert status.json()["operation"] == "linguistic_acoustic"


@pytest.mark.asyncio
async def test_unversioned_alias_also_works(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/analyses/linguistic-vs-acoustic",
        json={**_BASE_PAYLOAD, "audio_ids": [audio_id]},
    )
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_grid_limit_rejected(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={
            "model": "whisper-base", "task": "transcription", "audio_ids": [audio_id],
            "sweeps": [
                {"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 15},
                {"property": "speed", "start": 0.5, "stop": 2.0, "steps": 15},
                {"property": "noise", "start": 40.0, "stop": 0.0, "steps": 15, "repeats": 10},
            ],
        },
    )
    assert response.status_code == 422
    detail = str(response.json()["detail"])
    assert "variants" in detail.lower()


@pytest.mark.asyncio
async def test_duplicate_property_rejected(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={
            "model": "whisper-base", "task": "transcription", "audio_ids": [audio_id],
            "sweeps": [
                {"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 3},
                {"property": "pitch", "start": -3.0, "stop": 3.0, "steps": 3},
            ],
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_cross_session_denied(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    async with AsyncClient(app=app, base_url="http://other") as other:
        response = await other.post(
            "/api/v1/analyses/linguistic-vs-acoustic",
            json={**_BASE_PAYLOAD, "audio_ids": [audio_id]},
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_classification_model_rejects_transcription_task(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={
            "model": "wav2vec2", "task": "transcription", "audio_ids": [audio_id],
            "sweeps": [{"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 3}],
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_auto_task_resolves_from_model_kind(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={
            "model": "wav2vec2", "task": "auto", "audio_ids": [audio_id],
            "sweeps": [{"property": "pitch", "start": -6.0, "stop": 6.0, "steps": 3}],
        },
    )
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_unknown_audio_id_rejected(client, sample_audio_file):
    await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={**_BASE_PAYLOAD, "audio_ids": ["not-a-real-audio-id"]},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_zero_width_sweep_rejected(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/api/v1/analyses/linguistic-vs-acoustic",
        json={
            "model": "whisper-base", "task": "transcription", "audio_ids": [audio_id],
            "sweeps": [{"property": "pitch", "start": 3.0, "stop": 3.0, "steps": 3}],
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_generic_jobs_endpoint_rejects_linguistic_acoustic(client, sample_audio_file):
    audio_id = await _upload(client, sample_audio_file)
    response = await client.post(
        "/jobs",
        json={"operation": "linguistic_acoustic", "model": "whisper-base", "audio_ids": [audio_id], "parameters": {}},
    )
    assert response.status_code == 422
