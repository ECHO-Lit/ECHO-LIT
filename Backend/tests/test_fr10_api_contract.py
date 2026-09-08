"""FR-10 API contract: GET /analyses/fairness/groupable, POST /analyses/fairness
(docs/FR10plan.md Part 1 S5.3, S9)."""

from types import SimpleNamespace

import pytest

from app.api.routes import analyses as analyses_routes
from app.core.settings import settings
from app.core.storage import get_storage


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


@pytest.mark.asyncio
async def test_groupable_columns_saa(client):
    response = await client.get("/api/v1/analyses/fairness/groupable", params={"dataset": "saa"})
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == 150
    columns = {c["column"]: c for c in body["columns"]}
    assert "native_language" in columns
    assert columns["native_language"]["n_values"] == 5


@pytest.mark.asyncio
async def test_groupable_columns_unknown_dataset_404(client):
    response = await client.get("/api/v1/analyses/fairness/groupable", params={"dataset": "not-a-dataset"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_202_contract(client):
    response = await client.post(
        "/api/v1/analyses/fairness",
        json={"dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["status_url"] == f"/jobs/{body['job_id']}"
    assert body["result_url"] == f"/jobs/{body['job_id']}/result"
    assert body["dataset"] == "saa"
    assert body["notes"] == []


@pytest.mark.asyncio
async def test_classification_model_rejects_transcription_task(client):
    response = await client.post(
        "/api/v1/analyses/fairness",
        json={"dataset": "ravdess", "grouping_key": ["gender"], "model": "wav2vec2", "task": "transcription"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_unowned_custom_dataset_404(client):
    response = await client.post(
        "/api/v1/analyses/fairness",
        json={"dataset": "custom:someone-elses-session:foo", "grouping_key": ["accent"], "model": "whisper-base"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_generic_job_endpoint_rejects_fairness_operation(client):
    response = await client.post(
        "/jobs",
        json={"operation": "fairness", "audio_ids": ["x"], "model": "whisper-base"},
    )
    assert response.status_code == 422
