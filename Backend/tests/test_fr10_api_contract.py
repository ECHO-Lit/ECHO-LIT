"""FR-10 API contract: GET /analyses/fairness/groupable, POST /analyses/fairness
(docs/FR10plan.md Part 1 S5.3, S9)."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes import analyses as analyses_routes
from app.core.settings import settings
from app.core.storage import get_storage
from tests._corpora import requires_corpora


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
@requires_corpora("saa")
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


# ---------------------------------------------------------------------------
# Section 3.1.2 additions -- docs/FR10plan.md Part 1 S9 tests #19 and #20.
# ---------------------------------------------------------------------------


@pytest.mark.critical
@pytest.mark.parametrize(
    ("dropped", "expected_note", "disabled_flag"),
    [
        (
            "embedding",
            "exposes no embeddings; representational comparison will be skipped.",
            "include_representation",
        ),
        (
            "saliency",
            "does not support saliency; explanation fairness will be skipped.",
            "include_explanations",
        ),
    ],
    ids=["no_embeddings", "no_saliency"],
)
async def test_a_missing_optional_capability_degrades_rather_than_rejects(
    client, monkeypatch, dropped, expected_note, disabled_flag
):
    """FR10plan S9 #19: a partially capable model still gets a fairness run.

    Rejecting here would make fairness unavailable to exactly the models most
    worth auditing.  The analysis narrows instead, and says so in `notes`.

    Every catalogued model currently declares both optional capabilities, so the
    branch is unreachable with a real model (OBS-06); the definition is stripped
    here to exercise it.
    """
    from app.core.model_catalog import MODEL_DEFINITIONS

    original = MODEL_DEFINITIONS["whisper-base"]
    stripped = dataclasses.replace(
        original, capabilities=frozenset(original.capabilities - {dropped})
    )
    monkeypatch.setitem(MODEL_DEFINITIONS, "whisper-base", stripped)

    response = await client.post(
        "/api/v1/analyses/fairness",
        json={"dataset": "ravdess", "grouping_key": ["emotion"], "model": "whisper-base"},
    )

    assert response.status_code == 202
    body = response.json()
    assert any(expected_note in note for note in body["notes"])

    # The narrowing is recorded on the job, not just announced in the response.
    from app.repositories.jobs import JobRepository

    record = await JobRepository().get(body["job_id"])
    assert record.parameters[disabled_flag] is False


@pytest.mark.critical
def test_the_control_plane_import_graph_excludes_the_heavy_stack():
    """FR10plan S9 #20: importing the API must not pull in the ML stack.

    The API image is deliberately built without torch/librosa/sklearn -- DSP and
    inference only ever run on workers.  A stray module-level import in a route
    or schema would not fail any other test; it would simply make the API image
    several gigabytes larger and slower to start.

    Checked in a clean subprocess: this pytest process has already imported all
    four (test_fr7_metrics imports jiwer, test_probing_service imports sklearn),
    so an in-process assertion could only ever pass by accident.
    """
    forbidden = {"torch", "librosa", "jiwer", "sklearn"}
    code = (
        "import json, sys; import app.main; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & "
        f"{forbidden!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Surface the traceback rather than letting an ImportError look like a pass.
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == []
