"""FR-10 cancellation and partial-shard-failure handling
(docs/FR10plan.md Part 1 S3.4, S2.6, S9 tests #15-16).

Regression coverage for a real bug found while writing these tests: infer_shard
and explain_shard called _check_cancel() but never caught JobCancelled to
actually set JobStatus.cancelled -- the flag was observed but never acted on,
so DELETE /jobs/{id} on a running FR-10 job would leave it stuck in whatever
state it was already in instead of transitioning to CANCELLED."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pytest

from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobOperation, JobProgress, JobRecord, JobStatus, TaskEnvelope
from app.services.fairness_service import (
    JobCancelled,
    _plan_key,
    aggregate_fairness,
    build_index,
    build_plan,
    infer_shard,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


async def _seed_job(session_id: str, params: dict, status: JobStatus = JobStatus.processing) -> str:
    job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await JobRepository().create(JobRecord(
        job_id=job_id, session_id=session_id, operation=JobOperation.fairness,
        model=params["model"], audio_ids=[], parameters=params,
        progress=JobProgress(current=1, total=10, message="Running"),
        created_at=now, updated_at=now,
    ))
    if status != JobStatus.queued:
        await JobRepository().update(job_id, status=status)
    return job_id


def _envelope(job_id: str, session_id: str, params: dict) -> dict:
    return TaskEnvelope(
        job_id=job_id, session_id=session_id, operation=JobOperation.fairness,
        model=params["model"], model_spec=None, audio=[], parameters=params,
        result_schema_version="v1", code_version="test",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_infer_shard_raises_and_marks_cancelled_when_cancel_requested():
    session_id = "sess-cancel-1"
    params = {"dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base",
              "task": "transcription", "shard_size": 8}
    job_id = await _seed_job(session_id, params)
    await JobRepository().request_cancel(job_id)

    envelope = _envelope(job_id, session_id, params)
    # Cancellation is checked BEFORE the plan is even loaded, so this must
    # raise without ever touching storage for the plan/model.
    with pytest.raises(JobCancelled):
        await infer_shard(envelope, {"shard_id": "s1", "group_label": "x", "item_ids": ["missing"]}, "task-id")

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.cancelled


@pytest.mark.asyncio
async def test_aggregate_is_noop_when_cancel_requested_between_stages():
    session_id = "sess-cancel-2"
    params = {"dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base",
              "task": "transcription", "shard_size": 8}
    job_id = await _seed_job(session_id, params)

    index = build_index("saa", ["native_language"], session_id, task="transcription",
                        min_group_size=8, min_speakers_per_group=2)
    storage = get_storage()
    storage.put_json(_plan_key(session_id, job_id), {"index": index.to_dict(), "plan": {}})

    await JobRepository().request_cancel(job_id)
    envelope = _envelope(job_id, session_id, params)

    await aggregate_fairness([], [], envelope)

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.cancelled
    assert record.result_key is None, "a cancelled job must not have a result written"


@pytest.mark.asyncio
async def test_aggregate_shrinks_group_on_partial_failure_and_excludes_below_threshold():
    session_id = "sess-fail-1"
    params = {
        "dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base",
        "task": "transcription", "min_group_size": 8, "min_speakers_per_group": 2,
        "include_representation": False, "include_explanations": False,
        "metrics": ["wer"], "n_bootstrap": 200, "seed": 0,
    }
    job_id = await _seed_job(session_id, params)
    index = build_index("saa", ["native_language"], session_id, task="transcription",
                        min_group_size=8, min_speakers_per_group=2)
    storage = get_storage()
    storage.put_json(_plan_key(session_id, job_id), {"index": index.to_dict(), "plan": {}})

    # Simulate what infer_shard would have written to storage: every group's
    # items get a valid transcription EXCEPT "arabic", which loses 27 of its
    # 30 items to (simulated) inference failures -- only 3 survive, below the
    # min_group_size=8 gate. Every other group keeps its full 30.
    infer_results = []
    for group in index.groups:
        records = []
        n_keep = 3 if group.label == "arabic" else group.n_items
        for item in group.items[:n_keep]:
            records.append({
                "item_id": item.item_id, "speaker_id": item.speaker_id, "content_id": item.content_id,
                "group_label": group.label, "hypothesis": item.reference_text, "reference": item.reference_text,
            })
        artifact_key = f"fairness/{session_id}/{job_id}/infer/{group.label}.json"
        storage.put_json(artifact_key, records)
        infer_results.append({
            "shard_id": group.label, "group_label": group.label, "operation": "prediction",
            "n_items": group.n_items, "n_cached": 0, "n_failed": group.n_items - n_keep,
            "artifact_key": artifact_key, "failures": [],
        })

    envelope = _envelope(job_id, session_id, params)
    await aggregate_fairness([], infer_results, envelope)

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.success, "a partially-failed shard must not fail the whole job"

    result = storage.get_json(record.result_key)
    excluded_labels = {e["label"]: e for e in result["excluded_groups"]}
    assert "arabic" in excluded_labels
    assert excluded_labels["arabic"]["reason"] == "insufficient_after_failures"
    assert excluded_labels["arabic"]["n_items"] == 3

    surviving_labels = {g["label"] for g in result["groups"]}
    assert "arabic" not in surviving_labels
    assert {"english", "mandarin", "russian", "spanish"} <= surviving_labels
    for group in result["groups"]:
        assert group["n_items"] == 30
        assert group["n_failed"] == 0
