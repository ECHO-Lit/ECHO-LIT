"""FR-10 worker orchestration: partitioning, progress counters, caching
(docs/FR10plan.md Part 1 S3.4, S9)."""

from datetime import datetime, timezone
import uuid

import pytest

from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobOperation, JobProgress, JobRecord, JobStatus
from app.services.fairness_service import (
    FairnessInputError,
    _advance_fr10_stage,
    complete_from_cache,
    fairness_cache_key,
    load_plan,
    prepare_analysis,
)
from app.schemas.jobs import TaskEnvelope


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


async def _seed_job(session_id: str, params: dict) -> str:
    job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    await JobRepository().create(JobRecord(
        job_id=job_id, session_id=session_id, operation=JobOperation.fairness,
        model=params["model"], audio_ids=[], parameters=params,
        progress=JobProgress(current=0, total=1, message="Queued"),
        created_at=now, updated_at=now,
    ))
    return job_id


def _envelope(job_id: str, session_id: str, params: dict) -> dict:
    return TaskEnvelope(
        job_id=job_id, session_id=session_id, operation=JobOperation.fairness,
        model=params["model"], model_spec=None, audio=[], parameters=params,
        result_schema_version="v1", code_version="test",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_prepare_analysis_partitions_saa_and_advances_progress():
    session_id = "sess-a"
    params = {
        "dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base",
        "task": "transcription", "min_group_size": 8, "min_speakers_per_group": 2,
        "include_explanations": True, "explain_per_group": 12, "explain_shard_size": 4,
        "shard_size": 8, "seed": 0,
    }
    job_id = await _seed_job(session_id, params)
    envelope = _envelope(job_id, session_id, params)

    infer_shards = await prepare_analysis(envelope, "celery-task-id")
    assert sum(len(s["item_ids"]) for s in infer_shards) == 150

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.processing
    assert record.progress.total == len(infer_shards) + 15 + 2  # infer + explain(15) + partition/aggregate
    assert "matched" in record.progress.message

    plan = await load_plan(envelope)
    assert plan["index"]["design"] == "matched"
    assert len(plan["plan"]["explain_shards"]) == 15


@pytest.mark.asyncio
async def test_prepare_analysis_insufficient_groups_fails_job_not_raises_uncaught():
    session_id = "sess-b"
    # Common Voice grouped by accent with a very high min_group_size -> every
    # group excluded -> FairnessInputError, job must end FAILURE not crash silently.
    params = {
        "dataset": "common-voice", "grouping_key": ["accent"], "model": "whisper-base",
        "task": "transcription", "min_group_size": 1000, "min_speakers_per_group": 2,
        "include_explanations": False, "shard_size": 8, "seed": 0,
    }
    job_id = await _seed_job(session_id, params)
    envelope = _envelope(job_id, session_id, params)

    with pytest.raises(FairnessInputError):
        await prepare_analysis(envelope, "celery-task-id")

    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.failure
    assert record.error.code == "fr10_invalid_input"


@pytest.mark.asyncio
async def test_advance_stage_is_idempotent_under_redelivery():
    session_id = "sess-c"
    params = {"dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base",
              "task": "transcription", "shard_size": 8, "seed": 0}
    job_id = await _seed_job(session_id, params)
    # Simulate a partitioned job with a known total.
    await JobRepository().update(job_id, status=JobStatus.processing,
                                 progress=JobProgress(current=1, total=5, message="Partitioned"))

    await _advance_fr10_stage(job_id, "fr10:infer", "shard-1", "Inference")
    await _advance_fr10_stage(job_id, "fr10:infer", "shard-1", "Inference")  # redelivered, same shard id
    record = await JobRepository().get(job_id)
    # SADD is idempotent: replaying the SAME shard_id must not advance progress twice.
    assert record.progress.current == 2  # 1 (partition) + 1 distinct shard

    await _advance_fr10_stage(job_id, "fr10:infer", "shard-2", "Inference")
    record = await JobRepository().get(job_id)
    assert record.progress.current == 3


@pytest.mark.asyncio
async def test_cache_round_trip():
    session_id = "sess-d"
    params = {"dataset": "saa", "grouping_key": ["native_language"], "model": "whisper-base",
              "task": "transcription", "metrics": ["wer"], "shard_size": 8, "seed": 0}
    job_id = await _seed_job(session_id, params)
    envelope = _envelope(job_id, session_id, params)

    assert await complete_from_cache(envelope) is False  # nothing cached yet

    storage = get_storage()
    digest = fairness_cache_key(TaskEnvelope.model_validate(envelope))
    storage.put_json(f"fairness/cache/{digest}.json", {
        "job_id": "old-job", "provenance": {"total_units": 42}, "groups": [],
    })

    assert await complete_from_cache(envelope) is True
    record = await JobRepository().get(job_id)
    assert record.status == JobStatus.success
    assert record.cache_hit is True
    assert record.progress.current == 42 == record.progress.total

    result = storage.get_json(f"results/{session_id}/{job_id}/result.json")
    assert result["job_id"] == job_id  # rewritten to the NEW job id, not the cached one
    assert result["provenance"]["cache_hit"] is True
