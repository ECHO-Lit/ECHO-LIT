"""Failover and Recovery -- corrupted data elements and invalid pointers.

Test Plan Section 3.1.7.  See tests/plans/3.1.7-failover-and-recovery.md.

ECHO's caches are pointers in Redis (`analysis-cache:*`, `analysis-item-cache:*`)
to JSON objects in the object store, plus FR-10's content-addressed files.  A
crash, a full disk or a partial copy can leave an object that no longer parses,
or that parses to the wrong shape.  A cache is an optimisation: a damaged entry
must cost a recomputation, never a job -- and must not keep costing jobs until
its pointer expires.

Also here: two writers of one object key, which the per-file cache produces
whenever two users analyse the same clip at once, and the one listing endpoint
that read every record in a session.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.api.routes import jobs as jobs_routes
from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import LocalObjectStorage, get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobOperation, JobStatus, TaskEnvelope
from app.services import fairness_service, linguistic_acoustic_service
from app.worker import executor, tasks
from tests._faults import ORIGIN, FaultHarness, envelope_for, seed_job, tolerant_client
from tests._fixtures import wav_bytes

pytestmark = [pytest.mark.failover, pytest.mark.critical]

TRUNCATED = '{"result": {"text": "the quick br'


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture(autouse=True)
def harness(monkeypatch):
    faults = FaultHarness()
    faults.install(monkeypatch)
    yield faults
    faults.reconnect()


@pytest.fixture
def model(monkeypatch):
    calls: list[str] = []

    def fake_model(operation, model, audio_path, parameters, model_spec=None):
        calls.append(audio_path.stem)
        return {"text": f"transcript of {audio_path.stem}"}

    monkeypatch.setattr(executor, "_execute_one", fake_model)
    return calls


def _corrupt(key: str, content: str = TRUNCATED) -> Path:
    path = get_storage().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


async def _poison_item_cache(envelope: dict, index: int = 0, content: str = TRUNCATED) -> Path:
    """Point the per-file cache for one clip at a damaged object."""
    parsed = TaskEnvelope.model_validate(envelope)
    digest = executor.item_cache_key(parsed, parsed.audio[index].sha256)
    key = f"cache-items/{digest}.json"
    await redis_module.redis.set(f"analysis-item-cache:{digest}", key)
    return _corrupt(key, content)


async def _poison_analysis_cache(envelope: dict, content: str = TRUNCATED) -> Path:
    digest = executor.analysis_cache_key(TaskEnvelope.model_validate(envelope))
    key = f"cache/{digest}/result.json"
    await redis_module.redis.set(f"analysis-cache:{digest}", key)
    return _corrupt(key, content)


class TestCorruptCaches:
    async def test_a_corrupt_file_cache_entry_is_recomputed_on_the_single_job_path(self, model):
        """FO-70: guards BUG-57.

        The single-file path read a cached entry with a bare `json.loads`: a
        truncated object failed the job as `execution_failed` -- non-retryable
        -- and every later job for that clip failed the same way until the
        pointer expired.
        """
        await seed_job("single")
        envelope = envelope_for("single")
        poisoned = await _poison_item_cache(envelope)

        await executor.execute(envelope, "t1")

        record = await JobRepository().get("single")
        assert record.status == JobStatus.success, record.error
        assert model == ["single-a0"]
        assert get_storage().get_json(f"cache-items/{poisoned.stem}.json")["result"]["text"] == "transcript of single-a0"

    async def test_a_corrupt_file_cache_entry_is_recomputed_on_the_batch_path(self, model):
        """FO-71: guards BUG-57 -- the same entry, read by a batch item."""
        envelope = envelope_for("batch", items=2)
        await seed_job("batch", audio_ids=[a["audio_id"] for a in envelope["audio"]], total=2)
        await _poison_item_cache(envelope, index=1)

        results = [await executor.execute_batch_item(envelope, i, f"t{i}") for i in range(2)]

        assert [r["result"]["text"] for r in results] == ["transcript of batch-a0", "transcript of batch-a1"]
        assert (await JobRepository().get("batch")).error is None

    async def test_a_corrupt_analysis_cache_entry_is_recomputed(self, model):
        """FO-72: guards BUG-57 -- the whole-analysis cache on the single-job path."""
        await seed_job("whole")
        envelope = envelope_for("whole")
        await _poison_analysis_cache(envelope)

        await executor.execute(envelope, "t1")

        record = await JobRepository().get("whole")
        assert record.status == JobStatus.success, record.error
        assert record.cache_hit is False
        assert model == ["whole-a0"]

    async def test_a_corrupt_batch_analysis_cache_lets_the_orchestrator_dispatch(self, monkeypatch):
        """FO-73: guards BUG-57 -- the orchestrator read it before dispatching.

        The damaged entry raised out of `orchestrate_batch` itself, which had no
        error handling: the job stayed `queued` forever.
        """
        envelope = envelope_for("fanout", items=3)
        await seed_job("fanout", audio_ids=[a["audio_id"] for a in envelope["audio"]], total=3)
        await _poison_analysis_cache(envelope)
        dispatched = []

        def chord(signatures):
            def publish(callback):
                dispatched.append(len(signatures))
                return SimpleNamespace(parent=SimpleNamespace(results=[]))
            return publish

        monkeypatch.setattr(tasks, "chord", chord)
        outcome = await asyncio.to_thread(tasks.orchestrate_batch.apply, args=[envelope])

        assert outcome.successful(), outcome.result
        assert dispatched == [3]

    async def test_corrupt_fr7_caches_are_recomputed(self, monkeypatch):
        """FO-74: guards BUG-57 -- the FR-7 sweep cache and per-variant cache."""
        await seed_job("sweep", status=JobStatus.processing, operation=JobOperation.linguistic_acoustic)
        envelope = envelope_for("sweep", operation="linguistic_acoustic",
                                parameters={"sweeps": [], "task": "transcription"})
        digest = linguistic_acoustic_service.sweep_cache_key(TaskEnvelope.model_validate(envelope))
        await redis_module.redis.set(f"analysis-cache:{digest}", f"cache/{digest}/result.json")
        _corrupt(f"cache/{digest}/result.json")

        assert await linguistic_acoustic_service.complete_sweep_from_cache(envelope) is False

        variant = get_storage().path_for("generated/s1/v1.wav")
        variant.parent.mkdir(parents=True, exist_ok=True)
        variant.write_bytes(wav_bytes())
        rendered = {"variant_id": "v1", "variant_audio_id": "v1", "object_key": "generated/s1/v1.wav",
                    "sha256": "variant-digest"}
        item = linguistic_acoustic_service._variant_item_cache_key(TaskEnvelope.model_validate(envelope), "variant-digest")
        await redis_module.redis.set(f"analysis-item-cache:{item}", f"cache-items/{item}.json")
        _corrupt(f"cache-items/{item}.json")
        predictions = []
        monkeypatch.setattr(linguistic_acoustic_service, "_run_prediction",
                            lambda model, spec, path: predictions.append(path) or {"text": "recomputed"})

        output = await linguistic_acoustic_service.infer_variant(envelope, rendered, "t1")

        assert output["output"] == {"text": "recomputed"}
        assert output["cache_hit"] is False
        assert len(predictions) == 1

    async def test_corrupt_fr10_caches_are_recomputed(self, monkeypatch, tmp_path):
        """FO-75: guards BUG-57 -- the FR-10 result cache and per-item prediction cache.

        The prediction cache was read outside the shard's per-item handling, so
        one damaged entry raised out of the shard and stalled the analysis.
        """
        from app.services import dataset_service

        params = {"dataset": "saa", "grouping_key": ["native_language"], "task": "transcription",
                  "include_representation": False}
        await seed_job("fair", status=JobStatus.processing, operation=JobOperation.fairness)
        envelope = envelope_for("fair", operation="fairness", parameters=params)
        digest = fairness_service.fairness_cache_key(TaskEnvelope.model_validate(envelope))
        _corrupt(f"fairness/cache/{digest}.json")

        assert await fairness_service.complete_from_cache(envelope) is False

        index = fairness_service.build_index("saa", ["native_language"], "s1", task="transcription",
                                             min_group_size=8, min_speakers_per_group=2)
        get_storage().put_json(fairness_service._plan_key("s1", "fair"), {"index": index.to_dict(), "plan": {}})
        group = index.groups[0]
        item_id = group.items[0].item_id
        pred_key = fairness_service._item_pred_cache_key(TaskEnvelope.model_validate(envelope), item_id)
        _corrupt(f"fairness/cache/pred/{pred_key}.json")
        clip = tmp_path / "clip.wav"
        clip.write_bytes(wav_bytes())
        monkeypatch.setattr(dataset_service, "resolve_file", lambda *args: clip)
        monkeypatch.setattr(fairness_service, "_run_op", lambda *args: {"text": "recomputed"})

        shard = await fairness_service.infer_shard(
            envelope, {"shard_id": "s1", "group_label": group.label, "item_ids": [item_id]}, "t1")

        assert shard["n_failed"] == 0
        assert shard["n_cached"] == 0
        assert get_storage().get_json(f"fairness/cache/pred/{pred_key}.json")["hypothesis"] == "recomputed"

    @pytest.mark.parametrize("content", ['["a", "list"]', '{"no_result_key": 1}', '"just a string"'])
    async def test_valid_json_of_the_wrong_shape_is_a_miss(self, model, content):
        """FO-76: guards BUG-57 -- corruption that still parses."""
        await seed_job("shape")
        envelope = envelope_for("shape")
        await _poison_item_cache(envelope, content=content)

        await executor.execute(envelope, "t1")

        record = await JobRepository().get("shape")
        assert record.status == JobStatus.success, record.error
        assert model == ["shape-a0"]

    async def test_the_next_identical_job_is_served_by_the_repaired_entry(self, model):
        """FO-77: guards BUG-57 -- recovery is complete: the entry is repaired, not bypassed."""
        first = envelope_for("first")
        await seed_job("first")
        await _poison_item_cache(first)
        await executor.execute(first, "t1")

        second = dict(first, job_id="second")
        await seed_job("second")
        await executor.execute(second, "t2")

        assert model == ["first-a0"]
        assert (await JobRepository().get("second")).status == JobStatus.success


class TestConcurrentWriters:
    def test_two_writers_of_one_key_both_succeed_and_leave_one_complete_version(self, tmp_path, monkeypatch):
        """FO-78: guards BUG-58.

        `put_file` staged every write of a key through the same `<key>.tmp`.
        Writer A copies, writer B overwrites the same temporary file and
        renames it into place, and A's rename then finds nothing to rename:
        `FileNotFoundError`, which the executor records as a non-retryable job
        failure.  Two users analysing the same clip share its cache entry, so
        this happens under ordinary concurrency.
        """
        import shutil

        from app.core import storage as storage_module

        storage = LocalObjectStorage(tmp_path / "root")
        sources = {}
        for name in ("a", "b"):
            sources[name] = tmp_path / f"{name}.json"
            sources[name].write_text(f'{{"writer": "{name}", "payload": "{name * 4096}"}}')
        a_copied, b_done = threading.Event(), threading.Event()
        real_copy = shutil.copyfile

        def gated_copy(src, dst, *args, **kwargs):
            result = real_copy(src, dst, *args, **kwargs)
            if Path(src) == sources["a"]:
                a_copied.set()
                b_done.wait(5)
            return result

        monkeypatch.setattr(storage_module.shutil, "copyfile", gated_copy)
        errors = []

        def write(name):
            try:
                storage.put_file("cache-items/shared.json", sources[name])
            except Exception as exc:  # recorded and asserted on below
                errors.append(exc)
            finally:
                if name == "b":
                    b_done.set()

        writer_a = threading.Thread(target=write, args=("a",))
        writer_a.start()
        assert a_copied.wait(5)
        write("b")
        writer_a.join(5)

        assert errors == []
        final = storage.get_json("cache-items/shared.json")
        assert final["writer"] in {"a", "b"}
        assert final["payload"] == final["writer"] * 4096
        assert list((storage.root / "cache-items").glob("*.tmp")) == []


class TestUnreadableRecords:
    async def test_cancel_all_cancels_the_rest_when_one_record_is_unreadable(self, harness, monkeypatch):
        """FO-79: guards BUG-59.

        `cancel-all` walks every job in the session.  One unreadable record
        raised a validation error and aborted the walk with a 500, so the
        user's other running analyses could not be cancelled at all.
        """
        monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", lambda *a, **k: None)
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = (await client.get("/session")).json()["sid"]
            await seed_job("fair-1", session_id=sid, status=JobStatus.processing, operation=JobOperation.fairness)
            await seed_job("fair-2", session_id=sid, status=JobStatus.queued, operation=JobOperation.fairness)
            harness.raw(1).set("job:fair-1", '{"job_id": "fair-1", "status": "proc')

            response = await client.post("/api/v1/analyses/fairness/cancel-all")

        assert response.status_code == 200, response.text[:200]
        assert response.json()["cancelled_job_ids"] == ["fair-2"]
        assert (await JobRepository().get("fair-2")).status == JobStatus.cancelled
        assert harness.raw(1).get("job:fair-1") == '{"job_id": "fair-1", "status": "proc'
