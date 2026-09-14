"""Load Testing -- batch throughput as workers are added.

Test Plan Section 3.1.5.  See tests/plans/3.1.5-load-testing.md.

SRS PE-2: "The asynchronous pipeline shall process multiple audio files
concurrently across worker queues; batch throughput shall scale approximately
linearly with the number of available workers up to hardware limits."

What this module can and cannot establish, stated once: it runs the real worker
code -- `execute_batch_item` for each file and `finalize_batch` for the chord
callback, exactly as `orchestrate_batch` fans them out -- on N OS threads, each
with its own event loop and its own Redis client over one shared store, as N
Celery worker processes would be.  The model call is replaced by a fixed,
GIL-releasing cost.  So the cases prove the *worker code* has no serialisation
point: no shared lock, no contended optimistic transaction that starts failing,
no per-job state that only one item may touch at a time.  The hardware half of
"up to hardware limits" is a property of the deployment's GPUs, not of this code.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from datetime import datetime, timezone

import fakeredis
import pytest
from fakeredis.aioredis import FakeRedis

from app.core import redis as redis_module
from app.core.settings import settings
from app.core.storage import get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobOperation, JobProgress, JobRecord, JobStatus, TaskAudio, TaskEnvelope
from app.worker import executor

pytestmark = [pytest.mark.performance, pytest.mark.slow]

ITEMS = 16
SERVICE_TIME = 0.1
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class _PerThreadRedis:
    """One FakeRedis client per (thread, event loop) over a shared server.

    redis.asyncio clients are bound to the loop that first used them, and a
    worker process has its own loop -- so each simulated worker needs its own
    client, while all of them must see the same data.
    """

    def __init__(self, server, db: int) -> None:
        self._server = server
        self._db = db
        self._local = threading.local()

    def _client(self):
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = None
        clients = self._local.__dict__.setdefault("clients", {})
        if loop_id not in clients:
            clients[loop_id] = FakeRedis(server=self._server, db=self._db, decode_responses=True)
        return clients[loop_id]

    def __getattr__(self, name):
        return getattr(self._client(), name)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "STORAGE_LOCAL_ROOT", str(tmp_path / "objects"))
    get_storage.cache_clear()
    yield
    get_storage.cache_clear()


@pytest.fixture
def shared_redis(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis_module, "redis", _PerThreadRedis(server, 0))
    monkeypatch.setattr(redis_module, "job_redis", _PerThreadRedis(server, 1))


@pytest.fixture
def model_calls(monkeypatch):
    """Replace the model with a fixed cost that releases the GIL, as a GPU call does."""
    calls: list[str] = []
    lock = threading.Lock()

    def fake_model(operation, model, audio_path, parameters, model_spec=None):
        time.sleep(SERVICE_TIME)
        with lock:
            calls.append(audio_path.name)
        return {"text": f"transcript of {audio_path.stem}"}

    monkeypatch.setattr(executor, "_execute_one", fake_model)
    return calls


def _batch(job_id: str, session_id: str = "s1") -> dict:
    storage = get_storage()
    audio = []
    for index in range(ITEMS):
        key = f"uploads/{session_id}/{job_id}-{index}.wav"
        storage.put_json(key, {"stand-in": index})  # the worker only downloads the bytes
        audio.append(TaskAudio(
            audio_id=f"{job_id}-a{index}", object_key=key, filename=f"clip-{index}.wav",
            media_type="audio/wav", sha256=hashlib.sha256(f"{job_id}-{index}".encode()).hexdigest(),
        ))
    asyncio.run(JobRepository().create(JobRecord(
        job_id=job_id, session_id=session_id, operation=JobOperation.prediction, model="whisper-base",
        audio_ids=[a.audio_id for a in audio], progress=JobProgress(current=0, total=ITEMS, message="Queued"),
        created_at=T0, updated_at=T0,
    )))
    envelope = TaskEnvelope(
        job_id=job_id, session_id=session_id, operation=JobOperation.prediction, model="whisper-base",
        audio=audio, result_schema_version=settings.RESULT_SCHEMA_VERSION, code_version=settings.CODE_VERSION,
    )
    return envelope.model_dump(mode="json")


def _run_workers(envelope: dict, workers: int) -> tuple[list[dict], float]:
    """Run every batch item across `workers` threads; return child results and wall time."""
    pending = list(range(ITEMS))
    pending_lock = threading.Lock()
    results: list[dict] = []
    failures: list[BaseException] = []

    def worker(number: int) -> None:
        loop = asyncio.new_event_loop()
        try:
            while True:
                with pending_lock:
                    if not pending:
                        return
                    index = pending.pop(0)
                try:
                    results.append(loop.run_until_complete(
                        executor.execute_batch_item(envelope, index, f"task-{number}-{index}")
                    ))
                except BaseException as exc:  # recorded, asserted on by the caller
                    failures.append(exc)
        finally:
            loop.close()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(workers)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    assert not failures, failures
    return results, elapsed


class TestWorkerScaling:
    def test_batch_throughput_scales_with_workers(self, shared_redis, model_calls):
        """LT-20: PE-2 -- 1, 2 and 4 workers on a 16-file batch.

        Scaling efficiency T1 / (N * TN) must stay at or above 0.7 -- "approximately
        linearly" with room for the fixed per-item Redis and storage overhead.
        """
        timings = {}
        for workers in (1, 2, 4):
            _, timings[workers] = _run_workers(_batch(f"scale-{workers}"), workers)
        print({n: f"{t:.2f}s eff={timings[1] / (n * t):.2f}" for n, t in timings.items()})

        for workers in (2, 4):
            efficiency = timings[1] / (workers * timings[workers])
            assert efficiency >= 0.7, f"{workers} workers: efficiency {efficiency:.2f}, timings {timings}"

    def test_concurrent_items_are_all_counted_and_none_fail(self, shared_redis, model_calls):
        """LT-21: 4 workers updating one job record and one completed-items set.

        Every item's progress update is an optimistic WATCH/MULTI transaction on
        the same job key, retried at most five times.  Contention that exhausted
        the retries would surface here as a RuntimeError from a batch item.
        """
        envelope = _batch("count")

        results, _ = _run_workers(envelope, 4)

        # Also guards BUG-48: before the fix, concurrent writes into the shared
        # cache-items/ directory failed items with "Object key escapes storage
        # root" on two of every four runs of this module on Windows.
        assert len(results) == ITEMS
        assert len(model_calls) == ITEMS
        record = asyncio.run(JobRepository().get("count"))
        assert record.progress.current == ITEMS
        assert record.status == JobStatus.processing
        assert record.error is None

    def test_finalize_orders_results_regardless_of_completion_order(self, shared_redis, model_calls):
        """LT-22: the chord callback reassembles the batch in submission order."""
        envelope = _batch("order")
        results, _ = _run_workers(envelope, 4)
        shuffled = sorted(results, key=lambda item: item["task_id"], reverse=True)

        asyncio.run(executor.finalize_batch(shuffled, envelope))

        record = asyncio.run(JobRepository().get("order"))
        assert record.status == JobStatus.success
        payload = get_storage().get_json(record.result_key)
        assert [item["audio_id"] for item in payload["items"]] == [f"order-a{i}" for i in range(ITEMS)]
        assert payload["summary"]["total_files"] == ITEMS

    def test_a_repeated_batch_is_served_from_the_item_cache_under_concurrency(self, shared_redis, model_calls):
        """LT-23: FR-4 "Serve an identical repeated request from cache", at 4 workers.

        The second run of the same files must not call the model at all -- and
        takes a fraction of the first run's time, which is the throughput a
        cache is for.
        """
        first = _batch("first")
        _, cold = _run_workers(first, 4)
        calls_after_first = len(model_calls)

        second = dict(first, job_id="second")
        asyncio.run(JobRepository().create(JobRecord(
            job_id="second", session_id="s1", operation=JobOperation.prediction, model="whisper-base",
            audio_ids=[a["audio_id"] for a in first["audio"]],
            progress=JobProgress(current=0, total=ITEMS, message="Queued"), created_at=T0, updated_at=T0,
        )))
        _, warm = _run_workers(second, 4)

        assert len(model_calls) == calls_after_first == ITEMS
        assert warm < cold / 2, f"cold {cold:.2f}s, warm {warm:.2f}s"


class TestStoragePathsUnderConcurrency:
    def test_an_extended_length_resolution_is_still_inside_the_root(self, tmp_path, monkeypatch):
        r"""LT-24: guards BUG-48 deterministically, without waiting for the race.

        Windows' `Path.resolve()` intermittently returns `\\?\C:\...` for a path
        in a directory other threads are writing.  The containment check must
        treat that as the same path, while still refusing a real escape.
        """
        from pathlib import Path

        from app.core.storage import LocalObjectStorage, StorageError

        storage = LocalObjectStorage(tmp_path / "root")
        expected = storage.root / "cache-items" / "abc.json"
        real_resolve = Path.resolve

        def extended_resolve(self, strict=False):
            return Path("\\\\?\\" + str(real_resolve(self, strict)))

        monkeypatch.setattr(Path, "resolve", extended_resolve)

        resolved = storage.path_for("cache-items/abc.json")

        assert resolved == expected
        monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path("C:\\elsewhere\\abc.json"))
        with pytest.raises(StorageError, match="escapes storage root"):
            storage.path_for("cache-items/abc.json")
