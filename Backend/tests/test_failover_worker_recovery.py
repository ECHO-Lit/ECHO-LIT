"""Failover and Recovery -- worker loss, time limits, retries and redelivery.

Test Plan Section 3.1.7.  See tests/plans/3.1.7-failover-and-recovery.md.

A worker can stop in four ways, and each reaches ECHO through a different door:

  * its pool child dies (SIGSEGV, OOM kill)  -> Celery's `Request.on_failure`
    with `WorkerLostError`; `reject_on_worker_lost` puts the task back;
  * it overruns the hard time limit          -> billiard kills the child, and
    `on_timeout` / `on_failure(TimeLimitExceeded)` ack the message;
  * it overruns the soft time limit          -> `SoftTimeLimitExceeded` raised
    inside the task;
  * a dependency fails under it              -> the task's own retry chain.

These cases drive Celery's own `Request` object for the first two -- so the
decision to requeue or acknowledge is Celery's, under ECHO's real configuration
-- and run the real tasks eagerly for the rest.

SRS RE-1: "The failure of a single worker shall not affect the availability of
the API or the interface.  Queued work shall continue on remaining workers."
SRS FR-4: "A worker error or time limit marks the job failed with an error
message; a failed task may be retried; no partial result is exposed."
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from unittest.mock import Mock

import pytest
from billiard.exceptions import SoftTimeLimitExceeded, TimeLimitExceeded, WorkerLostError
from pydantic import ValidationError

from app.api.routes import jobs as jobs_routes
from app.core.celery_app import celery_app
from app.core.settings import Settings, settings
from app.core.storage import LocalObjectStorage, StorageError, get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobError, JobOperation, JobStatus
from app.services import fairness_service, linguistic_acoustic_service
from app.worker import executor, tasks
from tests._faults import (
    ORIGIN,
    FaultHarness,
    ResilientWorker,
    celery_request,
    envelope_for,
    exception_info,
    seed_job,
    tolerant_client,
)
from tests._fixtures import wav_bytes

pytestmark = [pytest.mark.failover, pytest.mark.critical]

# Every task that carries a job: all of them must be requeued when their worker dies.
JOB_TASKS = [
    "app.worker.tasks.execute_job",
    "app.worker.tasks.execute_job_item",
    "app.worker.tasks.orchestrate_batch",
    "app.worker.tasks.finalize_batch_job",
    "app.worker.tasks.fr7_orchestrate",
    "app.worker.tasks.fr7_render_variant",
    "app.worker.tasks.fr7_dispatch_inference",
    "app.worker.tasks.fr7_infer_variant",
    "app.worker.tasks.fr7_aggregate",
    "app.worker.tasks.fr10_orchestrate",
    "app.worker.tasks.fr10_infer_shard",
    "app.worker.tasks.fr10_dispatch_explain",
    "app.worker.tasks.fr10_explain_shard",
    "app.worker.tasks.fr10_aggregate",
]


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


@pytest.fixture(autouse=True)
def no_publish(monkeypatch):
    """Nothing here may reach a broker; a chord would try redis://localhost."""
    def refuse(*args, **kwargs):
        raise AssertionError("a task was published")

    monkeypatch.setattr(tasks, "chord", refuse)
    for callback in (tasks.fr7_aggregate, tasks.fr10_aggregate):
        monkeypatch.setattr(callback, "apply_async", refuse)


@pytest.fixture
def backend(monkeypatch):
    """Celery's result backend for every job task (DB3 in production)."""
    mock = Mock()
    for name in JOB_TASKS:
        monkeypatch.setattr(celery_app.tasks[name], "_backend", mock)
    return mock


@pytest.fixture
def model(monkeypatch):
    calls: list[str] = []

    def fake_model(operation, model, audio_path, parameters, model_spec=None):
        calls.append(audio_path.name)
        return {"text": f"transcript of {audio_path.stem}"}

    monkeypatch.setattr(executor, "_execute_one", fake_model)
    return calls


def _args_for(task_name: str, envelope: dict) -> list:
    shard = {"shard_id": "s1", "group_label": "g", "item_ids": ["i1"]}
    spec = {"variant_id": "v1", "filename": "clip-0.wav", "object_key": "k", "baseline_sha256": "x",
            "audio_id": "a", "property": "pitch", "theta": 1.0, "repeat": 0, "is_control": False}
    return {
        "app.worker.tasks.execute_job": [envelope],
        "app.worker.tasks.execute_job_item": [envelope, 0],
        "app.worker.tasks.orchestrate_batch": [envelope],
        "app.worker.tasks.finalize_batch_job": [[], envelope],
        "app.worker.tasks.fr7_orchestrate": [envelope],
        "app.worker.tasks.fr7_render_variant": [envelope, spec],
        "app.worker.tasks.fr7_dispatch_inference": [[], envelope],
        "app.worker.tasks.fr7_infer_variant": [envelope, {"variant_id": "v1"}],
        "app.worker.tasks.fr7_aggregate": [[], envelope, []],
        "app.worker.tasks.fr10_orchestrate": [envelope],
        "app.worker.tasks.fr10_infer_shard": [envelope, shard],
        "app.worker.tasks.fr10_dispatch_explain": [[], envelope],
        "app.worker.tasks.fr10_explain_shard": [envelope, shard],
        "app.worker.tasks.fr10_aggregate": [[], [], envelope],
    }[task_name]


async def _apply(task, args, task_id="attempt"):
    """Run a task eagerly on its own worker thread and event loop (`tasks._loop`)."""
    return await asyncio.to_thread(task.apply, args=args, task_id=task_id)


# --------------------------------------------------------------------------
# The worker process dies
# --------------------------------------------------------------------------


class TestWorkerLost:
    @pytest.mark.parametrize("task_name", JOB_TASKS)
    async def test_a_lost_worker_requeues_every_job_task(self, backend, task_name):
        """FO-20: RE-1 -- a dying worker gives its task back to the queue.

        `acks_late` with `reject_on_worker_lost` means the message is rejected
        with requeue rather than acknowledged, so another worker receives it.
        Celery makes that decision in `Request.on_failure`; it is driven here
        with ECHO's configuration for every task that carries a job.
        """
        envelope = envelope_for("lost")
        request, outcome = celery_request(task_name, _args_for(task_name, envelope))

        request.on_failure(exception_info(WorkerLostError("Worker exited prematurely: signal 9 (SIGKILL).")))

        assert outcome.rejected == [True]
        assert outcome.acked == 0
        backend.mark_as_failure.assert_not_called()

    async def test_a_lost_worker_leaves_the_job_waiting_for_its_redelivery(self, backend):
        """FO-21: RE-1 -- a requeued job is not failed.

        The job keeps its state and gains no error: the redelivered task is
        still coming, and failing the job would throw that work away.
        """
        await seed_job("waiting", status=JobStatus.processing)
        request, _ = celery_request("app.worker.tasks.execute_job", [envelope_for("waiting")])

        request.on_failure(exception_info(WorkerLostError("Worker exited prematurely: signal 11 (SIGSEGV).")))

        record = await JobRepository().get("waiting")
        assert record.status == JobStatus.processing
        assert record.error is None

    @pytest.mark.parametrize("task_name", ["app.worker.tasks.execute_job", "app.worker.tasks.execute_job_item"])
    async def test_the_hard_time_limit_fails_the_job_with_a_message(self, backend, harness, task_name):
        """FO-22: guards BUG-52 -- FR-4 "a ... time limit marks the job failed".

        At the hard limit billiard kills the pool child; Celery acknowledges the
        message and records the failure only in its own result backend.  The
        ECHO job record stayed `processing` until its TTL, polled forever.
        """
        await seed_job("overrun", status=JobStatus.processing)
        envelope = envelope_for("overrun")
        request, outcome = celery_request(task_name, _args_for(task_name, envelope))

        request.on_timeout(soft=False, timeout=settings.TASK_TIME_LIMIT_SECONDS)
        request.on_failure(exception_info(TimeLimitExceeded(settings.TASK_TIME_LIMIT_SECONDS)))

        assert outcome.acked == 1
        assert outcome.rejected == []
        record = await JobRepository().get("overrun")
        assert record.status == JobStatus.failure
        assert record.error.code == "time_limit_exceeded"
        assert "time limit" in record.error.message
        assert await JobRepository().cancellation_requested("overrun")

    async def test_the_hard_time_limit_handler_never_overwrites_a_finished_job(self, backend, harness):
        """FO-23: guards BUG-52 -- the handler only ever moves a running job.

        A job that finished before the limit fired keeps its outcome, and a
        Redis outage at that moment is logged rather than raised into Celery.
        """
        await seed_job("finished", status=JobStatus.success)
        await seed_job("running", status=JobStatus.processing)
        for job_id in ("finished", "running"):
            request, _ = celery_request("app.worker.tasks.execute_job", [envelope_for(job_id)])
            if job_id == "running":
                with harness.outage():
                    request.on_failure(exception_info(TimeLimitExceeded(3600)))
            else:
                request.on_failure(exception_info(TimeLimitExceeded(3600)))

        assert (await JobRepository().get("finished")).status == JobStatus.success
        assert (await JobRepository().get("finished")).error is None
        assert (await JobRepository().get("running")).status == JobStatus.processing


class TestSoftTimeLimit:
    async def test_a_soft_limit_in_the_model_call_is_reported_as_a_time_limit(self, monkeypatch):
        """FO-24: guards BUG-53.

        `SoftTimeLimitExceeded()` carries no message, so the job failed as
        `execution_failed` with an empty message: the user saw a bare
        "Analysis failed".
        """
        await seed_job("slow")

        def overrun(*args, **kwargs):
            raise SoftTimeLimitExceeded()

        monkeypatch.setattr(executor, "_execute_one", overrun)
        with pytest.raises(SoftTimeLimitExceeded):
            await executor.execute(envelope_for("slow"), "t1")

        record = await JobRepository().get("slow")
        assert record.status == JobStatus.failure
        assert record.error.code == "time_limit_exceeded"
        assert "time limit" in record.error.message
        assert record.result_key is None

    async def test_a_soft_limit_escaping_the_executor_still_fails_the_job(self, monkeypatch):
        """FO-25: guards BUG-53 -- the limit can fire outside the executor's handlers.

        The signal lands wherever the child happens to be -- between awaits, in
        the task wrapper.  Before the fix it left the task with no handler at
        all, and the job `queued` until its TTL.
        """
        await seed_job("escaped")

        async def overrun(envelope, task_id):
            raise SoftTimeLimitExceeded()

        monkeypatch.setattr(tasks, "execute", overrun)
        outcome = await _apply(tasks.execute_job, [envelope_for("escaped")])

        assert isinstance(outcome.result, SoftTimeLimitExceeded)
        record = await JobRepository().get("escaped")
        assert record.status == JobStatus.failure
        assert record.error.code == "time_limit_exceeded"

    @pytest.mark.parametrize("stage", ["infer", "explain"])
    async def test_fr10_items_do_not_swallow_the_soft_limit(self, monkeypatch, tmp_path, stage):
        """FO-26: guards BUG-53.

        FR-10 shards record a per-item failure and carry on -- the right
        behaviour for one bad clip, the wrong one for the time limit: the shard
        kept going until the hard limit killed it.
        """
        from app.services import dataset_service

        await seed_job("fair", operation=JobOperation.fairness, status=JobStatus.processing)
        index = fairness_service.build_index("saa", ["native_language"], "s1", task="transcription",
                                             min_group_size=8, min_speakers_per_group=2)
        get_storage().put_json(fairness_service._plan_key("s1", "fair"), {"index": index.to_dict(), "plan": {}})
        group = index.groups[0]
        shard = {"shard_id": "s1", "group_label": group.label, "item_ids": [i.item_id for i in group.items[:3]]}
        clip = tmp_path / "clip.wav"
        clip.write_bytes(wav_bytes())
        monkeypatch.setattr(dataset_service, "resolve_file", lambda *args: clip)

        def overrun(*args, **kwargs):
            raise SoftTimeLimitExceeded()

        envelope = envelope_for("fair", operation="fairness", parameters={
            "dataset": "saa", "grouping_key": ["native_language"], "task": "transcription",
            "include_representation": False,
        })
        if stage == "infer":
            monkeypatch.setattr(fairness_service, "_run_op", overrun)
            with pytest.raises(SoftTimeLimitExceeded):
                await fairness_service.infer_shard(envelope, shard, "t1")
        else:
            monkeypatch.setitem(sys.modules, "app.services.saliency_service",
                                types.SimpleNamespace(generate_saliency=overrun))
            with pytest.raises(SoftTimeLimitExceeded):
                await fairness_service.explain_shard(envelope, shard, "t1")


# --------------------------------------------------------------------------
# A dependency fails under the worker: the retry chain
# --------------------------------------------------------------------------


class TestRetryChain:
    async def test_a_dependency_failure_during_the_first_attempt_is_retried_to_success(self, monkeypatch, model):
        """FO-27: RE-2 "Failed tasks shall be retryable" / UC-1 "a worker failure
        triggers a retry" -- the automatic half of RE-2.

        The object store fails the first download; Celery's retry runs the job
        again and it succeeds, with the transient error cleared from the record.
        """
        await seed_job("flaky")
        real_download = LocalObjectStorage.download_file
        downloads = []

        def flaky_download(self, key, destination):
            downloads.append(key)
            if len(downloads) == 1:
                raise StorageError("object store unavailable")
            return real_download(self, key, destination)

        monkeypatch.setattr(LocalObjectStorage, "download_file", flaky_download)
        outcome = await _apply(tasks.execute_job, [envelope_for("flaky")])

        assert outcome.successful(), outcome.result
        assert len(downloads) == 2
        record = await JobRepository().get("flaky")
        assert record.status == JobStatus.success
        assert record.error is None
        assert len(model) == 1

    @pytest.mark.parametrize(("task_name", "args"), [
        ("execute_job", lambda envelope: [envelope]),
        ("execute_job_item", lambda envelope: [envelope, 0]),
    ])
    async def test_a_persistent_failure_ends_as_dependency_unavailable(self, monkeypatch, task_name, args):
        """FO-28: FR-4 -- a dependency that never returns fails the job with a reason.

        Four attempts (the first and `max_retries=3`), then a terminal
        `dependency_unavailable` error rather than an endless retry.
        """
        await seed_job("down")
        attempts = []

        def always_down(self, key, destination):
            attempts.append(key)
            raise StorageError("object store unavailable")

        monkeypatch.setattr(LocalObjectStorage, "download_file", always_down)
        outcome = await _apply(getattr(tasks, task_name), args(envelope_for("down")))

        assert isinstance(outcome.result, StorageError)
        assert len(attempts) == 4
        record = await JobRepository().get("down")
        assert record.status == JobStatus.failure
        assert record.error.code == "dependency_unavailable"
        assert record.result_key is None

    async def test_resubmitting_a_failed_batch_reuses_its_completed_items(self, monkeypatch):
        """FO-29: RE-2 -- the manual half of "retryable": submit it again.

        A batch whose third item failed is resubmitted; the per-file cache
        serves the items that had finished, so only the failed one runs again.
        """
        calls = []

        def model_failing_once(operation, model, audio_path, parameters, model_spec=None):
            calls.append(audio_path.stem)
            if audio_path.stem.endswith("a2") and calls.count(audio_path.stem) == 1:
                raise ValueError("decoder error")
            return {"text": f"transcript of {audio_path.stem}"}

        monkeypatch.setattr(executor, "_execute_one", model_failing_once)
        first = envelope_for("first", items=4)
        await seed_job("first", audio_ids=[a["audio_id"] for a in first["audio"]], total=4)
        for index in range(4):
            try:
                await executor.execute_batch_item(first, index, f"t{index}")
            except ValueError:
                pass
        assert (await JobRepository().get("first")).status == JobStatus.failure

        second = dict(first, job_id="second")
        await seed_job("second", audio_ids=[a["audio_id"] for a in first["audio"]], total=4)
        results = [await executor.execute_batch_item(second, index, f"r{index}") for index in range(4)]
        await executor.finalize_batch(results, second)

        assert calls.count("first-a2") == 2
        assert len(calls) == 5  # four on the first run, the failed item again on the second
        record = await JobRepository().get("second")
        assert record.status == JobStatus.success
        assert record.error is None


# --------------------------------------------------------------------------
# RE-1: the rest of the system carries on
# --------------------------------------------------------------------------


class _WorkerDied(BaseException):
    """What the task sees of a SIGKILL: nothing -- no `except Exception` runs."""


class TestRemainingWorkers:
    async def test_a_batch_item_lost_with_its_worker_completes_on_another(self, monkeypatch):
        """FO-30: RE-1 "Queued work shall continue on remaining workers".

        The worker holding item 1 dies mid-model-call.  No handler runs, so the
        job keeps its state and gains no error; the redelivered item completes
        on another worker, and the batch finishes in order with exact progress.
        """
        died = []

        def model(operation, model, audio_path, parameters, model_spec=None):
            if audio_path.stem.endswith("a1") and not died:
                died.append(audio_path.stem)
                raise _WorkerDied()
            return {"text": f"transcript of {audio_path.stem}"}

        monkeypatch.setattr(executor, "_execute_one", model)
        envelope = envelope_for("batch", items=3)
        await seed_job("batch", audio_ids=[a["audio_id"] for a in envelope["audio"]], total=3)

        results = []
        for index in range(3):
            try:
                results.append(await executor.execute_batch_item(envelope, index, f"w1-{index}"))
            except _WorkerDied:
                pass
        mid = await JobRepository().get("batch")
        results.append(await executor.execute_batch_item(envelope, 1, "w2-1"))  # the redelivery
        await executor.finalize_batch(results, envelope)

        assert mid.status == JobStatus.processing and mid.error is None
        assert mid.progress.current == 2
        record = await JobRepository().get("batch")
        assert record.status == JobStatus.success
        assert record.progress.current == record.progress.total == 3
        payload = get_storage().get_json(record.result_key)
        assert [item["audio_id"] for item in payload["items"]] == ["batch-a0", "batch-a1", "batch-a2"]

    async def test_the_api_stays_available_without_workers_and_the_queue_drains_on_a_replacement(
        self, monkeypatch
    ):
        """FO-31: RE-1 -- with every worker gone, the API and the interface carry on.

        Submissions are accepted and polled within PE-1's 500 ms budget while
        nothing consumes the queue; a replacement worker then drains it.
        """
        worker = ResilientWorker(service_time=0.01, concurrency=2)
        monkeypatch.setattr(jobs_routes.celery_app, "send_task", worker.send_task)
        latencies: list[float] = []
        job_ids: list[str] = []

        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            upload = await client.post("/upload", files={"file": ("clip.wav", wav_bytes(), "audio/wav")})
            audio_id = upload.json()["audio_id"]
            for _ in range(10):
                started = time.perf_counter()
                submitted = await client.post("/jobs", json={
                    "operation": "prediction", "model": "whisper-base", "audio_ids": [audio_id]})
                latencies.append(time.perf_counter() - started)
                assert submitted.status_code == 202
                job_ids.append(submitted.json()["job_id"])
            for job_id in job_ids:
                started = time.perf_counter()
                polled = await client.get(f"/jobs/{job_id}")
                latencies.append(time.perf_counter() - started)
                assert polled.json()["status"] == "queued"

            worker.start()
            try:
                deadline = time.perf_counter() + 5
                while time.perf_counter() < deadline:
                    statuses = {(await client.get(f"/jobs/{j}")).json()["status"] for j in job_ids}
                    if statuses == {"success"}:
                        break
                    await asyncio.sleep(0.02)
            finally:
                await worker.stop()

        assert sum(latencies) / len(latencies) < 0.5
        assert statuses == {"success"}


# --------------------------------------------------------------------------
# Redelivery and timing
# --------------------------------------------------------------------------


def _work_targets():
    """The function each job task delegates to, as (module, attribute)."""
    return {
        "app.worker.tasks.execute_job": (tasks, "execute"),
        "app.worker.tasks.execute_job_item": (tasks, "execute_batch_item"),
        "app.worker.tasks.orchestrate_batch": (tasks, "complete_batch_from_cache"),
        "app.worker.tasks.finalize_batch_job": (tasks, "finalize_batch"),
        "app.worker.tasks.fr7_orchestrate": (linguistic_acoustic_service, "complete_sweep_from_cache"),
        "app.worker.tasks.fr7_render_variant": (linguistic_acoustic_service, "render_and_register"),
        "app.worker.tasks.fr7_dispatch_inference": (linguistic_acoustic_service, "mark_render_complete"),
        "app.worker.tasks.fr7_infer_variant": (linguistic_acoustic_service, "infer_variant"),
        "app.worker.tasks.fr7_aggregate": (linguistic_acoustic_service, "aggregate_sweep"),
        "app.worker.tasks.fr10_orchestrate": (fairness_service, "complete_from_cache"),
        "app.worker.tasks.fr10_infer_shard": (fairness_service, "infer_shard"),
        "app.worker.tasks.fr10_dispatch_explain": (fairness_service, "mark_infer_complete"),
        "app.worker.tasks.fr10_explain_shard": (fairness_service, "explain_shard"),
        "app.worker.tasks.fr10_aggregate": (fairness_service, "aggregate_fairness"),
    }


class TestRedelivery:
    @pytest.mark.parametrize("state", ["success", "failure", "cancelled", "missing"])
    @pytest.mark.parametrize("task_name", JOB_TASKS)
    async def test_a_task_for_a_finished_job_does_no_work(self, monkeypatch, task_name, state):
        """FO-32: guards BUG-56 -- a redelivered or late task finds its job done.

        A job can finish while one of its tasks is still in the broker: the
        reaper failed it, the user cancelled it, or the visibility timeout
        handed a still-running task to a second worker.  That task must not
        run the model again for a job nobody is waiting on.

        The one exception is `fr7_render_variant` with no record: the on-demand
        playback render (`POST /audio/{id}/variant`) runs it without a job.
        """
        ran = []

        async def must_not_run(*args, **kwargs):
            ran.append(task_name)
            return {"applicable": False} if task_name.endswith("render_variant") else None

        module, attribute = _work_targets()[task_name]
        monkeypatch.setattr(module, attribute, must_not_run)
        if state != "missing":
            await seed_job("done", status=JobStatus(state),
                           error=JobError(code="earlier", message="earlier") if state == "failure" else None)
        before = await JobRepository().get("done")

        await _apply(celery_app.tasks[task_name], _args_for(task_name, envelope_for("done")))

        on_demand = task_name.endswith("fr7_render_variant") and state == "missing"
        assert ran == ([task_name] if on_demand else [])
        assert await JobRepository().get("done") == before

    def test_the_timings_let_a_lost_task_be_redelivered_before_its_job_is_reaped(self):
        """FO-33: guards BUG-56.

        kombu's Redis transport redelivers an unacknowledged task after its
        visibility timeout, 3600 s by default -- exactly the hard time limit, so
        a task still legitimately running at the limit was handed to a second
        worker.  The order must be: soft < hard < visibility < stale-job cutoff,
        so a lost task is redelivered before the reaper gives up on its job.
        """
        assert (settings.TASK_SOFT_TIME_LIMIT_SECONDS < settings.TASK_TIME_LIMIT_SECONDS
                < settings.BROKER_VISIBILITY_TIMEOUT_SECONDS < settings.STALE_JOB_SECONDS)
        options = celery_app.conf.broker_transport_options
        assert options["visibility_timeout"] == settings.BROKER_VISIBILITY_TIMEOUT_SECONDS

    @pytest.mark.parametrize("overrides", [
        {"TASK_SOFT_TIME_LIMIT_SECONDS": 3600, "TASK_TIME_LIMIT_SECONDS": 3600},
        {"TASK_TIME_LIMIT_SECONDS": 3600, "BROKER_VISIBILITY_TIMEOUT_SECONDS": 3600},
        {"BROKER_VISIBILITY_TIMEOUT_SECONDS": 5000, "STALE_JOB_SECONDS": 4000},
    ])
    def test_mis_ordered_timings_are_refused_at_startup(self, overrides):
        """FO-34: guards BUG-56 -- a deployment cannot configure the race back in."""
        with pytest.raises(ValidationError, match="must be ordered"):
            Settings(**overrides)
