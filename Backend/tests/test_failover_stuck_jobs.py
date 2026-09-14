"""Failover and Recovery -- incomplete cycles: jobs a failure left unfinished.

Test Plan Section 3.1.7.  See tests/plans/3.1.7-failover-and-recovery.md.

The RUP template's "incomplete cycles" are, in ECHO, jobs whose worker stopped
before writing a terminal state.  A job record only reaches `success`,
`failure` or `cancelled` because some task wrote it there; if every task that
could have written it is gone, the record sits at `processing` and the client
polls it until the 24-hour TTL.  This module covers the three ways out:

  * the stale-job reaper, a beat task that fails jobs nothing has touched for
    longer than any task can legitimately run;
  * stale cancel, so the owner of a stuck job can end it at once;
  * convergence in the multi-stage pipelines (batch, FR-7, FR-10), where a
    failed stage used to stall the chord with nothing recording why.

SRS FR-4: "A worker error or time limit marks the job failed with an error
message; ... no partial result is exposed."
SRS FR-14: "A running job can be canceled."
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from kombu.exceptions import OperationalError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from app.api.routes import jobs as jobs_routes
from app.core.celery_app import celery_app
from app.core.settings import settings
from app.core.storage import StorageError, get_storage
from app.repositories.jobs import JobRepository
from app.schemas.jobs import JobError, JobOperation, JobProgress, JobStatus
from app.services import fairness_service, linguistic_acoustic_service
from app.worker import executor, tasks
from tests._faults import ORIGIN, FaultHarness, envelope_for, seed_job, tolerant_client
from tests._fixtures import wav_bytes

pytestmark = [pytest.mark.failover, pytest.mark.critical]


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
def broker(monkeypatch):
    calls: list[dict] = []
    revoked: list[str] = []

    def _send_task(name, args=None, kwargs=None, queue=None, **rest):
        calls.append({"name": name, "args": args})
        return SimpleNamespace(id=f"celery-{len(calls)}")

    monkeypatch.setattr(jobs_routes.celery_app, "send_task", _send_task)
    monkeypatch.setattr(jobs_routes.celery_app.control, "revoke", lambda task_id, **kw: revoked.append(task_id))

    def refuse(*args, **kwargs):
        raise AssertionError("a task was published")

    monkeypatch.setattr(tasks, "chord", refuse)
    for callback in (tasks.fr7_aggregate, tasks.fr10_aggregate):
        monkeypatch.setattr(callback, "apply_async", refuse)
    return SimpleNamespace(calls=calls, revoked=revoked)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stale() -> datetime:
    return _now() - timedelta(seconds=settings.STALE_JOB_SECONDS + 60)


async def _apply(task, args, task_id="attempt"):
    return await asyncio.to_thread(task.apply, args=args, task_id=task_id)


async def _reap(now=None) -> int:
    from app.worker.recovery import reap_stale_jobs

    return await reap_stale_jobs(now=now)


# --------------------------------------------------------------------------
# The reaper
# --------------------------------------------------------------------------


REAPER_MATRIX = [
    # (id, status, age, error, reaped)
    ("FO-40 stale started", JobStatus.started, "stale", None, True),
    ("FO-41 stale processing", JobStatus.processing, "stale", None, True),
    ("FO-42 fresh processing", JobStatus.processing, "fresh", None, False),
    ("FO-43 never-started queued", JobStatus.queued, "stale", None, False),
    ("FO-44 queued after a transient failure", JobStatus.queued, "stale",
     JobError(code="transient_failure", message="A transient dependency failed", retryable=True), True),
    ("FO-45a finished success", JobStatus.success, "stale", None, False),
    ("FO-45b finished failure", JobStatus.failure, "stale", JobError(code="model_error", message="boom"), False),
    ("FO-45c finished cancelled", JobStatus.cancelled, "stale", None, False),
]


class TestReaper:
    @pytest.mark.parametrize(("case", "status", "age", "error", "reaped"), REAPER_MATRIX,
                             ids=[row[0] for row in REAPER_MATRIX])
    async def test_the_reaper_fails_exactly_the_jobs_nothing_is_running(self, case, status, age, error, reaped):
        """FO-40..FO-45: guards BUG-54.

        A job is stale when it is unfinished and nothing has written to it for
        longer than any task may run *plus* the broker's redelivery window.  A
        `queued` job that never started is not stale however old it is: its
        message is durable in the broker and a worker will take it.  A queued
        job carrying a transient error *has* started -- its task was put back by
        a failed retry chain -- and is.
        """
        stamp = _stale() if age == "stale" else _now() - timedelta(seconds=60)
        before = await seed_job("j", status=status, updated_at=stamp, error=error)

        count = await _reap()

        record = await JobRepository().get("j")
        if reaped:
            assert count == 1
            assert record.status == JobStatus.failure
            assert record.error.code == "worker_lost"
            assert await JobRepository().cancellation_requested("j")
        else:
            assert count == 0
            assert record == before
            assert not await JobRepository().cancellation_requested("j")

    async def test_a_job_updated_after_the_scan_is_left_alone(self, monkeypatch):
        """FO-46: guards BUG-54 -- the reaper loses every race it should lose.

        A worker writes progress between the reaper reading a stale record and
        failing it.  The transition re-checks staleness inside its WATCH
        transaction, so the live job is not failed.
        """
        from app.repositories import jobs as jobs_repository

        await seed_job("racing", status=JobStatus.processing, updated_at=_stale())
        real_finish = jobs_repository.JobRepository.finish

        async def progress_then_finish(self, job_id, *args, **kwargs):
            await JobRepository().update(job_id, progress=JobProgress(current=1, total=1, message="still going"))
            return await real_finish(self, job_id, *args, **kwargs)

        monkeypatch.setattr(jobs_repository.JobRepository, "finish", progress_then_finish)

        assert await _reap() == 0
        assert (await JobRepository().get("racing")).status == JobStatus.processing

    async def test_sub_keys_are_ignored_and_unreadable_records_skipped(self, harness):
        """FO-47: guards BUG-54 -- invalid keys and pointers under `job:*`.

        The job database also holds `job:{id}:cancel`, `job:{id}:completed-items`
        and stage counters, and a record can be unreadable.  None of that may
        stop the sweep reaching the records it should fail.
        """
        await seed_job("good", status=JobStatus.processing, updated_at=_stale())
        raw = harness.raw(1)
        raw.set("job:good:cancel", "0")
        raw.sadd("job:good:completed-items", "a")
        raw.sadd("job:good:fr10:infer", "s1")
        raw.set("job:bad", "{not json")

        assert await _reap() == 1
        assert (await JobRepository().get("good")).status == JobStatus.failure
        assert raw.get("job:bad") == "{not json"

    async def test_the_reaper_tells_the_user_what_happened_and_what_to_do(self):
        """FO-48: guards BUG-54 -- FR-4 "marks the job failed with an error message"."""
        await seed_job("lost", status=JobStatus.processing, updated_at=_stale())

        await _reap()

        error = (await JobRepository().get("lost")).error
        assert error.code == "worker_lost"
        assert error.retryable is True
        assert "stopped responding" in error.message
        assert "again" in error.message

    async def test_an_outage_stops_the_reaper_without_changing_anything(self, harness):
        """FO-49: guards BUG-54 -- the sweep fails safe.

        A Redis error propagates and fails that run; nothing is half-reaped, and
        the next run after recovery does the work.
        """
        await seed_job("stuck", status=JobStatus.processing, updated_at=_stale())

        with harness.outage():
            with pytest.raises(RedisError):
                await _reap()
        assert (await JobRepository().get("stuck")).status == JobStatus.processing

        assert await _reap() == 1

    def test_the_reaper_is_scheduled(self):
        """FO-50: guards BUG-54 -- the beat entry that runs the sweep."""
        entry = celery_app.conf.beat_schedule["reap-stale-jobs"]

        assert entry["task"] == "app.worker.tasks.reap_stale_jobs"
        assert entry["schedule"] == settings.STALE_JOB_SWEEP_SECONDS
        assert entry["options"]["queue"] == "cpu"
        assert "app.worker.tasks.reap_stale_jobs" in celery_app.tasks
        assert settings.STALE_JOB_SWEEP_SECONDS < settings.STALE_JOB_SECONDS


class TestReaperEndToEnd:
    async def test_a_job_whose_worker_died_is_failed_and_reported(self, broker):
        """FO-51: guards BUG-54 -- the user-visible recovery.

        The worker dies after marking the job `processing`.  Until the sweep
        runs the client sees a running job; afterwards it sees a failure with a
        message, no result URL, and a result endpoint that exposes nothing.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            upload = await client.post("/upload", files={"file": ("clip.wav", wav_bytes(), "audio/wav")})
            submitted = await client.post("/jobs", json={
                "operation": "prediction", "model": "whisper-base", "audio_ids": [upload.json()["audio_id"]]})
            job_id = submitted.json()["job_id"]
            await JobRepository().update(job_id, status=JobStatus.processing)  # ...and then the worker died

            before = (await client.get(f"/jobs/{job_id}")).json()
            await _reap(now=_now() + timedelta(seconds=settings.STALE_JOB_SECONDS + 60))
            after = (await client.get(f"/jobs/{job_id}")).json()
            result = await client.get(f"/jobs/{job_id}/result")
            deleted = await client.delete(f"/jobs/{job_id}")

        assert before["status"] == "processing"
        assert after["status"] == "failure"
        assert after["error"]["code"] == "worker_lost"
        assert after["result_url"] is None
        assert result.status_code == 409
        assert deleted.status_code == 204

    async def test_an_outage_that_outlasts_the_retry_chain_is_reaped_after_recovery(self, harness, monkeypatch):
        """FO-52: guards BUG-54 -- the failure write that could not be written.

        Redis goes away mid-job and stays away for every retry.  The task's
        final "dependency unavailable" write needs Redis too, so it cannot land;
        the record is left `processing`.  After recovery the reaper fails it.
        """
        await seed_job("orphan")

        def model(*args, **kwargs):
            harness.disconnect()
            return {"text": "never stored"}

        monkeypatch.setattr(executor, "_execute_one", model)
        outcome = await _apply(tasks.execute_job, [envelope_for("orphan")])
        harness.reconnect()

        assert isinstance(outcome.result, RedisError)
        assert (await JobRepository().get("orphan")).status == JobStatus.processing
        await _reap(now=_now() + timedelta(seconds=settings.STALE_JOB_SECONDS + 60))
        record = await JobRepository().get("orphan")
        assert record.status == JobStatus.failure
        assert record.error.code == "worker_lost"


# --------------------------------------------------------------------------
# Stale cancel
# --------------------------------------------------------------------------


class TestStaleCancel:
    async def _session(self, client) -> str:
        return (await client.get("/session")).json()["sid"]

    async def test_delete_cancels_an_unresponsive_running_job_at_once(self, broker):
        """FO-53a: guards BUG-54 -- FR-14 "A running job can be canceled".

        DELETE on a running job only set a cancel flag for the worker to see.
        With the worker gone, nothing ever saw it: the job could be neither
        cancelled nor deleted.  A job silent for longer than the hard time limit
        cannot still be running its last step, so the cancel takes effect now.
        """
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = await self._session(client)
            await seed_job("stuck", session_id=sid, status=JobStatus.processing,
                           updated_at=_now() - timedelta(seconds=settings.TASK_TIME_LIMIT_SECONDS + 60))
            response = await client.delete("/jobs/stuck")
            status = (await client.get("/jobs/stuck")).json()
            deleted = await client.delete("/jobs/stuck")

        assert response.status_code == 202
        assert response.json()["status"] == "cancellation_requested"
        assert status["status"] == "cancelled"
        assert deleted.status_code == 204

    async def test_cancel_all_cancels_unresponsive_running_jobs_at_once(self, broker):
        """FO-53b: guards BUG-54 -- the session-wide fairness cancel, same rule."""
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = await self._session(client)
            await seed_job("fair", session_id=sid, status=JobStatus.processing, operation=JobOperation.fairness,
                           updated_at=_now() - timedelta(seconds=settings.TASK_TIME_LIMIT_SECONDS + 60))
            response = await client.post("/api/v1/analyses/fairness/cancel-all")
            status = (await client.get("/jobs/fair")).json()

        assert response.json()["cancelled_job_ids"] == ["fair"]
        assert status["status"] == "cancelled"

    async def test_a_responsive_running_job_is_still_cancelled_by_its_worker(self, broker):
        """FO-54: the FT-90 contract is unchanged for a job that is making progress."""
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = await self._session(client)
            await seed_job("live", session_id=sid, status=JobStatus.processing,
                           updated_at=_now() - timedelta(seconds=30))
            response = await client.delete("/jobs/live")
            status = (await client.get("/jobs/live")).json()

        assert response.json()["status"] == "cancellation_requested"
        assert status["status"] == "processing"
        assert await JobRepository().cancellation_requested("live")

    async def test_after_a_stale_cancel_a_late_worker_changes_nothing(self, broker, monkeypatch):
        """FO-55: guards BUG-54 and BUG-56.

        If the worker was only slow, it may still report back.  Its success
        write is refused by the terminal-state guard, and a redelivered copy of
        its task finds the job finished and does no work.
        """
        ran = []

        async def must_not_run(envelope, task_id):
            ran.append(task_id)

        monkeypatch.setattr(tasks, "execute", must_not_run)
        async with tolerant_client(headers={"Origin": ORIGIN}) as client:
            sid = await self._session(client)
            await seed_job("late", session_id=sid, status=JobStatus.processing,
                           updated_at=_now() - timedelta(seconds=settings.TASK_TIME_LIMIT_SECONDS + 60))
            await client.delete("/jobs/late")

            await JobRepository().update("late", status=JobStatus.success, result_key="results/x/late/result.json")
            await _apply(tasks.execute_job, [envelope_for("late", session_id=sid)])
            status = (await client.get("/jobs/late")).json()

        assert status["status"] == "cancelled"
        assert status["result_url"] is None
        assert ran == []


# --------------------------------------------------------------------------
# Convergence of the multi-stage pipelines
# --------------------------------------------------------------------------


def _fr7_spec() -> dict:
    return {"variant_id": "v1", "filename": "clip-0.wav", "object_key": "uploads/s1/x-0.wav",
            "baseline_sha256": "x", "audio_id": "a", "property": "pitch", "theta": 1.0,
            "repeat": 0, "is_control": False}


class TestPipelineConvergence:
    async def test_an_fr7_render_that_fails_fails_its_job(self, monkeypatch):
        """FO-56: guards BUG-55.

        A variant render that raised left its chord without a callback: the job
        stayed `processing` with no error, whatever went wrong.
        """
        await seed_job("sweep", status=JobStatus.processing, operation=JobOperation.linguistic_acoustic)

        async def unreadable(envelope, spec):
            raise ValueError("baseline audio is unreadable")

        monkeypatch.setattr(linguistic_acoustic_service, "render_and_register", unreadable)
        await _apply(tasks.fr7_render_variant, [envelope_for("sweep", operation="linguistic_acoustic"), _fr7_spec()])

        record = await JobRepository().get("sweep")
        assert record.status == JobStatus.failure
        assert record.error.code == "fr7_render_failed"
        assert "unreadable" in record.error.message

    async def test_an_fr7_inference_whose_retries_run_out_fails_its_job(self, monkeypatch):
        """FO-57: guards BUG-55 -- the exhausted retry chain in an FR-7 child."""
        await seed_job("sweep", status=JobStatus.processing, operation=JobOperation.linguistic_acoustic)
        attempts = []

        async def store_down(envelope, rendered, task_id):
            attempts.append(task_id)
            raise StorageError("object store unavailable")

        monkeypatch.setattr(linguistic_acoustic_service, "infer_variant", store_down)
        await _apply(tasks.fr7_infer_variant,
                     [envelope_for("sweep", operation="linguistic_acoustic"), {"variant_id": "v1"}])

        assert len(attempts) == tasks.fr7_infer_variant.max_retries + 1
        record = await JobRepository().get("sweep")
        assert record.status == JobStatus.failure
        assert record.error.code == "dependency_unavailable"

    async def test_an_fr10_shard_with_a_corrupt_plan_fails_its_job(self, harness):
        """FO-58: guards BUG-55 -- invalid data under an FR-10 shard.

        The shard reads its plan from the object store outside its per-item
        error handling, so a damaged plan raised out of the shard -- and, like
        FR-7, stalled the chord with the job left `processing`.
        """
        await seed_job("fair", status=JobStatus.processing, operation=JobOperation.fairness)
        path = get_storage().path_for(fairness_service._plan_key("s1", "fair"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"index": {"groups": [')

        await _apply(tasks.fr10_infer_shard, [
            envelope_for("fair", operation="fairness"), {"shard_id": "s1", "group_label": "g", "item_ids": ["i1"]}])

        record = await JobRepository().get("fair")
        assert record.status == JobStatus.failure
        assert record.error.code == "fr10_infer_failed"

    async def test_an_fr10_shard_whose_retries_run_out_still_degrades(self, monkeypatch):
        """FO-59: the designed FR-10 behaviour is kept -- a dead shard shrinks its group."""
        await seed_job("fair", status=JobStatus.processing, operation=JobOperation.fairness)

        async def store_down(envelope, shard, task_id):
            raise StorageError("object store unavailable")

        monkeypatch.setattr(fairness_service, "infer_shard", store_down)
        outcome = await _apply(tasks.fr10_infer_shard, [
            envelope_for("fair", operation="fairness"), {"shard_id": "s1", "group_label": "g", "item_ids": ["i1", "i2"]}])

        assert outcome.successful()
        assert outcome.result["n_failed"] == 2
        assert (await JobRepository().get("fair")).status == JobStatus.processing

    @pytest.mark.parametrize(("task_name", "module", "attribute", "operation"), [
        ("orchestrate_batch", tasks, "complete_batch_from_cache", "prediction"),
        ("fr7_orchestrate", linguistic_acoustic_service, "complete_sweep_from_cache", "linguistic_acoustic"),
        ("fr10_orchestrate", fairness_service, "complete_from_cache", "fairness"),
    ])
    async def test_an_orchestrator_whose_dependency_stays_down_is_retried_then_failed(
        self, monkeypatch, task_name, module, attribute, operation
    ):
        """FO-60: guards BUG-55.

        The orchestrators had no error handling at all: one Redis or storage
        error and the job stayed `queued` -- the one state the reaper must
        leave alone, because it looks like a job still waiting for a worker.
        """
        await seed_job("orch", operation=JobOperation(operation))
        attempts = []

        async def store_down(envelope):
            attempts.append(1)
            raise StorageError("object store unavailable")

        monkeypatch.setattr(module, attribute, store_down)
        await _apply(getattr(tasks, task_name), [envelope_for("orch", operation=operation, items=2)])

        assert len(attempts) == 4
        record = await JobRepository().get("orch")
        assert record.status == JobStatus.failure
        assert record.error.code == "dependency_unavailable"

    @pytest.mark.parametrize("task_name", ["orchestrate_batch", "fr7_orchestrate", "fr10_orchestrate"])
    async def test_an_orchestrator_that_cannot_reach_the_broker_is_retried_then_failed(
        self, monkeypatch, task_name
    ):
        """FO-61: guards BUG-55 -- communication interruption to the broker at dispatch."""
        operation = {"orchestrate_batch": "prediction", "fr7_orchestrate": "linguistic_acoustic",
                     "fr10_orchestrate": "fairness"}[task_name]
        await seed_job("orch", operation=JobOperation(operation))

        async def not_cached(envelope):
            return False

        async def prepared(envelope, task_id):
            return [{"shard_id": "s1"}]

        monkeypatch.setattr(tasks, "complete_batch_from_cache", not_cached)
        monkeypatch.setattr(linguistic_acoustic_service, "complete_sweep_from_cache", not_cached)
        monkeypatch.setattr(linguistic_acoustic_service, "prepare_sweep", prepared)
        monkeypatch.setattr(fairness_service, "complete_from_cache", not_cached)
        monkeypatch.setattr(fairness_service, "prepare_analysis", prepared)
        dispatches = []

        def broker_down(signatures):
            def publish(callback):
                dispatches.append(1)
                raise OperationalError("Error 10061 connecting to redis:6379")
            return publish

        monkeypatch.setattr(tasks, "chord", broker_down)
        await _apply(getattr(tasks, task_name), [envelope_for("orch", operation=operation, items=2)])

        assert len(dispatches) == 4
        record = await JobRepository().get("orch")
        assert record.status == JobStatus.failure
        assert record.error.code == "dependency_unavailable"

    @pytest.mark.parametrize(("task_name", "module", "attribute", "operation"), [
        ("fr7_dispatch_inference", linguistic_acoustic_service, "mark_render_complete", "linguistic_acoustic"),
        ("fr10_dispatch_explain", fairness_service, "mark_infer_complete", "fairness"),
    ])
    async def test_a_dispatcher_that_fails_fails_its_job(self, monkeypatch, task_name, module, attribute, operation):
        """FO-62: guards BUG-55 -- the chord callbacks between FR-7/FR-10 stages."""
        await seed_job("stage", status=JobStatus.processing, operation=JobOperation(operation))

        async def broken(envelope, results):
            raise ValueError("stage results are unreadable")

        monkeypatch.setattr(module, attribute, broken)
        await _apply(getattr(tasks, task_name), [[], envelope_for("stage", operation=operation)])

        record = await JobRepository().get("stage")
        assert record.status == JobStatus.failure
        assert "unreadable" in record.error.message

    async def test_a_failure_recording_child_ids_does_not_dispatch_again(self, monkeypatch):
        """FO-63: guards BUG-55 -- a retry must never publish a second chord.

        Recording `child_task_ids` after the chord is published is bookkeeping
        (it lets DELETE revoke the children).  If that write fails, retrying the
        orchestrator would publish every item a second time.
        """
        await seed_job("once", status=JobStatus.queued)
        dispatches = []

        async def not_cached(envelope):
            return False

        def counting_chord(signatures):
            def publish(callback):
                dispatches.append(len(signatures))
                children = [SimpleNamespace(id=f"child-{i}") for i in range(len(signatures))]
                return SimpleNamespace(parent=SimpleNamespace(results=children))
            return publish

        real_update = JobRepository.update

        async def refuse_bookkeeping(self, job_id, **fields):
            if "child_task_ids" in fields:
                raise RedisConnectionError("job database blipped")
            return await real_update(self, job_id, **fields)

        monkeypatch.setattr(tasks, "complete_batch_from_cache", not_cached)
        monkeypatch.setattr(tasks, "chord", counting_chord)
        monkeypatch.setattr(JobRepository, "update", refuse_bookkeeping)
        outcome = await _apply(tasks.orchestrate_batch, [envelope_for("once", items=3)])

        assert dispatches == [3]
        assert outcome.successful(), outcome.result
        assert (await JobRepository().get("once")).status == JobStatus.queued

    async def test_convergence_never_overwrites_the_error_a_stage_recorded(self, monkeypatch):
        """FO-64: guards BUG-55 -- the first recorded reason is the one the user sees.

        The executor records its own error before re-raising; the task's
        failure handling must leave that error in place, not replace it with a
        generic one.
        """
        await seed_job("reasoned", status=JobStatus.processing)

        async def model_error(envelope, task_id):
            await JobRepository().update("reasoned", status=JobStatus.failure,
                                         error=JobError(code="model_error", message="CUDA out of memory"))
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(tasks, "execute", model_error)
        await _apply(tasks.execute_job, [envelope_for("reasoned")])

        record = await JobRepository().get("reasoned")
        assert record.error.code == "model_error"
        assert record.error.message == "CUDA out of memory"
