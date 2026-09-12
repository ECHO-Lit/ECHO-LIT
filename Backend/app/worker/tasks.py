from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import time

from celery.exceptions import TimeLimitExceeded
from celery.signals import heartbeat_sent, task_failure
from kombu.exceptions import OperationalError
from pydantic import ValidationError
from redis import from_url as sync_redis_from_url
from redis.exceptions import RedisError

from celery import chord

from app.core.celery_app import celery_app, queue_for
from app.core.heartbeat import record_worker_heartbeat
from app.core.storage import StorageError
from app.core.settings import settings
from app.repositories.jobs import TERMINAL_STATES, JobRepository
from app.schemas.jobs import JobError, JobStatus
from app.worker.custom_model_validation import validate_custom_model
from app.worker.executor import complete_batch_from_cache, execute, execute_batch_item, finalize_batch
from app.worker.recovery import describe_failure, finish_failed_sync, hard_time_limit_error, job_id_from_args


logger = logging.getLogger(__name__)


_heartbeat_redis = sync_redis_from_url(settings.JOB_REDIS_URL, decode_responses=True)
# Session keys live on the session database (DB0), not the job database.
_session_redis = sync_redis_from_url(settings.REDIS_URL, decode_responses=True)

# One persistent event loop per worker process. _run() creates and
# closes a fresh loop per task, but the module-level redis.asyncio clients
# pool connections bound to whichever loop first used them — a later task on
# a new loop then hits "RuntimeError: Event loop is closed".
_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


# A child process that dies natively (SIGSEGV, OOM kill) never reaches the task's
# except blocks, and task_reject_on_worker_lost puts the message straight back on the
# queue -- so a deterministic native crash is redelivered forever, killing a fresh
# pool child each time. Deliveries are counted per (task id, retry number) because
# self.retry() reuses the task id; a crash redelivery repeats the same pair.
MAX_DELIVERIES_PER_ATTEMPT = 3


class RepeatedWorkerCrash(RuntimeError):
    pass


def _fail_if_redelivered_too_often(task, job_id: str) -> None:
    key = f"task-deliveries:{task.request.id}:{task.request.retries}"
    try:
        pipe = _heartbeat_redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, settings.JOB_TTL_SECONDS)
        deliveries = pipe.execute()[0]
    except RedisError:
        # The guard is a backstop; an unreachable counter must not block the job itself.
        return
    if deliveries <= MAX_DELIVERIES_PER_ATTEMPT:
        return

    _run(JobRepository().update(
        job_id,
        status=JobStatus.failure,
        error=JobError(
            code="worker_crashed",
            message="The worker process crashed repeatedly while running this job",
            retryable=False,
        ),
    ))
    raise RepeatedWorkerCrash(
        f"task {task.request.id} lost its worker {deliveries - 1} times; not running it again"
    )


@heartbeat_sent.connect
def publish_worker_heartbeat(sender=None, **kwargs) -> None:
    del kwargs
    # The signal's sender is Celery's Heart, which has no .hostname of its own;
    # its event dispatcher does. str(sender) would be an object repr carrying a
    # memory address -- a new member on every restart.
    hostname = (
        getattr(sender, "hostname", None)
        or getattr(getattr(sender, "eventer", None), "hostname", None)
        or str(sender or "worker")
    )
    record_worker_heartbeat(_heartbeat_redis, hostname, time.time())


@task_failure.connect
def fail_job_on_hard_time_limit(sender=None, task_id=None, exception=None, args=None, **kwargs) -> None:
    """FR-4: a task stopped at the hard time limit fails its job with a message.

    billiard kills the pool child at the hard limit, so nothing inside the task
    runs; Celery acknowledges the message and records the failure only in its
    own result backend, leaving the job record `processing`.  This receiver
    runs in the worker's main process, so it writes through the synchronous
    client -- the asyncio clients belong to the pool children's event loops.
    """
    del sender, task_id, kwargs
    if not isinstance(exception, TimeLimitExceeded):
        return
    job_id = job_id_from_args(args)
    if not job_id:
        return
    try:
        finish_failed_sync(_heartbeat_redis, job_id, hard_time_limit_error())
    except RedisError:
        logger.warning("could not record the hard time limit for job %s; the reaper will", job_id)


# Failures that can clear on their own: the object store, Redis, the broker.
TRANSIENT = (StorageError, RedisError, OperationalError)


def _job_finished(job_id: str, *, missing: bool = True) -> bool:
    """Whether a job needs no more work: it finished, its record is unreadable,
    or -- unless `missing` says otherwise -- it no longer exists.

    A task can outlive its job: the reaper failed it, its owner cancelled it,
    or the broker redelivered a task another worker was still running.
    """
    try:
        record = _run(JobRepository().get(job_id))
    except ValidationError:
        logger.warning("job %s has an unreadable record; not running its task", job_id)
        return True
    if record is None:
        return missing
    return record.status in TERMINAL_STATES


def _finish_failed(job_id: str, error: JobError) -> None:
    """Record why a job failed.  If Redis is down as well, the reaper fails it later."""
    try:
        _run(JobRepository().finish(job_id, JobStatus.failure, error=error))
    except RedisError:
        logger.warning("could not record the failure of job %s; the reaper will", job_id)


def _converge(task, envelope: dict, work, *, code: str, degrade=None, missing: bool = True):
    """Run a job task's `work` so that its job always reaches an outcome.

    A finished job's task does nothing.  A transient failure is retried with
    back-off; when the retries run out the job fails as `dependency_unavailable`
    -- or, where a stage can lose a part and still finish (FR-10 shards),
    `degrade` supplies that part's result instead.  Any other exception fails
    the job with its message.  Failing never overwrites an error the stage
    already recorded.
    """
    job_id = envelope["job_id"]
    try:
        if _job_finished(job_id, missing=missing):
            logger.info("skipping %s: job %s is finished", task.name, job_id)
            return None
        return work()
    except TRANSIENT as exc:
        if task.request.retries < task.max_retries:
            raise task.retry(exc=exc, countdown=min(2 ** task.request.retries, 30))
        if degrade is not None:
            return degrade(exc)
        _finish_failed(job_id, JobError(
            code="dependency_unavailable",
            message="A required dependency remained unavailable",
            retryable=False,
        ))
        raise
    except Exception as exc:
        _finish_failed(job_id, describe_failure(exc, code))
        raise


def _record_children(job_id: str, task_id: str, result) -> None:
    """Note a published chord's children so DELETE can revoke them.

    Bookkeeping only, done after the chord is published and never retried: a
    retry of the orchestrator would publish every child a second time.
    """
    if result is None:
        return
    child_ids = [child.id for child in (result.parent.results if result.parent else [])]
    try:
        _run(JobRepository().update(job_id, task_id=task_id, child_task_ids=child_ids))
    except RedisError:
        logger.warning("could not record the child tasks of job %s", job_id)


def _dead_shard(shard: dict, operation: str) -> dict:
    """An FR-10 shard whose retries ran out: every item failed, and the
    aggregator shrinks the group instead of failing the analysis."""
    result = {
        "shard_id": shard["shard_id"], "group_label": shard["group_label"],
        "operation": operation, "n_items": len(shard["item_ids"]),
    }
    if operation == "prediction":
        result["n_cached"] = 0
    result.update({
        "n_failed": len(shard["item_ids"]), "artifact_key": None,
        "failures": [{"item_id": i, "code": "dependency_unavailable"} for i in shard["item_ids"]],
    })
    return result


@celery_app.task(
    bind=True,
    name="app.worker.tasks.execute_job",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
)
def execute_job(self, envelope: dict) -> None:
    _fail_if_redelivered_too_often(self, envelope["job_id"])
    _converge(self, envelope, lambda: _run(execute(envelope, self.request.id)), code="execution_failed")


@celery_app.task(bind=True, name="app.worker.tasks.validate_custom_model", acks_late=True)
def validate_registered_custom_model(self, model_id: str) -> None:
    del self
    _run(validate_custom_model(model_id))


@celery_app.task(bind=True, name="app.worker.tasks.orchestrate_batch", max_retries=3)
def orchestrate_batch(self, envelope: dict) -> None:
    def dispatch():
        if _run(complete_batch_from_cache(envelope)):
            return None
        operation = envelope["operation"]
        model = envelope.get("model")
        signatures = [
            execute_job_item.s(envelope, index).set(queue=queue_for(operation, model))
            for index in range(len(envelope["audio"]))
        ]
        callback = finalize_batch_job.s(envelope).set(queue="cpu")
        return chord(signatures)(callback)

    result = _converge(self, envelope, dispatch, code="batch_dispatch_failed")
    _record_children(envelope["job_id"], self.request.id, result)


@celery_app.task(
    bind=True,
    name="app.worker.tasks.execute_job_item",
    acks_late=True,
    max_retries=3,
)
def execute_job_item(self, envelope: dict, asset_index: int) -> dict:
    _fail_if_redelivered_too_often(self, envelope["job_id"])
    return _converge(
        self, envelope, lambda: _run(execute_batch_item(envelope, asset_index, self.request.id)),
        code="batch_item_failed",
    )


@celery_app.task(name="app.worker.tasks.finalize_batch_job", acks_late=True)
def finalize_batch_job(child_results: list[dict], envelope: dict) -> None:
    try:
        if _job_finished(envelope["job_id"]):
            return
        _run(finalize_batch(child_results, envelope))
    except Exception as exc:
        _finish_failed(envelope["job_id"], describe_failure(exc, "batch_finalize_failed"))
        raise


@celery_app.task(bind=True, name="app.worker.tasks.fr7_orchestrate", acks_late=True, max_retries=3)
def fr7_orchestrate(self, envelope: dict) -> None:
    from app.services.linguistic_acoustic_service import complete_sweep_from_cache, prepare_sweep

    def dispatch():
        if _run(complete_sweep_from_cache(envelope)):
            return None
        specs = _run(prepare_sweep(envelope, self.request.id))
        signatures = [fr7_render_variant.s(envelope, spec).set(queue="cpu") for spec in specs]
        return chord(signatures)(fr7_dispatch_inference.s(envelope).set(queue="cpu"))

    result = _converge(self, envelope, dispatch, code="fr7_dispatch_failed")
    _record_children(envelope["job_id"], self.request.id, result)


@celery_app.task(
    bind=True, name="app.worker.tasks.fr7_render_variant", acks_late=True, max_retries=3
)
def fr7_render_variant(self, envelope: dict, spec: dict) -> dict:
    from app.services.linguistic_acoustic_service import render_and_register

    # `missing=False`: the on-demand playback render (POST /audio/{id}/variant)
    # runs this task for a job id that has no record.
    return _converge(
        self, envelope, lambda: _run(render_and_register(envelope, spec)),
        code="fr7_render_failed", missing=False,
    )


@celery_app.task(bind=True, name="app.worker.tasks.fr7_dispatch_inference", acks_late=True, max_retries=3)
def fr7_dispatch_inference(self, rendered: list[dict], envelope: dict) -> None:
    from app.services.linguistic_acoustic_service import mark_render_complete

    def dispatch():
        _run(mark_render_complete(envelope, rendered))
        queue = queue_for("prediction", envelope.get("model"))
        runnable = [item for item in rendered if item.get("applicable")]
        if not runnable:
            fr7_aggregate.apply_async(args=[[], envelope, rendered], queue="cpu")
            return
        signatures = [fr7_infer_variant.s(envelope, item).set(queue=queue) for item in runnable]
        chord(signatures)(fr7_aggregate.s(envelope, rendered).set(queue="cpu"))

    _converge(self, envelope, dispatch, code="fr7_dispatch_failed")


@celery_app.task(
    bind=True, name="app.worker.tasks.fr7_infer_variant", acks_late=True, max_retries=2
)
def fr7_infer_variant(self, envelope: dict, rendered: dict) -> dict:
    from app.services.linguistic_acoustic_service import infer_variant

    return _converge(
        self, envelope, lambda: _run(infer_variant(envelope, rendered, self.request.id)),
        code="fr7_infer_failed",
    )


@celery_app.task(name="app.worker.tasks.fr7_aggregate", acks_late=True)
def fr7_aggregate(outputs: list[dict], envelope: dict, rendered: list[dict]) -> None:
    from app.services.linguistic_acoustic_service import aggregate_sweep

    try:
        if _job_finished(envelope["job_id"]):
            return
        _run(aggregate_sweep(outputs, rendered, envelope))
    except Exception as exc:
        _finish_failed(envelope["job_id"], describe_failure(exc, "fr7_aggregate_failed"))
        raise


@celery_app.task(bind=True, name="app.worker.tasks.fr10_orchestrate", acks_late=True, max_retries=3)
def fr10_orchestrate(self, envelope: dict) -> None:
    from app.services.fairness_service import FairnessInputError, complete_from_cache, prepare_analysis

    def dispatch():
        if _run(complete_from_cache(envelope)):
            return None
        try:
            infer_shards = _run(prepare_analysis(envelope, self.request.id))
        except FairnessInputError:
            return None  # prepare_analysis already wrote the JobError; nothing to dispatch.
        if not infer_shards:
            fr10_aggregate.apply_async(args=[[], [], envelope], queue="cpu")
            return None
        queue = queue_for("prediction", envelope.get("model"))
        signatures = [fr10_infer_shard.s(envelope, shard).set(queue=queue) for shard in infer_shards]
        return chord(signatures)(fr10_dispatch_explain.s(envelope).set(queue="cpu"))

    result = _converge(self, envelope, dispatch, code="fr10_dispatch_failed")
    _record_children(envelope["job_id"], self.request.id, result)


@celery_app.task(bind=True, name="app.worker.tasks.fr10_infer_shard", acks_late=True, max_retries=2)
def fr10_infer_shard(self, envelope: dict, shard: dict) -> dict:
    from app.services.fairness_service import infer_shard

    # A dead shard degrades the run rather than killing it: the aggregator
    # sees n_failed == n_items and shrinks the group.
    return _converge(
        self, envelope, lambda: _run(infer_shard(envelope, shard, self.request.id)),
        code="fr10_infer_failed", degrade=lambda exc: _dead_shard(shard, "prediction"),
    )


@celery_app.task(bind=True, name="app.worker.tasks.fr10_dispatch_explain", acks_late=True, max_retries=3)
def fr10_dispatch_explain(self, infer_results: list[dict], envelope: dict) -> None:
    from app.services.fairness_service import load_plan, mark_infer_complete

    def dispatch():
        _run(mark_infer_complete(envelope, infer_results))
        if not envelope["parameters"].get("include_explanations", True):
            fr10_aggregate.apply_async(args=[[], infer_results, envelope], queue="cpu")
            return
        plan = _run(load_plan(envelope))
        shards = plan["plan"]["explain_shards"]
        if not shards:
            fr10_aggregate.apply_async(args=[[], infer_results, envelope], queue="cpu")
            return
        signatures = [fr10_explain_shard.s(envelope, shard).set(queue="gpu-large") for shard in shards]
        chord(signatures)(fr10_aggregate.s(infer_results, envelope).set(queue="cpu"))

    _converge(self, envelope, dispatch, code="fr10_dispatch_failed")


@celery_app.task(bind=True, name="app.worker.tasks.fr10_explain_shard", acks_late=True, max_retries=1)
def fr10_explain_shard(self, envelope: dict, shard: dict) -> dict:
    from app.services.fairness_service import explain_shard

    return _converge(
        self, envelope, lambda: _run(explain_shard(envelope, shard, self.request.id)),
        code="fr10_explain_failed", degrade=lambda exc: _dead_shard(shard, "saliency"),
    )


@celery_app.task(name="app.worker.tasks.fr10_aggregate", acks_late=True)
def fr10_aggregate(explain_results: list[dict], infer_results: list[dict], envelope: dict) -> None:
    from app.services.fairness_service import aggregate_fairness

    try:
        if _job_finished(envelope["job_id"]):
            return
        _run(aggregate_fairness(explain_results, infer_results, envelope))
    except Exception as exc:
        _finish_failed(envelope["job_id"], describe_failure(exc, "fr10_aggregate_failed"))
        raise


@celery_app.task(name="app.worker.tasks.reap_stale_jobs")
def reap_stale_jobs_task() -> int:
    """Fail jobs whose worker stopped without recording an outcome.

    A Redis error fails the run having changed nothing; the next run retries.
    """
    from app.worker.recovery import reap_stale_jobs

    return _run(reap_stale_jobs())


@celery_app.task(name="app.worker.tasks.cleanup_expired_session_datasets")
def cleanup_expired_session_datasets_task() -> int:
    """PE-3: remove custom datasets whose owning session has expired.

    A Redis error propagates and fails the run, so an unreachable Redis can
    never be mistaken for "every session is gone".
    """
    from app.core.redis import k_meta
    from app.services.custom_dataset_service import cleanup_expired_session_datasets

    return cleanup_expired_session_datasets(lambda sid: bool(_session_redis.exists(k_meta(sid))))


@celery_app.task(name="app.worker.tasks.cleanup_expired_local_objects")
def cleanup_expired_local_objects() -> int:
    if settings.STORAGE_BACKEND.lower() != "local":
        return 0
    root = Path(settings.STORAGE_LOCAL_ROOT).resolve()
    if not root.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - settings.JOB_TTL_SECONDS
    deleted = 0
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            deleted += 1
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return deleted
