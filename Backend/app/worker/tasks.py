from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from celery.signals import heartbeat_sent
from redis import from_url as sync_redis_from_url
from redis.exceptions import RedisError

from celery import chord

from app.core.celery_app import celery_app, queue_for
from app.core.storage import StorageError
from app.core.settings import settings
from app.repositories.jobs import JobRepository
from app.worker.custom_model_validation import validate_custom_model
from app.worker.executor import complete_batch_from_cache, execute, execute_batch_item, finalize_batch


_heartbeat_redis = sync_redis_from_url(settings.JOB_REDIS_URL, decode_responses=True)

# One persistent event loop per worker process. _run() creates and
# closes a fresh loop per task, but the module-level redis.asyncio clients
# pool connections bound to whichever loop first used them — a later task on
# a new loop then hits "RuntimeError: Event loop is closed".
_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


@heartbeat_sent.connect
def publish_worker_heartbeat(sender=None, **kwargs) -> None:
    del kwargs
    hostname = getattr(sender, "hostname", None) or str(sender or "worker")
    _heartbeat_redis.set(f"worker-heartbeat:{hostname}", "1", ex=90)


@celery_app.task(
    bind=True,
    name="app.worker.tasks.execute_job",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
)
def execute_job(self, envelope: dict) -> None:
    try:
        _run(execute(envelope, self.request.id))
    except (StorageError, RedisError) as exc:
        if self.request.retries >= self.max_retries:
            from app.schemas.jobs import JobError, JobStatus

            _run(JobRepository().update(
                envelope["job_id"],
                status=JobStatus.failure,
                error=JobError(
                    code="dependency_unavailable",
                    message="A required dependency remained unavailable",
                    retryable=False,
                ),
            ))
            raise
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries, 30))


@celery_app.task(bind=True, name="app.worker.tasks.validate_custom_model", acks_late=True)
def validate_registered_custom_model(self, model_id: str) -> None:
    del self
    _run(validate_custom_model(model_id))


@celery_app.task(bind=True, name="app.worker.tasks.orchestrate_batch")
def orchestrate_batch(self, envelope: dict) -> None:
    if _run(complete_batch_from_cache(envelope)):
        return
    operation = envelope["operation"]
    model = envelope.get("model")
    signatures = [
        execute_job_item.s(envelope, index).set(queue=queue_for(operation, model))
        for index in range(len(envelope["audio"]))
    ]
    callback = finalize_batch_job.s(envelope).set(queue="cpu")
    result = chord(signatures)(callback)
    child_ids = [child.id for child in (result.parent.results if result.parent else [])]
    _run(JobRepository().update(
        envelope["job_id"],
        task_id=self.request.id,
        child_task_ids=child_ids,
    ))


@celery_app.task(
    bind=True,
    name="app.worker.tasks.execute_job_item",
    acks_late=True,
    max_retries=3,
)
def execute_job_item(self, envelope: dict, asset_index: int) -> dict:
    try:
        return _run(execute_batch_item(envelope, asset_index, self.request.id))
    except (StorageError, RedisError) as exc:
        if self.request.retries >= self.max_retries:
            from app.schemas.jobs import JobError, JobStatus

            _run(JobRepository().update(
                envelope["job_id"],
                status=JobStatus.failure,
                error=JobError(
                    code="dependency_unavailable",
                    message="A batch dependency remained unavailable",
                    retryable=False,
                ),
            ))
            raise
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries, 30))


@celery_app.task(name="app.worker.tasks.finalize_batch_job", acks_late=True)
def finalize_batch_job(child_results: list[dict], envelope: dict) -> None:
    try:
        _run(finalize_batch(child_results, envelope))
    except Exception as exc:
        from app.schemas.jobs import JobError, JobStatus

        _run(JobRepository().update(
            envelope["job_id"],
            status=JobStatus.failure,
            error=JobError(code="batch_finalize_failed", message=str(exc)[:500], retryable=False),
        ))
        raise


@celery_app.task(bind=True, name="app.worker.tasks.fr7_orchestrate", acks_late=True)
def fr7_orchestrate(self, envelope: dict) -> None:
    from app.services.linguistic_acoustic_service import complete_sweep_from_cache, prepare_sweep

    if _run(complete_sweep_from_cache(envelope)):
        return
    specs = _run(prepare_sweep(envelope, self.request.id))
    signatures = [fr7_render_variant.s(envelope, spec).set(queue="cpu") for spec in specs]
    result = chord(signatures)(fr7_dispatch_inference.s(envelope).set(queue="cpu"))
    child_ids = [child.id for child in (result.parent.results if result.parent else [])]
    _run(JobRepository().update(envelope["job_id"], task_id=self.request.id, child_task_ids=child_ids))


@celery_app.task(
    bind=True, name="app.worker.tasks.fr7_render_variant", acks_late=True, max_retries=3
)
def fr7_render_variant(self, envelope: dict, spec: dict) -> dict:
    from app.services.linguistic_acoustic_service import render_and_register

    try:
        return _run(render_and_register(envelope, spec))
    except (StorageError, RedisError) as exc:
        if self.request.retries >= self.max_retries:
            raise
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries, 30))


@celery_app.task(bind=True, name="app.worker.tasks.fr7_dispatch_inference", acks_late=True)
def fr7_dispatch_inference(self, rendered: list[dict], envelope: dict) -> None:
    from app.services.linguistic_acoustic_service import mark_render_complete

    _run(mark_render_complete(envelope, rendered))
    queue = queue_for("prediction", envelope.get("model"))
    runnable = [item for item in rendered if item.get("applicable")]
    if not runnable:
        fr7_aggregate.apply_async(args=[[], envelope, rendered], queue="cpu")
        return
    signatures = [fr7_infer_variant.s(envelope, item).set(queue=queue) for item in runnable]
    chord(signatures)(fr7_aggregate.s(envelope, rendered).set(queue="cpu"))


@celery_app.task(
    bind=True, name="app.worker.tasks.fr7_infer_variant", acks_late=True, max_retries=2
)
def fr7_infer_variant(self, envelope: dict, rendered: dict) -> dict:
    from app.services.linguistic_acoustic_service import infer_variant

    try:
        return _run(infer_variant(envelope, rendered, self.request.id))
    except (StorageError, RedisError) as exc:
        if self.request.retries >= self.max_retries:
            raise
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries, 30))


@celery_app.task(name="app.worker.tasks.fr7_aggregate", acks_late=True)
def fr7_aggregate(outputs: list[dict], envelope: dict, rendered: list[dict]) -> None:
    from app.services.linguistic_acoustic_service import aggregate_sweep

    try:
        _run(aggregate_sweep(outputs, rendered, envelope))
    except Exception as exc:
        from app.schemas.jobs import JobError, JobStatus

        _run(JobRepository().update(
            envelope["job_id"],
            status=JobStatus.failure,
            error=JobError(code="fr7_aggregate_failed", message=str(exc)[:500], retryable=False),
        ))
        raise


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
