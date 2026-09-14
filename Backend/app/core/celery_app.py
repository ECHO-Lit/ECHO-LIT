import asyncio
from collections.abc import Iterable

from celery import Celery

from .settings import settings


celery_app = Celery(
    "echo",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    result_expires=settings.JOB_TTL_SECONDS,
    task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT_SECONDS,
    task_time_limit=settings.TASK_TIME_LIMIT_SECONDS,
    # kombu's Redis transport redelivers an unacknowledged task after this
    # long. Its 3600 s default equalled the hard time limit, so a task still
    # running near the limit was handed to a second worker.
    broker_transport_options={"visibility_timeout": settings.BROKER_VISIBILITY_TIMEOUT_SECONDS},
    broker_connection_retry_on_startup=True,
    timezone="UTC",
    beat_schedule={
        "cleanup-expired-local-objects": {
            "task": "app.worker.tasks.cleanup_expired_local_objects",
            "schedule": 60 * 60,
            "options": {"queue": "cpu"},
        },
        # Custom datasets live outside the object store the task above sweeps.
        "cleanup-expired-session-datasets": {
            "task": "app.worker.tasks.cleanup_expired_session_datasets",
            "schedule": 60 * 60,
            "options": {"queue": "cpu"},
        },
        # Fails jobs whose worker stopped without recording an outcome.
        "reap-stale-jobs": {
            "task": "app.worker.tasks.reap_stale_jobs",
            "schedule": settings.STALE_JOB_SWEEP_SECONDS,
            "options": {"queue": "cpu", "expires": settings.STALE_JOB_SWEEP_SECONDS},
        },
    },
)


async def send_task_async(name: str, **options):
    """Publish a task without blocking the API's event loop.

    `send_task` is a synchronous network publish to the broker. Called directly
    from an async route it stalls every in-flight request on that worker for as
    long as the broker takes to answer -- which is longest exactly when the
    broker is congested or reconnecting. Celery's producer pool is safe to use
    from worker threads.
    """
    return await asyncio.to_thread(celery_app.send_task, name, **options)


async def revoke_async(task_ids: Iterable[str | None]) -> None:
    """Revoke tasks off the event loop; `control.revoke` is a broker broadcast."""
    ids = [task_id for task_id in task_ids if task_id]
    if not ids:
        return

    def _revoke() -> None:
        for task_id in ids:
            celery_app.control.revoke(task_id, terminate=False)

    await asyncio.to_thread(_revoke)


def queue_for(operation: str, model: str | None) -> str:
    if operation in {"perturbation", "audio_features", "linguistic_acoustic", "fairness"}:
        return "cpu"
    if operation in {"saliency", "attention", "jacobian_lens_fit", "jacobian_lens_apply"} or model == "whisper-large":
        return "gpu-large"
    # `hidden_states` and `layer_probe` are encoder forward passes, the same
    # shape of work as `embedding`, so they share its routing.
    return "gpu-fast"
