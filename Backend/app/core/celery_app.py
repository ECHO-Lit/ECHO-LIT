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
    broker_connection_retry_on_startup=True,
    timezone="UTC",
    beat_schedule={
        "cleanup-expired-local-objects": {
            "task": "app.worker.tasks.cleanup_expired_local_objects",
            "schedule": 60 * 60,
            "options": {"queue": "cpu"},
        }
    },
)


def queue_for(operation: str, model: str | None) -> str:
    if operation in {"perturbation", "audio_features", "linguistic_acoustic", "fairness"}:
        return "cpu"
    if operation in {"saliency", "attention", "jacobian_lens_fit", "jacobian_lens_apply"} or model == "whisper-large":
        return "gpu-large"
    # `hidden_states` and `layer_probe` are encoder forward passes, the same
    # shape of work as `embedding`, so they share its routing.
    return "gpu-fast"
