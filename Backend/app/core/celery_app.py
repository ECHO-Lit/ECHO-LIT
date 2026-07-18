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


def queue_for(
    operation: str,
    model: str | None,
    execution_profile: str | None = None,
) -> str:
    profile = execution_profile or settings.EXECUTION_PROFILE
    if profile == "mock":
        return "mock"
    if operation in {"perturbation", "audio_features"}:
        return "cpu"
    if operation in {"saliency", "attention"} or model == "whisper-large":
        return "gpu-large"
    return "gpu-fast"


def finalization_queue_for(execution_profile: str | None = None) -> str:
    profile = execution_profile or settings.EXECUTION_PROFILE
    return "mock" if profile == "mock" else "cpu"
