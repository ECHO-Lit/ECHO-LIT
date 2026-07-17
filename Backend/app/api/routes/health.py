import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core import redis as redis_module
from app.core.storage import get_storage


router = APIRouter()


@router.get("/metrics")
async def metrics():
    job_counts = await redis_module.job_redis.hgetall("metrics:jobs")
    queue_depth = {
        queue: await redis_module.broker_redis.llen(queue)
        for queue in ("cpu", "gpu-fast", "gpu-large")
    }
    return {
        "jobs": {key: int(value) for key, value in job_counts.items()},
        "queue_depth": queue_depth,
    }


@router.get("/health")
async def health():
    checks = {"session_redis": False, "job_redis": False, "broker_redis": False, "storage": False}
    details = []
    try:
        checks["session_redis"] = bool(await redis_module.redis.ping())
    except Exception as exc:
        details.append(f"session redis: {exc}")
    try:
        checks["job_redis"] = bool(await redis_module.job_redis.ping())
    except Exception as exc:
        details.append(f"job redis: {exc}")
    try:
        checks["broker_redis"] = bool(await redis_module.broker_redis.ping())
    except Exception as exc:
        details.append(f"broker redis: {exc}")
    try:
        storage = get_storage()
        await asyncio.to_thread(storage.exists, "healthcheck")
        checks["storage"] = True
    except Exception as exc:
        details.append(f"storage: {exc}")
    try:
        worker_keys = await redis_module.job_redis.keys("worker-heartbeat:*")
    except Exception:
        worker_keys = []
    payload = {
        "status": "ok" if all(checks.values()) else "degraded",
        **checks,
        "workers": len(worker_keys),
        "queue_depth": {
            queue: await redis_module.broker_redis.llen(queue) if checks["broker_redis"] else None
            for queue in ("cpu", "gpu-fast", "gpu-large")
        },
    }
    if details:
        payload["detail"] = "; ".join(details)
    return JSONResponse(payload, status_code=200 if all(checks.values()) else 503)
