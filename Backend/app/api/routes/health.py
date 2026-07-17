from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from ...core.redis import redis
from ...core.device import INFERENCE_RUNTIME

router = APIRouter()

@router.get("/health")
async def health():
    try:
        pong = await redis.ping()
        return {"status": "ok", "redis": bool(pong), "inference": INFERENCE_RUNTIME.as_dict()}
    except RedisError as e:
        # Return 503 if Redis isn’t reachable
        return JSONResponse(
            {
                "status": "degraded",
                "redis": False,
                "inference": INFERENCE_RUNTIME.as_dict(),
                "detail": str(e),
            },
            status_code=503,
        )
