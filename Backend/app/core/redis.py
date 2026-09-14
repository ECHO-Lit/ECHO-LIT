import json, logging, re, uuid
from typing import Any
from redis.asyncio import from_url
from redis.exceptions import ResponseError
from .settings import settings

logger = logging.getLogger(__name__)

# A minted session id is `uuid.uuid4().hex` -- 32 lowercase hex characters.
# The id is not merely a Redis key: `SessionMiddleware` puts it on every request
# and it is then used verbatim as a *filesystem path segment*
# (`uploads/sessions/<sid>/…`, `datasets/<sid>/…`, `results/<sid>/…`) and as an
# object-storage key prefix. Because it arrives from a client-controlled cookie,
# a value containing path separators or `..` escapes all of those namespaces at
# once -- in the worst case `POST /upload/dataset/cleanup` resolves
# `SESSIONS_BASE_DIR / ".."` and `rmtree`s the parent of the sessions root,
# destroying every session's data. Anything that is not the exact shape we mint
# is therefore not a session we issued, and is replaced rather than trusted.
_VALID_SID = re.compile(r"\A[0-9a-f]{32}\Z")

# Initialize Redis connection with connection pool
redis = from_url(
    settings.REDIS_URL, 
    decode_responses=True,
    max_connections=20,
    retry_on_timeout=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

job_redis = from_url(
    settings.JOB_REDIS_URL,
    decode_responses=True,
    max_connections=20,
    retry_on_timeout=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

broker_redis = from_url(
    settings.CELERY_BROKER_URL,
    decode_responses=True,
    max_connections=10,
    socket_connect_timeout=5,
    socket_timeout=5,
)

def k_sess(sid: str) -> str:  return f"sess:{sid}"
def k_queue(sid: str) -> str: return f"{k_sess(sid)}:queue"
def k_meta(sid: str) -> str:  return f"{k_sess(sid)}:meta"
def k_result(model: str, h: str) -> str: return f"result:{model}:{h}"

async def _touch_session(sid: str) -> None:
    p = redis.pipeline()
    p.hsetnx(k_meta(sid), "created", "1")
    p.expire(k_queue(sid), settings.SESSION_TTL_SECONDS)
    p.expire(k_meta(sid), settings.SESSION_TTL_SECONDS)
    await p.execute()

async def ensure_session(sid: str | None) -> str:
    if not sid: sid = uuid.uuid4().hex
    try:
        await _touch_session(sid)
    except ResponseError as exc:
        # A meta key of the wrong type fails HSETNX with WRONGTYPE on every
        # request, and the same pipeline refreshes its TTL, so the session
        # could never expire its way out.  Start a new one instead.  Any other
        # refusal (an OOM under noeviction) is the store's problem, not this
        # session's, and propagates.
        if "WRONGTYPE" not in str(exc):
            raise
        logger.warning("Session %s has a key of the wrong type; issuing a new session", sid)
        sid = uuid.uuid4().hex
        await _touch_session(sid)
    return sid

def _empty_queue() -> dict[str, Any]:
    return {"items": [], "processing": None, "completed": []}

async def get_queue(sid: str) -> dict[str, Any]:
    raw = await redis.get(k_queue(sid))
    if not raw:
        return _empty_queue()
    try:
        state = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # A truncated or otherwise unreadable queue must not brick the session.
        # add_item and set_progress both read before writing, so raising here
        # would make /queue, /queue/add and /queue/progress fail for the whole
        # 24h TTL with no way back except direct Redis access.
        logger.warning("Discarding unreadable queue state for session %s", sid)
        return _empty_queue()
    if not isinstance(state, dict):
        logger.warning("Discarding non-object queue state for session %s", sid)
        return _empty_queue()
    return state

async def put_queue(sid: str, state: dict[str, Any]) -> None:
    await redis.set(k_queue(sid), json.dumps(state), ex=settings.SESSION_TTL_SECONDS)

async def cache_result(model: str, h: str, payload: dict, ttl: int = 6*60*60) -> None:
    await redis.set(k_result(model, h), json.dumps(payload), ex=ttl)

async def get_result(model: str, h: str) -> dict | None:
    raw = await redis.get(k_result(model, h))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # A corrupt cache entry is a miss, not an error: the caller recomputes.
        logger.warning("Discarding unreadable cached result for %s:%s", model, h)
        return None
