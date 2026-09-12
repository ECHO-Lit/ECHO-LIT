import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from .settings import settings
from .redis import ensure_session

logger = logging.getLogger(__name__)

# The monitoring endpoints report on the dependencies a session needs, and are
# read most during an outage -- so they must not need a session themselves.
SESSIONLESS_PATHS = {"/health", "/metrics"}


def service_unavailable() -> JSONResponse:
    """The response for a request that could not reach Redis.

    A readable body the client can act on, and a hint of when to try again.
    """
    return JSONResponse(
        {"detail": "The service is temporarily unavailable. Wait a moment and try again."},
        status_code=503,
        headers={"Retry-After": "5"},
    )


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in SESSIONLESS_PATHS:
            return await call_next(request)
        cookie = request.cookies.get(settings.SESSION_COOKIE_NAME)
        try:
            sid = await ensure_session(cookie)
        except RedisError:
            # This middleware sits outside the app's exception handlers, so an
            # error here would otherwise become a bare 500.
            logger.warning("session store unavailable for %s %s", request.method, request.url.path)
            return service_unavailable()
        request.state.sid = sid
        resp: Response = await call_next(request)
        if sid != cookie:
            resp.set_cookie(
                settings.SESSION_COOKIE_NAME, sid,
                max_age=settings.SESSION_TTL_SECONDS,
                httponly=True, secure=settings.COOKIE_SECURE,
                samesite=settings.COOKIE_SAMESITE, domain=settings.COOKIE_DOMAIN, path="/",
            )
        return resp
