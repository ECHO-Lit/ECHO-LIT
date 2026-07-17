from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from .settings import settings
from .redis import ensure_session

class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        sid = await ensure_session(request.cookies.get(settings.SESSION_COOKIE_NAME))
        request.state.sid = sid
        resp: Response = await call_next(request)
        if settings.SESSION_COOKIE_NAME not in request.cookies:
            resp.set_cookie(
                settings.SESSION_COOKIE_NAME, sid,
                max_age=settings.SESSION_TTL_SECONDS,
                httponly=True, secure=settings.COOKIE_SECURE,
                samesite=settings.COOKIE_SAMESITE, domain=settings.COOKIE_DOMAIN, path="/",
            )
        return resp
