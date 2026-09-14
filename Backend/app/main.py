import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from .api.routes import (
    analyses as analyses_routes,
    dataset_management as dataset_management_routes,
    datasets as datasets_routes,
    debug as debug_routes,
    health as health_routes,
    jobs as jobs_routes,
    models as models_routes,
    session as session_routes,
    upload as upload_routes,
)
from .core.session import SessionMiddleware, service_unavailable
from .core.settings import settings
from .core.storage import StorageUnavailable


logger = logging.getLogger(__name__)

app = FastAPI(title="ECHO API", version="2.0")

app.add_middleware(SessionMiddleware)


@app.exception_handler(RedisError)
async def redis_unavailable(request: Request, exc: RedisError):
    # A Redis failure inside a route gets the same readable 503 the session
    # middleware returns, rather than a bare 500 the browser cannot use.
    logger.warning("redis unavailable during %s %s: %s", request.method, request.url.path, exc)
    return service_unavailable()


@app.exception_handler(StorageUnavailable)
async def storage_unavailable(request: Request, exc: StorageUnavailable):
    # The object store's equivalent of the Redis handler above: an S3 bucket
    # the deployment cannot reach is a readable, retryable 503, not a 500.
    logger.warning("storage unavailable during %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        {"detail": "Storage is temporarily unavailable. Wait a moment and try again."},
        status_code=503,
        headers={"Retry-After": "5"},
    )


LEGACY_PREFIXES = ("/inferences", "/saliency", "/perturb", "/results")
legacy_sync_enabled = settings.ENABLE_LEGACY_SYNC_INFERENCE and not settings.is_production


@app.middleware("http")
async def legacy_api_gate(request: Request, call_next):
    if not legacy_sync_enabled and request.url.path.startswith(LEGACY_PREFIXES):
        return JSONResponse(
            status_code=410,
            content={"detail": "Synchronous inference APIs are disabled; use POST /jobs"},
        )
    if legacy_sync_enabled and request.method in {"POST", "PUT", "PATCH"}:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        raw_path = payload.get("file_path") if isinstance(payload, dict) else None
        if raw_path:
            candidate = Path(raw_path).resolve()
            allowed_roots = [Path("uploads").resolve(), Path("data").resolve()]
            if not any(candidate == root or root in candidate.parents for root in allowed_roots):
                return JSONResponse(status_code=400, content={"detail": "Legacy file path is outside approved roots"})
    try:
        response = await call_next(request)
    except RuntimeError as exc:
        # BaseHTTPMiddleware raises this when the client goes away mid-request:
        # the inner app never sends a response, so there is nothing to return.
        # The browser cancels requests routinely (switching dataset aborts the
        # in-flight metadata/materialize calls), and each one logged a full
        # traceback. Only swallow it when the client really did disconnect --
        # anything else is a genuine bug and must still surface.
        if str(exc) == "No response returned." and await request.is_disconnected():
            # 499 (client closed request); nothing is left to receive it.
            return Response(status_code=499)
        raise
    if legacy_sync_enabled and request.url.path.startswith(LEGACY_PREFIXES):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "one release after the /jobs migration"
    return response


# Added last, so outermost: every response -- including the session
# middleware's 503 and the gate's 410 -- carries the CORS headers a
# cross-origin browser needs to read it, and a preflight is answered here
# without ever reaching the session store.  The allow-list is validated by
# Settings (SE-2): explicit origins only, never "*" alongside credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(session_routes.router, tags=["Session"])
app.include_router(upload_routes.router, tags=["Audio"])
app.include_router(jobs_routes.router, tags=["Jobs"])
app.include_router(models_routes.router, tags=["Custom Models"])
app.include_router(analyses_routes.router, prefix=settings.API_V1_PREFIX, tags=["Analyses"])
app.include_router(
    analyses_routes.router, tags=["Analyses (unversioned alias)"], include_in_schema=False
)
app.include_router(dataset_management_routes.router, prefix="/upload", tags=["Dataset Management"])
app.include_router(datasets_routes.router, tags=["Datasets"])
app.include_router(health_routes.router, tags=["Health"])
app.include_router(debug_routes.router, tags=["Debug"])

if legacy_sync_enabled:
    from .api.routes import (
        inferences as inferences_routes,
        perturbations as perturbations_routes,
        results as results_routes,
        saliency as saliency_routes,
    )

    app.include_router(results_routes.router, tags=["Legacy Results"])
    app.include_router(inferences_routes.router, tags=["Legacy Inferences"])
    app.include_router(saliency_routes.router, tags=["Legacy Saliency"])
    app.include_router(perturbations_routes.router, tags=["Legacy Perturbations"])
