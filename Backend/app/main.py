import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import (
    dataset_management as dataset_management_routes,
    datasets as datasets_routes,
    debug as debug_routes,
    health as health_routes,
    jobs as jobs_routes,
    session as session_routes,
    upload as upload_routes,
)
from .core.session import SessionMiddleware
from .core.settings import settings


app = FastAPI(title="ECHO API", version="2.0")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = (
    [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    if allowed_origins_env
    else ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8080"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware)


LEGACY_PREFIXES = ("/inferences", "/saliency", "/perturb", "/results")
legacy_sync_enabled = (
    settings.ENABLE_LEGACY_SYNC_INFERENCE
    and settings.ENVIRONMENT.strip().lower() not in {"production", "prod"}
)


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
    response = await call_next(request)
    if legacy_sync_enabled and request.url.path.startswith(LEGACY_PREFIXES):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "one release after the /jobs migration"
    return response


app.include_router(session_routes.router, tags=["Session"])
app.include_router(upload_routes.router, tags=["Audio"])
app.include_router(jobs_routes.router, tags=["Jobs"])
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
