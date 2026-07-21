import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.session import SessionMiddleware

from .api.routes import session as session_routes, results as results_routes, inferences as inferences_routes, upload as upload_routes, health as health_routes
from .api.routes import datasets as datasets_routes, saliency as saliency_routes, perturbations as perturbations_routes, dataset_management as dataset_management_routes, debug as debug_routes
from .api.routes import custom_models as custom_models_routes

app = FastAPI(title="LIT for Voice – API")

# Configure CORS origins - default to common development origins if not set
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    # Default development origins
    origins = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:8080"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware)

app.include_router(session_routes.router, tags=["Session"])
app.include_router(results_routes.router, tags=["Results"])
app.include_router(inferences_routes.router, tags=["Inferences"])
app.include_router(upload_routes.router, tags=["Upload"])
app.include_router(dataset_management_routes.router, prefix="/upload", tags=["Dataset Management"])
app.include_router(datasets_routes.router, tags=["Datasets"])
app.include_router(saliency_routes.router, tags=["Saliency"])
app.include_router(perturbations_routes.router, tags=["Perturbations"])
app.include_router(custom_models_routes.router, tags=["Custom Models"])
app.include_router(health_routes.router, tags=["Health"])
app.include_router(debug_routes.router, tags=["Debug"])