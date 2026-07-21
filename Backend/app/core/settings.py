from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # use "none" on cross-site + https
    COOKIE_DOMAIN: str | None = None

    # ── Custom model limits ────────────────────────────────────────
    # Enforced by app.services.custom_model_service during registration.
    # A model that exceeds any of these is rejected rather than accepted
    # and left to OOM the worker mid-inference.
    CUSTOM_MODEL_MAX_PARAMS: int = 1_000_000_000        # 1B parameters
    CUSTOM_MODEL_MAX_STORAGE_BYTES: int = 5 * 1024**3   # 5 GiB on disk
    CUSTOM_MODEL_MAX_INFERENCE_SECONDS: float = 120.0   # per-file wall clock
    CUSTOM_MODEL_PROBE_TIMEOUT_SECONDS: float = 60.0    # smoke-test budget
    # Resident models are cached in-process; beyond this the least recently
    # used one is evicted so a session cannot pin unbounded memory.
    CUSTOM_MODEL_CACHE_SIZE: int = 2
    CUSTOM_MODEL_MAX_PER_SESSION: int = 10
    # Registration downloads weights from the Hub. When the container runs
    # with HF_HUB_OFFLINE=1 only already-cached repos can be registered;
    # set this false to surface that as a clear error instead of a stack trace.
    CUSTOM_MODEL_ALLOW_DOWNLOAD: bool = True

settings = Settings()
