from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    REDIS_URL: str = "redis://localhost:6379/0"
    JOB_REDIS_URL: str = "redis://localhost:6379/1"
    CELERY_BROKER_URL: str = "redis://localhost:6379/2"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/3"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    JOB_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # use "none" on cross-site + https
    COOKIE_DOMAIN: str | None = None
    ML_DEVICE: str = "auto"
    # Run eager-attention extraction on the CPU regardless of ML_DEVICE.
    # `output_attentions=True` forces transformers onto its eager attention path,
    # whose manual bmm torch dispatches to a Triton kernel; that kernel segfaults
    # the worker during JIT compilation on some torch/Triton builds. Triton is
    # GPU-only, so running this one model on the CPU avoids it. Prediction and
    # embedding are unaffected -- they use fused SDPA and stay on the GPU.
    # Set to false to retest the GPU path after a torch/Triton upgrade.
    ATTENTION_FORCE_CPU: bool = True
    ENABLE_LEGACY_SYNC_INFERENCE: bool = False

    STORAGE_BACKEND: str = "local"
    STORAGE_LOCAL_ROOT: str = "shared-storage"
    S3_BUCKET: str | None = None
    S3_ENDPOINT_URL: str | None = None
    S3_REGION: str = "us-east-1"
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None

    MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
    MAX_AUDIO_DURATION_SECONDS: float = 10 * 60
    API_V1_PREFIX: str = "/api/v1"
    RESULT_SCHEMA_VERSION: str = "v1"
    CODE_VERSION: str = "development"
    TASK_SOFT_TIME_LIMIT_SECONDS: int = 55 * 60
    TASK_TIME_LIMIT_SECONDS: int = 60 * 60

    # FR-10: Accent and language fairness analysis (docs/FR10plan.md Part 1 S2.3)
    FR10_MIN_GROUP_SIZE: int = 8
    FR10_MIN_SPEAKERS_PER_GROUP: int = 2

settings = Settings()
