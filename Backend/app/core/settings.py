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
    RESULT_SCHEMA_VERSION: str = "v1"
    CODE_VERSION: str = "development"
    TASK_SOFT_TIME_LIMIT_SECONDS: int = 55 * 60
    TASK_TIME_LIMIT_SECONDS: int = 60 * 60

settings = Settings()
