import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# The ML_DEVICE grammar, checked without importing torch: the API validates its
# configuration too, and the control plane carries no ML dependency (DC-2).
DEVICE_PREFERENCE = re.compile(r"auto|cpu|mps|cuda(?::\d+)?|nvidia|rocm|amd|gpu|accelerator")


def parse_origins(raw: str) -> list[str]:
    """The CORS allow-list: comma-separated `scheme://host[:port]` origins.

    A browser's Origin header is exactly that -- never a path, never a trailing
    slash -- so anything else could never match and is refused rather than
    silently blocking the site.  `*` is refused because the API allows
    credentials, and with credentials Starlette answers `*` by echoing back
    whatever origin asked: every site on the web could read a session.
    """
    origins: list[str] = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise ValueError(
                "ALLOWED_ORIGINS must list origins explicitly; '*' with credentialed "
                "CORS would admit every site"
            )
        parts = urlsplit(origin)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.path or parts.query or parts.fragment:
            raise ValueError(f"ALLOWED_ORIGINS entry {item.strip()!r} is not an origin such as http://host:port")
        if origin not in origins:
            origins.append(origin)
    if not origins:
        raise ValueError("ALLOWED_ORIGINS lists no origin; no browser could use the API")
    return origins


class Settings(BaseSettings):
    # Settings come from the process environment only.  Compose injects
    # `Backend/.env` into the containers; the code never reads a dotenv file
    # itself, so no test run takes its configuration from a developer's `.env`.
    # An empty value (`KEY=`) means "unset".
    model_config = SettingsConfigDict(env_ignore_empty=True)

    ENVIRONMENT: str = "development"
    REDIS_URL: str = "redis://localhost:6379/0"
    JOB_REDIS_URL: str = "redis://localhost:6379/1"
    CELERY_BROKER_URL: str = "redis://localhost:6379/2"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/3"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: PositiveInt = 24 * 60 * 60
    JOB_TTL_SECONDS: PositiveInt = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"  # "none" needs COOKIE_SECURE
    COOKIE_DOMAIN: str | None = None
    # Comma-separated browser origins allowed to call the API with credentials.
    ALLOWED_ORIGINS: str = "http://localhost:8080,http://127.0.0.1:8080"
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

    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    STORAGE_LOCAL_ROOT: str = "shared-storage"
    S3_BUCKET: str | None = None
    S3_ENDPOINT_URL: str | None = None
    S3_REGION: str = "us-east-1"
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None

    MAX_UPLOAD_BYTES: PositiveInt = 100 * 1024 * 1024
    MAX_AUDIO_DURATION_SECONDS: PositiveFloat = 10 * 60
    API_V1_PREFIX: str = "/api/v1"
    RESULT_SCHEMA_VERSION: str = "v1"
    CODE_VERSION: str = "development"
    TASK_SOFT_TIME_LIMIT_SECONDS: PositiveInt = 55 * 60
    TASK_TIME_LIMIT_SECONDS: PositiveInt = 60 * 60
    # A task lost with its worker is redelivered once the broker's visibility
    # timeout passes; a job nothing has written to for longer than that is
    # failed by the stale-job reaper. Unset, both derive from the hard time
    # limit, so a task is never redelivered while it may still be running and
    # never reaped before its redelivery had a chance.
    BROKER_VISIBILITY_TIMEOUT_SECONDS: PositiveInt | None = None
    STALE_JOB_SECONDS: PositiveInt | None = None
    STALE_JOB_SWEEP_SECONDS: PositiveInt = 5 * 60

    # Worker resource bounds (PE-3): how many model variants a worker keeps
    # loaded, how long an idle one survives, and the saliency analysis window.
    MODEL_REGISTRY_MAX_ENTRIES: PositiveInt = 3
    MODEL_REGISTRY_IDLE_SECONDS: PositiveInt = 30 * 60
    MAX_SALIENCY_SECONDS: PositiveInt = 12
    MAX_SALIENCY_SECONDS_SHAP: PositiveInt = 6
    SALIENCY_SHAP_SAMPLES: int = Field(default=8, ge=2)

    # FR-10: Accent and language fairness analysis (docs/FR10plan.md Part 1 S2.3)
    FR10_MIN_GROUP_SIZE: PositiveInt = 8
    FR10_MIN_SPEAKERS_PER_GROUP: PositiveInt = 2

    @field_validator("ENVIRONMENT", "COOKIE_SAMESITE", "STORAGE_BACKEND", "ML_DEVICE", mode="before")
    @classmethod
    def _normalise(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("ML_DEVICE")
    @classmethod
    def _known_device(cls, value: str) -> str:
        # Checked here so a typo fails on every host.  The CUDA index used to be
        # parsed only when a GPU was present: `cuda:abc` fell back to the CPU on
        # a CPU machine and crashed the GPU worker at import.
        if not DEVICE_PREFERENCE.fullmatch(value):
            raise ValueError(
                "ML_DEVICE must be one of: auto, cpu, mps, cuda, cuda:<index>, nvidia, rocm, amd"
            )
        return value

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def _valid_origins(cls, value: str) -> str:
        return ",".join(parse_origins(value))

    @property
    def allowed_origins(self) -> list[str]:
        return parse_origins(self.ALLOWED_ORIGINS)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.strip().lower() in {"production", "prod"}

    @model_validator(mode="after")
    def _order_recovery_timings(self) -> "Settings":
        if self.BROKER_VISIBILITY_TIMEOUT_SECONDS is None:
            self.BROKER_VISIBILITY_TIMEOUT_SECONDS = self.TASK_TIME_LIMIT_SECONDS + 10 * 60
        if self.STALE_JOB_SECONDS is None:
            self.STALE_JOB_SECONDS = self.BROKER_VISIBILITY_TIMEOUT_SECONDS + 30 * 60
        timings = (
            self.TASK_SOFT_TIME_LIMIT_SECONDS,
            self.TASK_TIME_LIMIT_SECONDS,
            self.BROKER_VISIBILITY_TIMEOUT_SECONDS,
            self.STALE_JOB_SECONDS,
        )
        if not timings[0] < timings[1] < timings[2] < timings[3]:
            raise ValueError(
                "TASK_SOFT_TIME_LIMIT_SECONDS < TASK_TIME_LIMIT_SECONDS < "
                "BROKER_VISIBILITY_TIMEOUT_SECONDS < STALE_JOB_SECONDS must be ordered, "
                f"got {timings}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_transport_security(self) -> "Settings":
        # SRS SE-2: "Session cookies shall be HttpOnly, with Secure and SameSite
        # enabled in production, and HTTPS shall be enforced."
        if self.COOKIE_SAMESITE == "none" and not self.COOKIE_SECURE:
            raise ValueError(
                "COOKIE_SAMESITE=none requires COOKIE_SECURE=true: browsers reject a "
                "SameSite=None cookie that is not Secure"
            )
        if self.is_production:
            if not self.COOKIE_SECURE:
                raise ValueError("ENVIRONMENT=production requires COOKIE_SECURE=true (SRS SE-2)")
            insecure = [origin for origin in self.allowed_origins if not origin.startswith("https://")]
            if insecure:
                raise ValueError(f"ENVIRONMENT=production requires https ALLOWED_ORIGINS, got {insecure}")
        return self


settings = Settings()
