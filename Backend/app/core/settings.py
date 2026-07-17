from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # use "none" on cross-site + https
    COOKIE_DOMAIN: str | None = None
    ML_DEVICE: str = "auto"

settings = Settings()
