import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "Office Attendance & Identity System"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/identity_verification"
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/identity_verification"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    SECRET_KEY: str = ""  # MUST be set via .env
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # File storage
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE_MB: int = 10

    # ML Model paths
    FAISS_INDEX_DIR: str = "data/faiss_indices"
    MODEL_CACHE_DIR: str = "data/models"

    # Biometric thresholds
    FACE_MATCH_THRESHOLD: float = 0.6
    VOICE_MATCH_THRESHOLD: float = 0.25
    FINGERPRINT_MATCH_THRESHOLD: float = 0.6
    LIVENESS_THRESHOLD: float = 0.5
    DEEPFAKE_THRESHOLD: float = 0.5

    # Fusion weights
    FACE_WEIGHT: float = 0.45
    VOICE_WEIGHT: float = 0.30
    FINGERPRINT_WEIGHT: float = 0.25

    # Fraud detection
    MAX_FAILED_ATTEMPTS: int = 5
    FAILED_ATTEMPTS_WINDOW_MINUTES: int = 10

    # Tokenization
    BIOMETRIC_TOKEN_SECRET: str = ""  # MUST be set via .env

    # Office Attendance
    OFFICE_GRACE_MINUTES: int = 15
    OVERTIME_THRESHOLD_MINUTES: int = 30
    MAX_LATE_MINUTES_BEFORE_ALERT: int = 15
    ABSENT_CHECK_HOUR: int = 11
    ALLOWED_CLOCK_IN_IPS: list[str] = []

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
