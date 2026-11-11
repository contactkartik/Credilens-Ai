"""
Core configuration settings for the backend application.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Backend Configuration
    BACKEND_HOST: str = "localhost"
    BACKEND_PORT: int = 8000
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./credilens.db"

    # ML Model Configuration
    MODEL_PATH: str = "models/"
    FEATURE_THRESHOLD: float = 0.5

    # Fine-tuned RoBERTa model (Stage 5) - relative paths are resolved
    # against the repository root by the credibility service.
    ROBERTA_MODEL_PATH: str = "models/roberta"

    class Config:
        env_file = ".env"


settings = Settings()
