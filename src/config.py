"""Configuration module for the Food Recommendation System.

Loads environment variables from .env file and provides
configuration settings for all services.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Gemini API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    # Neo4j Database Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # Server Configuration
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_DEBUG: bool = os.getenv("APP_DEBUG", "0") == "1"
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    # File Upload Configuration
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))  # 16MB
    TEMP_UPLOAD_DIR: Path = Path(os.getenv("TEMP_UPLOAD_DIR", "/tmp/food-recsys/uploads"))
    ALLOWED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "webp", "gif"}

    # Batch Processing Configuration
    MAX_WORKERS: int = 5

    # Ingredient Canonicalization Configuration
    SIMILARITY_THRESHOLD_HIGH: float = float(os.getenv("SIMILARITY_THRESHOLD_HIGH", "0.98"))
    SIMILARITY_THRESHOLD_LOW: float = float(os.getenv("SIMILARITY_THRESHOLD_LOW", "0.94"))
    CANONICAL_TOP_K: int = int(os.getenv("CANONICAL_TOP_K", "3"))

    @classmethod
    def validate(cls) -> list[str]:
        """Validate that all required configuration is present.

        Returns:
            List of missing configuration keys.
        """
        missing = []
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not cls.NEO4J_PASSWORD:
            missing.append("NEO4J_PASSWORD")
        return missing

    @classmethod
    def ensure_temp_dir(cls) -> None:
        """Ensure the temporary upload directory exists."""
        cls.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Create singleton config instance
config = Config()
