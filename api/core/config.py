"""Configuration settings for Smart Second Brain Graph API."""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for Graph API."""

    # Application
    app_name: str = "Smart Second Brain"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")

    # OpenAI/Azure OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    azure_openai_endpoint_url: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT_URL")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow"  # Allow extra environment variables
    )


# Create settings instance
settings = Settings()
