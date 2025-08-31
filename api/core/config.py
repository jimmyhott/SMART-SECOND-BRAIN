"""Configuration settings for CryptoAnalyst."""

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "CryptoAnalyst"
    version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # Database
    database_url: str = Field(env="DATABASE_URL")
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # External APIs
    coinapi_key: str = Field(default="", env="COINAPI_KEY")
    binance_api_key: str = Field(default="", env="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", env="BINANCE_SECRET_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()
