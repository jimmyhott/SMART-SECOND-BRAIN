"""
Configuration settings for Smart Second Brain Graph API.

This module defines all configuration settings for the Smart Second Brain application,
including server settings, API credentials, logging configuration, and environment
variable handling. It uses Pydantic Settings for type-safe configuration management
with automatic environment variable loading.

Key Features:
- Environment variable integration with .env file support
- Type-safe configuration with Pydantic validation
- Flexible configuration for development and production
- Support for both OpenAI and Azure OpenAI configurations

Environment Variables:
- DEBUG: Enable debug mode (default: False)
- HOST: Server host binding (default: 0.0.0.0)
- PORT: Server port number (default: 8000)
- ALLOWED_HOSTS: CORS allowed origins (default: ["*"])
- OPENAI_API_KEY: OpenAI API key for AI features
- AZURE_OPENAI_ENDPOINT_URL: Azure OpenAI endpoint URL
- LOG_LEVEL: Logging level (default: INFO)

Author: Smart Second Brain Team
Version: 0.1.0
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for Smart Second Brain Graph API.
    
    This class manages all configuration settings for the application,
    automatically loading values from environment variables and .env files.
    It provides type-safe configuration with sensible defaults for development.
    
    Attributes:
        app_name: Application name for identification
        version: Application version string
        debug: Debug mode flag for development features
        host: Server host binding address
        port: Server port number
        allowed_hosts: CORS allowed origins list
        openai_api_key: OpenAI API key for AI functionality
        azure_openai_endpoint_url: Azure OpenAI service endpoint
        log_level: Logging level configuration
    """

    # =============================================================================
    # APPLICATION IDENTITY
    # =============================================================================
    
    app_name: str = "Smart Second Brain"
    """Application name used in logs, headers, and documentation."""
    
    version: str = "0.1.0"
    """Application version for API versioning and tracking."""
    
    debug: bool = Field(default=False, env="DEBUG")
    """
    Debug mode flag that enables development features.
    
    When True, enables:
    - API documentation at /docs and /redoc
    - Auto-reload on code changes
    - Detailed error messages
    - Development logging
    
    Set via DEBUG environment variable or .env file.
    """

    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    
    host: str = Field(default="0.0.0.0", env="HOST")
    """
    Server host binding address.
    
    Default: 0.0.0.0 (bind to all available network interfaces)
    Can be set to specific IP for production deployments.
    
    Set via HOST environment variable or .env file.
    """
    
    port: int = Field(default=8000, env="PORT")
    """
    Server port number for HTTP connections.
    
    Default: 8000
    Ensure this port is available and not blocked by firewall.
    
    Set via PORT environment variable or .env file.
    """
    
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    """
    CORS allowed origins for cross-origin requests.
    
    Default: ["*"] (allow all origins - suitable for development)
    Production should restrict to specific domains:
    - ["https://yourdomain.com", "https://app.yourdomain.com"]
    - ["http://localhost:5173"] for local development
    
    Set via ALLOWED_HOSTS environment variable or .env file.
    """

    # =============================================================================
    # AI/LLM CONFIGURATION
    # =============================================================================
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    """
    OpenAI API key for AI functionality.
    
    Required for:
    - Text embeddings (text-embedding-3-small)
    - Language model interactions (GPT-4o)
    - Document processing and knowledge extraction
    
    Set via OPENAI_API_KEY environment variable or .env file.
    Note: This is also used for Azure OpenAI as the API key.
    """
    
    azure_openai_endpoint_url: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT_URL")
    """
    Azure OpenAI service endpoint URL.
    
    Required for Azure OpenAI integration:
    - Format: https://your-resource.openai.azure.com/
    - Used for both embeddings and language model calls
    - Must be configured for production Azure deployments
    
    Set via AZURE_OPENAI_ENDPOINT_URL environment variable or .env file.
    """

    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    """
    Logging level for application logging.
    
    Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    Default: INFO (balanced between detail and performance)
    
    Set via LOG_LEVEL environment variable or .env file.
    """

    # =============================================================================
    # PYDANTIC CONFIGURATION
    # =============================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file in project root
        case_sensitive=False,      # Environment variables are case-insensitive
        extra="allow"              # Allow extra environment variables beyond those defined
    )
    """
    Pydantic settings configuration for environment variable handling.
    
    env_file: Automatically load .env file from project root
    case_sensitive: Environment variables can be uppercase or lowercase
    extra: Allow additional environment variables beyond those defined
    """


# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================

# Create a global settings instance that can be imported throughout the application
# This instance automatically loads configuration from environment variables
settings = Settings()
