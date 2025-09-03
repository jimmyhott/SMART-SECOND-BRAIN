"""
Main FastAPI application entry point for Smart Second Brain.

This module serves as the primary entry point for the FastAPI application,
configuring the server, middleware, routes, and providing basic health endpoints.
It integrates the LangGraph API router and sets up CORS for frontend communication.

Key Components:
- FastAPI application configuration
- CORS middleware setup
- API router integration
- Health check endpoints
- Development server configuration

Author: Smart Second Brain Team
Version: 0.1.0
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings

# =============================================================================
# FASTAPI APPLICATION CONFIGURATION
# =============================================================================

# Create the main FastAPI application instance with metadata and configuration
app = FastAPI(
    title="Smart Second Brain API",
    description="An intelligent knowledge management system powered by LangGraph and AI",
    version="0.1.0",
    # Only show API docs in debug mode for security
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# =============================================================================
# MIDDLEWARE CONFIGURATION
# =============================================================================

# Add CORS (Cross-Origin Resource Sharing) middleware to allow frontend communication
# This enables the NiceGUI frontend to make API calls from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,  # Configured origins from settings
    allow_credentials=True,                # Allow cookies and authentication headers
    allow_methods=["*"],                   # Allow all HTTP methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],                   # Allow all request headers
)

# =============================================================================
# BASIC API ENDPOINTS
# =============================================================================

@app.get("/test")
async def test_endpoint():
    """
    Simple test endpoint to verify API functionality.
    
    Returns:
        dict: Confirmation message that the API is working
        
    Usage:
        GET /test -> {"message": "API is working"}
    """
    return {"message": "API is working"}

# =============================================================================
# API ROUTER INTEGRATION
# =============================================================================

# Dynamically import and include the LangGraph API router
# This router provides the core functionality: /ingest, /query, and /health endpoints
try:
    from api.routes.v1.graph_api import router as graph_router
    # Include the router with a prefix for API versioning and organization
    app.include_router(graph_router, prefix="/smart-second-brain", tags=["graph"])
except ImportError as e:
    # Gracefully handle import errors during development or deployment
    print(f"Warning: Could not import graph router: {e}")

# =============================================================================
# ROOT AND HEALTH ENDPOINTS
# =============================================================================

@app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint providing basic API information.
    
    Returns:
        dict: Welcome message for the Smart Second Brain API
        
    Usage:
        GET / -> {"message": "Welcome to Smart Second Brain API"}
    """
    return {"message": "Welcome to Smart Second Brain API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Basic health check endpoint for monitoring and load balancers.
    
    This endpoint provides a simple health status that can be used by:
    - Load balancers to check if the service is alive
    - Monitoring systems to track API availability
    - Health check scripts and automated testing
    
    Returns:
        dict: Basic health status indicating the API is running
        
    Usage:
        GET /health -> {"status": "healthy"}
    """
    return {"status": "healthy"}

# =============================================================================
# DEVELOPMENT SERVER CONFIGURATION
# =============================================================================

# Enable running the application directly with: python api/main.py
# This is useful for development and testing without external uvicorn commands
if __name__ == "__main__":
    try:
        import uvicorn
        # Start the development server with configuration from settings
        uvicorn.run(
            "api.main:app",                    # Application import string
            host=settings.host,                # Host binding (default: 0.0.0.0)
            port=settings.port,                # Port number (default: 8000)
            reload=settings.debug,             # Auto-reload on code changes in debug mode
            log_level=settings.log_level.lower(),  # Logging level from settings
        )
    except ImportError:
        # Provide helpful error message if uvicorn is not available
        print("uvicorn not available, run with: uvicorn api.main:app --reload")
