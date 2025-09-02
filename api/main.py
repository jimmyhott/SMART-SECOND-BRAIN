"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
# Create FastAPI application
app = FastAPI(
    title="Smart Second Brain API",
    description="An intelligent knowledge management system powered by LangGraph and AI",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple test endpoint
@app.get("/test")
async def test_endpoint():
    return {"message": "API is working"}

# Include API routers
try:
    from api.routes.v1.graph_api import router as graph_router
    app.include_router(graph_router, prefix="/smart-second-brain", tags=["graph"])
except ImportError as e:
    print(f"Warning: Could not import graph router: {e}")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to Smart Second Brain API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# For running with uvicorn directly
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
        )
    except ImportError:
        print("uvicorn not available, run with: uvicorn api.main:app --reload")
