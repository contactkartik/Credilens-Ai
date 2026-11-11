"""
Main application entry point for the FastAPI backend.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.core.config import settings
from app.core.database import Base, get_engine
from app.models import prediction_log  # noqa: F401 - registers tables on Base
from app.services.credibility_service import get_credibility_service

logger = logging.getLogger("credilens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables, then load the trained RoBERTa model (warm-up)."""
    try:
        Base.metadata.create_all(get_engine())
        logger.info("Database tables ready (%s)", settings.DATABASE_URL.split('@')[-1])
    except Exception as exc:
        logger.warning("Could not initialize database: %s", exc)

    service = get_credibility_service()
    if service.load():
        logger.info("RoBERTa model loaded from %s", service.model_path)
    else:
        logger.warning(
            "RoBERTa model failed to load: %s - /api/predict will return 503",
            service.load_error,
        )
    yield


app = FastAPI(
    title="CrediLens AI API",
    description="Information Credibility Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CrediLens AI API",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )
