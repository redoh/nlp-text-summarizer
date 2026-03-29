import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers.summarize import router as summarize_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.info("Application started")
    if settings.preload_abstractive:
        from app.routers.summarize import _abstractive

        logger.info("Preloading abstractive model: %s", settings.abstractive_model)
        _abstractive._load_pipeline()
        logger.info("Abstractive model loaded successfully")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title=settings.app_name,
    description="Automatic text summarization API supporting extractive and abstractive strategies",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(summarize_router)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
