from fastapi import FastAPI

from app.config import settings
from app.routers.summarize import router as summarize_router

app = FastAPI(
    title=settings.app_name,
    description="Automatic text summarization API supporting extractive and abstractive strategies",
    version="0.1.0",
)

app.include_router(summarize_router)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
