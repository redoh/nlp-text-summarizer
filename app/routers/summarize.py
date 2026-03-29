import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import (
    HealthResponse,
    Strategy,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.abstractive import AbstractiveSummarizer
from app.services.extractive import ExtractiveSummarizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["summarization"])

_extractive = ExtractiveSummarizer()
_abstractive = AbstractiveSummarizer()


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    logger.info(
        "Summarization request: strategy=%s, input_length=%d",
        request.strategy.value,
        len(request.text),
    )
    if len(request.text) > settings.max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input text exceeds maximum length of {settings.max_input_length} characters",
        )

    kwargs = {}
    if request.strategy == Strategy.EXTRACTIVE:
        summarizer = _extractive
        if request.num_sentences is not None:
            kwargs["num_sentences"] = request.num_sentences
    else:
        summarizer = _abstractive
        if request.max_length is not None:
            kwargs["max_length"] = request.max_length

    try:
        summary = summarizer.summarize(request.text, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}") from e

    original_length = len(request.text)
    summary_length = len(summary)

    return SummarizeResponse(
        summary=summary,
        strategy=request.strategy,
        original_length=original_length,
        summary_length=summary_length,
        compression_ratio=round(summary_length / original_length, 4) if original_length else 0,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        test_sentence = (
            "The quick brown fox jumps over the lazy dog. This is a test sentence for health check."
        )
        _extractive.summarize(test_sentence, num_sentences=1)
        status = "healthy"
    except Exception:
        logger.exception("Health check failed")
        status = "unhealthy"
    return HealthResponse(status=status, version="0.1.0")
