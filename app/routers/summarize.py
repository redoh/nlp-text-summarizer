import logging

from fastapi import APIRouter, HTTPException

from app.cache import SummaryCache
from app.config import settings
from app.exceptions import (
    CircuitOpenError,
    InputValidationError,
    ModelInferenceError,
    SummarizationError,
)
from app.models.schemas import (
    BatchSummarizeRequest,
    BatchSummarizeResponse,
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
_cache = SummaryCache()


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

    # Check cache first
    cached = _cache.get(request.text, request.strategy.value, **kwargs)
    if cached is not None:
        logger.info("Returning cached summary for strategy=%s", request.strategy.value)
        summary = cached
    else:
        try:
            summary = summarizer.summarize(request.text, **kwargs)
        except CircuitOpenError as e:
            logger.exception("Circuit breaker open: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Try again later.",
            ) from e
        except InputValidationError as e:
            logger.exception("Input validation error: %s", e)
            raise HTTPException(
                status_code=400,
                detail=str(e),
            ) from e
        except ModelInferenceError as e:
            logger.exception("Model inference error: %s", e)
            raise HTTPException(
                status_code=502,
                detail="Model inference failed. Try again later.",
            ) from e
        except SummarizationError as e:
            logger.exception("Summarization error: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Summarization failed.",
            ) from e
        except Exception:
            logger.exception("Unexpected error during summarization")
            raise HTTPException(
                status_code=500,
                detail="Internal server error",
            )

        _cache.put(request.text, request.strategy.value, summary, **kwargs)

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


@router.post("/summarize/batch", response_model=BatchSummarizeResponse)
async def batch_summarize(request: BatchSummarizeRequest) -> BatchSummarizeResponse:
    results = []
    for item in request.items:
        response = await summarize(item)
        results.append(response)
    return BatchSummarizeResponse(results=results, total=len(results))


@router.get("/cache/stats")
async def cache_stats() -> dict:
    return _cache.stats
