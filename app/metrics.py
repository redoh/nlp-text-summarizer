import logging

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

# Metrics
REQUESTS_TOTAL = Counter(
    "summarizer_requests_total",
    "Total summarization requests",
    ["strategy", "status"],
)
REQUEST_DURATION = Histogram(
    "summarizer_request_duration_seconds",
    "Time spent processing summarization requests",
    ["strategy"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
ACTIVE_REQUESTS = Gauge(
    "summarizer_active_requests",
    "Number of active summarization requests",
)
MODEL_LOADED = Gauge(
    "summarizer_model_loaded",
    "Whether the abstractive model is loaded",
)
CACHE_SIZE = Gauge(
    "summarizer_cache_size",
    "Current number of items in summary cache",
)


@router.get("/metrics")
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
