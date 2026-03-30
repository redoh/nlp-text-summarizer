import logging
import time
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

OPEN_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/", "/metrics"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication. Disabled when api_key is empty."""

    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self.api_key:
            return await call_next(request)

        if request.url.path in OPEN_PATHS:
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, app, requests_per_minute: int = 30):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._window = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in OPEN_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean old entries
        self._requests[client_ip] = [t for t in self._requests[client_ip] if now - t < self._window]

        if len(self._requests[client_ip]) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
