import hashlib
import logging
import time
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


class SummaryCache:
    """Thread-safe LRU cache for summarization results."""

    def __init__(self, max_size: int = 128, ttl: float = 3600.0):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, strategy: str, **kwargs) -> str:
        content = f"{strategy}:{text}:{sorted(kwargs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, strategy: str, **kwargs) -> str | None:
        key = self._make_key(text, strategy, **kwargs)
        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if time.monotonic() - timestamp < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    logger.debug("Cache hit for key %s", key[:12])
                    return result
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, text: str, strategy: str, result: str, **kwargs) -> None:
        key = self._make_key(text, strategy, **kwargs)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (result, time.monotonic())
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
            }
