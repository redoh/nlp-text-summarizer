import threading
import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open" and self._last_failure_time:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half_open"
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = "open"

    @property
    def is_available(self) -> bool:
        return self.state != "open"
