import re

from app.config import settings
from app.exceptions import CircuitOpenError, ModelInferenceError, ModelLoadError
from app.services.base import BaseSummarizer
from app.services.circuit_breaker import CircuitBreaker

_CONTROL_CHAR_PATTERN = re.compile(r"[^\x20-\x7E\s]")


class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarizer using HuggingFace Transformers.

    Uses a pre-trained BART model to generate new summary text
    that may contain words not present in the original.
    """

    def __init__(self):
        self._pipeline = None
        self._circuit_breaker = CircuitBreaker()

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ModelLoadError("Failed to import transformers library") from exc

            self._pipeline = pipeline(
                "summarization",
                model=settings.abstractive_model,
            )
        return self._pipeline

    @staticmethod
    def _sanitize(text: str) -> str:
        """Filter control characters, keeping only printable + whitespace."""
        return _CONTROL_CHAR_PATTERN.sub("", text)

    def summarize(self, text: str, **kwargs) -> str:
        if not self._circuit_breaker.is_available:
            raise CircuitOpenError("Abstractive summarization service is temporarily unavailable")

        text = self._sanitize(text)
        pipe = self._load_pipeline()

        max_length = kwargs.get("max_length") or settings.abstractive_max_length
        min_length = kwargs.get("min_length") or settings.abstractive_min_length

        if min_length >= max_length:
            min_length = max(10, max_length - 20)

        try:
            result = pipe(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
        except RuntimeError as exc:
            self._circuit_breaker.record_failure()
            raise ModelInferenceError(f"Model inference failed: {exc}") from exc

        try:
            summary = result[0]["summary_text"]
        except (IndexError, KeyError) as exc:
            self._circuit_breaker.record_failure()
            raise ModelInferenceError("Unexpected model output format") from exc

        self._circuit_breaker.record_success()
        return summary
