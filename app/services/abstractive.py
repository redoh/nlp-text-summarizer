from app.config import settings
from app.services.base import BaseSummarizer


class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarizer using HuggingFace Transformers.

    Uses a pre-trained BART model to generate new summary text
    that may contain words not present in the original.
    """

    def __init__(self):
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "summarization",
                model=settings.abstractive_model,
            )
        return self._pipeline

    def summarize(self, text: str, **kwargs) -> str:
        pipe = self._load_pipeline()

        max_length = kwargs.get("max_length") or settings.abstractive_max_length
        min_length = kwargs.get("min_length") or settings.abstractive_min_length

        if min_length >= max_length:
            min_length = max(10, max_length - 20)

        result = pipe(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

        return result[0]["summary_text"]
