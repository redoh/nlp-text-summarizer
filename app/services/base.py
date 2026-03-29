from abc import ABC, abstractmethod


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        """Summarize the given text and return the summary."""
