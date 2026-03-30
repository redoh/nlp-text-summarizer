class SummarizationError(Exception):
    """Base exception for summarization errors."""


class ModelLoadError(SummarizationError):
    """Failed to load the ML model."""


class ModelInferenceError(SummarizationError):
    """Model inference failed during summarization."""


class InputValidationError(SummarizationError):
    """Input text failed validation."""


class CircuitOpenError(SummarizationError):
    """Circuit breaker is open, service temporarily unavailable."""
