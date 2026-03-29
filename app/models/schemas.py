from enum import Enum

from pydantic import BaseModel, Field


class Strategy(str, Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="The text to summarize")
    strategy: Strategy = Field(
        default=Strategy.EXTRACTIVE,
        description="Summarization strategy to use",
    )
    max_length: int | None = Field(
        default=None,
        ge=20,
        le=1000,
        description="Maximum length of the summary (abstractive only)",
    )
    num_sentences: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of sentences to extract (extractive only)",
    )


class SummarizeResponse(BaseModel):
    summary: str
    strategy: Strategy
    original_length: int
    summary_length: int
    compression_ratio: float


class HealthResponse(BaseModel):
    status: str
    version: str
