from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "NLP Text Summarizer"
    debug: bool = False
    default_strategy: str = "extractive"
    max_input_length: int = 50000
    abstractive_model: str = "facebook/bart-large-cnn"
    abstractive_max_length: int = 150
    abstractive_min_length: int = 40
    extractive_sentence_count: int = 5
    preload_abstractive: bool = False
    request_timeout: int = 120
    log_level: str = "INFO"
    rate_limit_per_minute: int = 30
    cache_max_size: int = 128
    cache_ttl: int = 3600
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: int = 60
    api_key: str = ""

    model_config = {"env_prefix": "SUMMARIZER_"}


settings = Settings()
