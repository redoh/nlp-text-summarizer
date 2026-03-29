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

    model_config = {"env_prefix": "SUMMARIZER_"}


settings = Settings()
