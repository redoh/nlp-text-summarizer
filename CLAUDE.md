# NLP Text Summarizer

## Project Overview
A Python-based text summarization API built with FastAPI and HuggingFace Transformers.
Supports both extractive and abstractive summarization strategies.

## Tech Stack
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **ML**: HuggingFace Transformers (facebook/bart-large-cnn for abstractive)
- **Testing**: pytest + httpx
- **Linting**: ruff

## Project Structure
```
app/
├── main.py           # FastAPI application entry point
├── config.py         # Settings via pydantic-settings
├── models/
│   └── schemas.py    # Request/response Pydantic models
├── services/
│   ├── base.py       # Abstract base summarizer
│   ├── extractive.py # Frequency-based extractive summarizer
│   └── abstractive.py# Transformer-based abstractive summarizer
└── routers/
    └── summarize.py  # /summarize API endpoints
tests/
├── test_api.py       # API integration tests
└── test_services.py  # Unit tests for summarization services
```

## Commands
- **Run server**: `uvicorn app.main:app --reload`
- **Run tests**: `pytest tests/ -v`
- **Lint**: `ruff check app/ tests/`
- **Format**: `ruff format app/ tests/`
- **Install deps**: `pip install -r requirements.txt`
- **Install dev deps**: `pip install -r requirements-dev.txt`

## Code Conventions
- Use type hints everywhere
- Pydantic models for all request/response schemas
- Services follow the Strategy pattern with a common base class
- Keep routes thin — business logic lives in services
- Tests use pytest fixtures and httpx AsyncClient for API tests
- Environment config via pydantic-settings (no hardcoded values)

## API Endpoints
- `POST /api/v1/summarize` — Summarize text (accepts strategy: extractive | abstractive)
- `GET /api/v1/health` — Health check
- `GET /docs` — Swagger UI (auto-generated)
