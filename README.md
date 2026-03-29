# NLP Text Summarizer

Automatic text summarization API supporting **extractive** and **abstractive** strategies. Built with FastAPI and HuggingFace Transformers.

## Features

- **Extractive Summarization** — TF-IDF based sentence scoring, no ML model needed, fast and lightweight
- **Abstractive Summarization** — Uses `facebook/bart-large-cnn` transformer to generate new summary text
- **REST API** — Clean JSON API with automatic Swagger docs
- **Configurable** — Environment variables for all settings
- **Docker-ready** — Dockerfile included for containerized deployment

## Quick Start

```bash
# Clone the repository
git clone https://github.com/redoh/nlp-text-summarizer.git
cd nlp-text-summarizer

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload

# Open API docs at http://localhost:8000/docs
```

## API Usage

### Summarize Text

```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "strategy": "extractive",
    "num_sentences": 3
  }'
```

**Response:**
```json
{
  "summary": "The extracted summary...",
  "strategy": "extractive",
  "original_length": 1500,
  "summary_length": 320,
  "compression_ratio": 0.2133
}
```

### Strategies

| Strategy | Description | Speed | Quality |
|----------|-------------|-------|---------|
| `extractive` (default) | Selects important sentences using TF-IDF | Fast | Good |
| `abstractive` | Generates new text via BART transformer | Slower | Higher |

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## Configuration

All settings can be overridden via environment variables with the `SUMMARIZER_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARIZER_DEFAULT_STRATEGY` | `extractive` | Default summarization strategy |
| `SUMMARIZER_MAX_INPUT_LENGTH` | `50000` | Maximum input text length (chars) |
| `SUMMARIZER_ABSTRACTIVE_MODEL` | `facebook/bart-large-cnn` | HuggingFace model for abstractive |
| `SUMMARIZER_EXTRACTIVE_SENTENCE_COUNT` | `5` | Default sentences to extract |

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Lint
ruff check app/ tests/

# Format
ruff format app/ tests/
```

## Docker

```bash
docker build -t nlp-summarizer .
docker run -p 8000:8000 nlp-summarizer
```

## Tech Stack

- **Python 3.10+**
- **FastAPI** — async web framework
- **HuggingFace Transformers** — abstractive summarization
- **Pydantic v2** — data validation
- **pytest + httpx** — testing

## Author

[@redoh](https://github.com/redoh)
