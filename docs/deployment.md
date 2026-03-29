# Deployment Guide

## Local Development

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Visit `/docs` for the interactive Swagger UI.

## Docker

### Build and Run

```bash
docker build -t nlp-summarizer .
docker run -p 8000:8000 nlp-summarizer
```

### With Abstractive Model Preloading

To avoid cold-start latency on the first abstractive request, preload the model at startup:

```bash
docker run -p 8000:8000 \
  -e SUMMARIZER_PRELOAD_ABSTRACTIVE=true \
  nlp-summarizer
```

> **Note:** The BART model requires ~1.6 GB of disk and ~2 GB of RAM. Ensure your container has sufficient resources.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SUMMARIZER_DEFAULT_STRATEGY` | `extractive` | Default summarization strategy |
| `SUMMARIZER_MAX_INPUT_LENGTH` | `50000` | Maximum input text length (characters) |
| `SUMMARIZER_ABSTRACTIVE_MODEL` | `facebook/bart-large-cnn` | HuggingFace model ID |
| `SUMMARIZER_ABSTRACTIVE_MAX_LENGTH` | `150` | Max tokens for abstractive summary |
| `SUMMARIZER_ABSTRACTIVE_MIN_LENGTH` | `40` | Min tokens for abstractive summary |
| `SUMMARIZER_EXTRACTIVE_SENTENCE_COUNT` | `5` | Default extracted sentences |
| `SUMMARIZER_PRELOAD_ABSTRACTIVE` | `false` | Preload abstractive model on startup |
| `SUMMARIZER_REQUEST_TIMEOUT` | `120` | Keep-alive timeout in seconds |
| `SUMMARIZER_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `SUMMARIZER_DEBUG` | `false` | Enable debug mode |

## Production Considerations

### Resource Requirements

| Strategy | RAM | CPU | Latency |
|----------|-----|-----|---------|
| Extractive | ~50 MB | Low | <100ms |
| Abstractive | ~2 GB | Medium-High | 2-10s |

### Scaling

- **Extractive-only deployment**: Lightweight, can run many workers. Set `--workers 4` in uvicorn.
- **Abstractive deployment**: Memory-heavy per worker. Use `--workers 1` and scale horizontally with container replicas.

### Model Caching

The abstractive model is downloaded from HuggingFace on first use. To cache it across container restarts:

```bash
docker run -p 8000:8000 \
  -v huggingface-cache:/home/appuser/.cache/huggingface \
  -e SUMMARIZER_PRELOAD_ABSTRACTIVE=true \
  nlp-summarizer
```

## Troubleshooting

### Common Issues

**Model download fails or times out**
- Ensure the container has internet access
- Set `HF_HOME` to a writable directory with sufficient disk space
- Pre-download the model: `python -c "from transformers import pipeline; pipeline('summarization', model='facebook/bart-large-cnn')"`

**Out of memory (OOM) errors**
- The BART model requires ~2 GB RAM. Increase container memory limit.
- Use extractive strategy for low-memory environments.

**Slow first request**
- The abstractive model is loaded lazily on first use. Set `SUMMARIZER_PRELOAD_ABSTRACTIVE=true` to load at startup.

**Health check returns unhealthy**
- The health endpoint runs a real extractive summarization. If failing, check logs for errors.
- Verify the application started correctly: `docker logs <container_id>`

**Tests fail with import errors**
- Ensure you installed dev dependencies: `pip install -r requirements-dev.txt`
- Run from the project root: `pytest tests/ -v`

### Performance Tuning

- For high-throughput extractive: increase workers (`--workers 4`)
- For abstractive: keep 1 worker, use GPU if available (mount NVIDIA runtime)
- Monitor with structured logs: set `SUMMARIZER_LOG_LEVEL=DEBUG` for detailed output
