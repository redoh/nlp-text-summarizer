import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

SAMPLE_TEXT = (
    "Natural language processing is a subfield of linguistics, computer science, "
    "and artificial intelligence concerned with the interactions between computers "
    "and human language. The goal is to enable computers to understand, interpret, "
    "and generate human language in a valuable way. NLP combines computational "
    "linguistics with statistical, machine learning, and deep learning models. "
    "These approaches enable the analysis of large amounts of natural language data. "
    "Applications of NLP include machine translation, sentiment analysis, chatbots, "
    "and text summarization. Text summarization is particularly important for "
    "condensing large documents into shorter versions while preserving key information. "
    "There are two main approaches to text summarization: extractive and abstractive. "
    "Extractive summarization selects important sentences from the original text. "
    "Abstractive summarization generates new sentences that capture the main ideas. "
    "Modern NLP systems often use transformer architectures like BERT and GPT. "
    "These models have achieved state-of-the-art results on many NLP benchmarks."
)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data


@pytest.mark.asyncio
async def test_summarize_extractive(client):
    response = await client.post(
        "/api/v1/summarize",
        json={"text": SAMPLE_TEXT, "strategy": "extractive", "num_sentences": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "extractive"
    assert data["summary_length"] < data["original_length"]
    assert 0 < data["compression_ratio"] < 1


@pytest.mark.asyncio
async def test_summarize_default_strategy(client):
    response = await client.post("/api/v1/summarize", json={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "extractive"


@pytest.mark.asyncio
async def test_summarize_too_short_text(client):
    response = await client.post("/api/v1/summarize", json={"text": "Too short"})
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_summarize_missing_text(client):
    response = await client.post("/api/v1/summarize", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_summarize_invalid_strategy(client):
    response = await client.post(
        "/api/v1/summarize",
        json={"text": SAMPLE_TEXT, "strategy": "nonexistent_strategy"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_summarize_with_max_length_param(client):
    response = await client.post(
        "/api/v1/summarize",
        json={"text": SAMPLE_TEXT, "strategy": "extractive", "num_sentences": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["summary_length"] < data["original_length"]


@pytest.mark.asyncio
async def test_summarize_text_too_long(client):
    long_text = "This is a sentence that is repeated many times. " * 5000
    response = await client.post(
        "/api/v1/summarize",
        json={"text": long_text, "strategy": "extractive"},
    )
    assert response.status_code == 400
    assert "exceeds maximum length" in response.json()["detail"]


@pytest.mark.asyncio
async def test_batch_summarize(client):
    response = await client.post(
        "/api/v1/summarize/batch",
        json={
            "items": [
                {"text": SAMPLE_TEXT, "strategy": "extractive", "num_sentences": 2},
                {"text": SAMPLE_TEXT, "strategy": "extractive", "num_sentences": 3},
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


@pytest.mark.asyncio
async def test_batch_summarize_empty(client):
    response = await client.post("/api/v1/summarize/batch", json={"items": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_cache_stats(client):
    response = await client.get("/api/v1/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert "misses" in data
    assert "size" in data


@pytest.mark.asyncio
async def test_request_id_header(client):
    response = await client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_custom_request_id(client):
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "test-123"})
    assert response.headers["X-Request-ID"] == "test-123"


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_summarize_caching(client):
    # First request
    await client.post(
        "/api/v1/summarize",
        json={"text": SAMPLE_TEXT, "strategy": "extractive", "num_sentences": 2},
    )
    # Second identical request should hit cache
    await client.post(
        "/api/v1/summarize",
        json={"text": SAMPLE_TEXT, "strategy": "extractive", "num_sentences": 2},
    )
    stats = await client.get("/api/v1/cache/stats")
    data = stats.json()
    assert data["hits"] >= 1
