# NLP Text Summarizer — Implementation Plan

## Phase 1: Project Foundation
- [x] Initialize repository with README
- [x] Create CLAUDE.md with project conventions
- [x] Create this implementation plan
- [x] Set up Python project structure (pyproject.toml, requirements)
- [x] Create FastAPI application skeleton
- [x] Configure settings management

## Phase 2: Core Summarization Services
- [x] Define abstract base summarizer interface
- [x] Implement extractive summarizer (frequency-based, no ML dependencies)
- [x] Implement abstractive summarizer (HuggingFace BART)
- [x] Create Pydantic request/response schemas

## Phase 3: API Layer
- [x] Create summarization router with POST endpoint
- [x] Add health check endpoint
- [x] Wire up dependency injection for summarizer selection
- [x] Add input validation and error handling

## Phase 4: Testing
- [x] Write unit tests for extractive summarizer
- [x] Write unit tests for abstractive summarizer (abstractive tested via API layer)
- [x] Write API integration tests
- [x] Verify all tests pass (13/13 passing)

## Phase 5: Polish
- [x] Add Dockerfile for containerized deployment
- [x] Update README with full setup instructions
- [x] Final lint and format pass

---

## Architecture Decisions

### Summarization Strategies
1. **Extractive**: Selects the most important sentences from the original text
   using TF-IDF scoring. Zero ML dependencies — fast and lightweight.
2. **Abstractive**: Uses `facebook/bart-large-cnn` transformer model to generate
   new summary text. Higher quality but requires model download (~1.6GB).

### Why FastAPI?
- Async support for handling concurrent requests
- Automatic OpenAPI/Swagger documentation
- Pydantic integration for validation
- Production-ready with uvicorn

### API Design
Single endpoint with strategy parameter keeps the API simple while supporting
multiple summarization approaches. Default strategy is extractive for speed.
