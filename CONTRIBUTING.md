# Contributing to NLP Text Summarizer

Thank you for your interest in contributing! This document outlines the process and guidelines for contributing to this project.

## Setting Up the Development Environment

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd nlp-text-summarizer
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. Verify everything works:

   ```bash
   pytest tests/ -v
   ruff check app/ tests/
   ```

5. Start the development server:

   ```bash
   uvicorn app.main:app --reload
   ```

## Branch Naming Convention

Use the following prefixes for your branches:

- `feature/` -- New features or enhancements (e.g., `feature/add-batch-summarization`)
- `fix/` -- Bug fixes (e.g., `fix/extractive-sentence-splitting`)
- `docs/` -- Documentation changes (e.g., `docs/update-api-examples`)

Always branch off `main` and keep your branch up to date with it.

## Pull Request Process

1. **Create a descriptive PR** -- Fill out the pull request template completely. Include a summary of what changed and why.
2. **All tests must pass** -- Run `pytest tests/ -v` locally before opening a PR.
3. **Lint must pass** -- Run `ruff check app/ tests/` and `ruff format app/ tests/` to ensure your code is clean.
4. **Keep PRs focused** -- Each PR should address a single concern. Split unrelated changes into separate PRs.
5. **Request a review** -- At least one approval is required before merging.

## Code Style

- **Linting and formatting**: We use [ruff](https://docs.astral.sh/ruff/) for both linting and formatting. Run `ruff check` and `ruff format` before committing.
- **Type hints**: Use type hints on all function signatures and variable declarations where the type is not obvious. This project targets Python 3.10+.
- **Pydantic models**: All request and response schemas must be defined as Pydantic models in `app/models/schemas.py`.
- **Strategy pattern**: Summarization services follow the strategy pattern. New strategies must inherit from the base class in `app/services/base.py`.
- **Thin routes**: Keep route handlers in `app/routers/` thin. Business logic belongs in `app/services/`.
- **Configuration**: Use pydantic-settings for all configuration. Do not hardcode values.

## Testing Requirements

- **All new features must include tests.** Add unit tests in `tests/test_services.py` and API integration tests in `tests/test_api.py` as appropriate.
- **Use pytest fixtures** for shared setup and teardown.
- **Use httpx AsyncClient** for API integration tests.
- **Maintain or improve coverage** -- do not submit PRs that reduce test coverage without justification.
- **Run the full test suite** before submitting:

  ```bash
  pytest tests/ -v
  ```
