# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-29

### Added

- FastAPI application with versioned API routes
- Extractive summarizer using frequency-based sentence scoring
- Abstractive summarizer powered by facebook/bart-large-cnn via HuggingFace Transformers
- Health check endpoint at `/api/v1/health`
- Docker support with multi-stage build
- Test suite with pytest and httpx covering API integration and service unit tests
