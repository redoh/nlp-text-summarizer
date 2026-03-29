# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue to report a security vulnerability.**

Instead, report vulnerabilities by sending an email to **security@nlp-text-summarizer.dev**. Include the following information in your report:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact of the vulnerability
- Any suggested fixes (if applicable)

## Response Timeline

- **Acknowledgment**: You will receive an initial acknowledgment within **48 hours** of your report.
- **Assessment**: We aim to provide a preliminary assessment of the vulnerability within **5 business days**.
- **Resolution**: Critical vulnerabilities will be patched as quickly as possible. Non-critical issues will be addressed in the next scheduled release.
- **Disclosure**: We will coordinate with you on public disclosure timing. We request a minimum of **90 days** before public disclosure to allow adequate time for a fix.

## Scope

This security policy covers the following:

- The NLP Text Summarizer API application code (everything under `app/`)
- API endpoint security (authentication, authorization, input validation)
- Dependency vulnerabilities in direct and transitive dependencies
- Docker image and container security
- Data handling and processing of user-submitted text

The following are **out of scope**:

- Third-party services or infrastructure not maintained by this project
- Vulnerabilities in upstream ML models (e.g., facebook/bart-large-cnn)
- Denial-of-service attacks that rely solely on sending large volumes of traffic
- Issues in development-only dependencies that do not affect production deployments
