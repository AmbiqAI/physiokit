# Contributing

Thanks for your interest in contributing to physiokit! This guide covers the basics to get you started.

## Quick start

### Prerequisites
- Python 3.12+ (see `pyproject.toml` for supported versions).
- Recommended: `uv` for dependency management.

### Setup
1. Fork the repo and create a branch:
   ```bash
   git checkout -b your-name/short-description
   ```
2. Install dev dependencies:
   ```bash
   uv sync --group dev
   ```

### Common commands
- Format code:
  ```bash
  uv run ruff format
  ```
- Lint:
  ```bash
  uv run ruff check
  ```
- Run tests:
  ```bash
  uv run pytest tests/
  ```

### Docs (optional)
```bash
uv sync --group docs
uv run mkdocs serve
```

## Pull requests
- Keep PRs focused and scoped to a single change.
- Include tests for fixes and new features.
- Update docs if behavior or APIs change.
- Ensure formatting/linting/test commands pass before requesting review.

## Reporting issues
If you find a bug or have a feature request, open an issue with:
- A clear description and expected behavior.
- Steps to reproduce (if applicable).
- Relevant logs, screenshots, or environment details.

Thanks again for contributing!
