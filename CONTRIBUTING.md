# Contributing

Thanks for your interest in contributing to physiokit! This guide covers the basics to get you started.

## Quick start

### Prerequisites
- Python 3.12–3.14 (see `pyproject.toml` for supported versions).
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

### Documentation

Use Node 24 (see `astro-site/.nvmrc`) and sync Python dependencies for generated API extraction. From the repository root:

```bash
uv sync --frozen
cd astro-site
npm ci
npm run dev
```

Before submitting documentation changes, run `npm run check`, `npm run build`, `npm run check:output`, and `npm test` from `astro-site/`. Install the browser once with `npx playwright install chromium`. See [the site README](astro-site/README.md) for authored and generated content sources.

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
