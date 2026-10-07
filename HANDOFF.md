# physioKIT Astro documentation migration

Goal: Replace the public MkDocs site with Astro/Starlight and shared helia-ui, preserving useful content, API reference, plots, and stable routes. Tracked by AmbiqAI/physiokit#25.

Current state: Worktree `codex/astro-docs` at `/Users/adam.page/Ambiq/adks/physiokit-astro-docs`, based on main `9b325a6`. Astro source is under `astro-site/`; home, quickstart, seven reference/example pages, 42 generated API modules (106 searchable symbols), and 22 Plotly assets are present. The source notebook remains at `notebooks/docs.ipynb` and links from the quickstart as plot source, with a clear note that it contains data placeholders. Its output path now points to `astro-site/public/assets/`. MkDocs source/config/dependencies have been removed locally. Local commit `4547f7f` contains the migration. No PR or public cutover yet; the published site remains MkDocs.

Verified: `npm run build`, `npm run check`, `npm run check:output`, 3 Playwright tests, `actionlint`, `ruff` on touched Python files, and `git diff --check` passed at the current implementation. Link/asset audit checked 76 HTML files, all local links/anchors, all 22 plots, and notebook source. Rendered light/dark desktop/mobile pages inspected; ECG Plotly iframe rendered in Chromium without failed asset requests. Quickstart ECG example ran in a clean Linux Python 3.12 container (`HrvTimeMetrics`, 9 peaks). Griffe now generates without warnings after three docstring corrections.

Decisions: Preserve existing `/reference/*`, `/tutorial/quickstart/`, and `/api/physiokit/*` paths. Add `/tutorial/` redirect. Use helia-ui alpha.23 official Ambiq blue light/white dark footer and `kit-physio` accent. Keep plots as lazy standalone iframes; they use the Plotly CDN. Keep the 23 MB notebook in its original source location rather than duplicating it in the published site. The `site/` MkDocs output cannot overwrite `astro-site/dist/`.

Next: Present findings before GitHub write; create PR only after approval under the user's supplied AGENTS instructions; resolve review/CI, then cut over and close/supersede logo-only PR #24 after deployment.

Gotchas: `.github/workflows/release.yaml` calls `.github/workflows/docs.yaml` at release tag; the updated docs workflow publishes Astro for both main and release paths. Public support/embedded compatibility claims belong to issue #21 if not source-backed. Local macOS SciPy import fails on `_propack` wheel; this is environment-specific and did not affect build/reference generation or Linux snippet verification.
