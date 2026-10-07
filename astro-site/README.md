# physioKIT documentation

This directory is the source for the public Astro/Starlight site at `/physiokit/`. Authored pages live in `src/content/docs/`; Plotly HTML assets live in `public/assets/`. The Python API pages under `src/content/docs/api/physiokit/` and the searchable catalog data are generated at build time from the package source with Griffe and helia-ui's reference renderer. Do not edit generated pages or catalog JSON directly.

Use Node 24 and npm 11. From this directory:

```bash
npm ci
npm run check
npm run build
npm run check:output
npx playwright install chromium-headless-shell
npm test
```

The build uses `uv tool run` for Griffe; `uv` must be on `PATH`. `npm run dev` prepares the API reference before starting a local preview. The site is hosted with the `/physiokit/` base path, so local links should use that prefix.

The 22 standalone Plotly visualizations are saved examples, loaded on demand in iframes. They use Plotly's CDN script and need network access for interaction. `notebooks/docs.ipynb` is their source and writes to `astro-site/public/assets/` when run from `notebooks/`; several cells use data placeholders, so it is not a standalone customer quickstart. Keep the published plot files when changing the notebook unless the regenerated visuals have been reviewed.

`.github/workflows/docs.yaml` builds and tests the Astro artifact for pull requests, publishes it on `main`, and can be called with a release tag from `.github/workflows/release.yaml`. The publishing workflow must continue to deploy the exact artifact that passed the build and browser checks.
