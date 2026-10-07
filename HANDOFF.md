# physioKIT Astro documentation migration

Goal: replace MkDocs with Astro/Starlight and shared helia-ui while preserving useful content, routes, Python API and saved plots. Issue #25; draft PR #26 on `codex/astro-docs`. The public site remains MkDocs until approval and cutover.

Implemented: branded compact hero, shared quick buttons, four main navigation sections, Getting Started overview and own-data pages, expanded quickstart, six icon-bearing signal cards, concise installation and whole-card documentation links. The site generates 42 API modules with 106 searchable symbols and carries 22 Plotly assets. MkDocs source, config and dependencies are removed on this branch. The source notebook stays at `notebooks/docs.ipynb` and is linked as plot source rather than copied into the site.

Verified: Astro check/build/output audit pass for 77 pages and 22 plots. Five browser tests pass, including light/dark quick-button readability and card-body navigation. The authored Python fences pass syntax parsing. Linux Python 3.12 executed the two signal-guide snippets, five overview snippets, the PPG workflow through respiratory-rate derivation, and an IMU counts check. Earlier quickstart and own-data workflows ran in Linux. Touched Python passes ruff; workflow actionlint passed before PR creation.

Review fixes: restrict custom hero anchor rules so shared buttons keep theme-aware styling; correct PPG synthesis arguments and unpacking, filter input, noise arguments, respiratory waveform input and FFT prose; supply the IMU recording rate; remove an invalid single-channel SpO2 example; use synthetic ECG for the short heart-rate example. No performance or clinical-validation claim is inferred from these checks.

Dependency: `header.titleRegularPrefix: 'physio'` needs the immutable helia-ui release containing PR #193. The pin remains alpha.23 and the local preview uses an untracked package patch. A clean npm install removes that preview patch. Do not call the header acceptance complete or merge this PR until the released tag is pinned, its lockfile regenerated, and checks rerun.

Next: resolve review findings, obtain the shared release after approval, update the package pin, recheck rendered light/dark/mobile layouts and CI, then ask Adam to approve the migration. After deployment, close or supersede logo-only PR #24.

Gotchas: existing `/reference/*`, `/tutorial/quickstart/`, and `/api/physiokit/*` routes are retained. `/tutorial/` is authored Getting Started content; `/tutorial/your-data/` is new. Plots are lazy standalone iframes using the Plotly CDN. Notebook data placeholders need external recordings. The release workflow calls the docs workflow, which publishes Astro after cutover. Local macOS SciPy `_propack` import is unreliable; runtime snippet validation used Linux.

Publication: final review fixes are pushed. Fresh CI and Copilot re-reviews requested; verify the final head before approval. Product PRs remain draft pending shared helia-ui approval, release and immutable dependency pins.

Review follow-up: guide prose now names the actual ECG preset and PPG/RSP synthesis APIs, sample-count inputs and tuple returns. Respiratory examples name the selected method.
