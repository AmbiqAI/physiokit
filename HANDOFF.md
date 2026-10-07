# physioKIT Astro documentation migration

Goal: replace MkDocs with Astro/Starlight and shared helia-ui while preserving useful content, routes, Python API and saved plots. Issue #25; PR #26 on `codex/astro-docs`. The public site remains MkDocs until this PR merges and the Astro deployment succeeds.

Implemented: branded compact hero, shared quick buttons, four main navigation sections, Getting Started overview and own-data pages, expanded quickstart, six icon-bearing signal cards, concise installation and whole-card documentation links. The site generates 42 API modules with 106 searchable symbols and carries 22 Plotly assets. MkDocs source, config and dependencies are removed on this branch. The source notebook stays at `notebooks/docs.ipynb` and is linked as plot source rather than copied into the site.

Verified: Astro check/build/output audit pass for 77 pages and 22 plots. Seven browser tests pass, including light/dark quick-button readability and card-body navigation. The authored Python fences pass syntax parsing. Linux Python 3.12 executed the two signal-guide snippets, five overview snippets, the PPG workflow through respiratory-rate derivation, and an IMU counts check. Earlier quickstart and own-data workflows ran in Linux. Touched Python passes ruff; workflow actionlint passed before PR creation.

Review fixes: restrict custom hero anchor rules so shared buttons keep theme-aware styling; correct PPG synthesis arguments and unpacking, filter input, noise arguments, respiratory waveform input and FFT prose; supply the IMU recording rate; remove an invalid single-channel SpO2 example; use synthetic ECG for the short heart-rate example. No performance or clinical-validation claim is inferred from these checks.

Dependency: helia-ui v0.1.0-alpha.24 is published from f7158bf. The consumer pins that immutable tag with an npm 11.19.0 lockfile; a clean install reproduces the header without a local patch.

Next: verify final review and CI for the contributor-command and Python support wording fixes, then merge under Adam's authorization and verify the Pages deployment. After deployment, supersede logo-only PR #24.

Gotchas: existing `/reference/*`, `/tutorial/quickstart/`, and `/api/physiokit/*` routes are retained. `/tutorial/` is authored Getting Started content; `/tutorial/your-data/` is new. Plots are lazy standalone iframes using the Plotly CDN. Notebook data placeholders need external recordings. The release workflow calls the docs workflow, which publishes Astro after cutover. Local macOS SciPy `_propack` import is unreliable; runtime snippet validation used Linux.

Publication: final review fixes are pushed. Fresh CI and Copilot re-reviews requested; verify the final head before approval. The shared release and clean consumer pin are verified; product publication needs green final CI and deployment.

Review follow-up: guide prose now names the actual ECG preset and PPG/RSP synthesis APIs, sample-count inputs and tuple returns. Respiratory examples name the selected method.

Not-found handling: restrict the product hero to the home route so unknown routes show the 404 page; built output and browser checks cover the fallback.


Release validation: clean installation of alpha.24 passed. Final check/build/output checks pass, with 7 rendered acceptance checks passing against the clean released dependency. User authorized merging after green CI.

Final review: contributor instructions now use Astro; install guidance matches requires-python >=3.12,<3.15. API catalog search browser coverage passes.

Final review follow-up: flatten the single ECG lead before noise injection and supply the documented Node 24 version file. Linux Python 3.12 executed the ECG synthesis and every documented noise call: 8000 finite samples with time-varying noise. Astro check/build/output and all seven browser tests pass after this fix.
