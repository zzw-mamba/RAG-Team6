# Project instructions

- Use Python 3.11+ and run commands from this repository root.
- Keep core imports standard-library-only. GPU/data dependencies belong behind optional imports.
- Read README.md and docs/architecture.md before changing module contracts.
- Do not pass Example.answers, rationale, gold_answers, or annotation labels to inference backends.
- Do not treat existing rationale as verified gold reasoning. Require explicit reviewed annotations.
- Never use Mock outputs as research results or silently fall back from a real model to Mock.
- Keep real data, weights, credentials, and generated experiment outputs out of Git.
- Retain distinct critic_stop, max_revisions, invalid_critique and invalid_revision outcomes.
- For behavioral code changes, add meaningful tests and run `python -m unittest discover -s tests -v`.
- Run `python -m ruff check .` when the development extra is installed.
- GPU changes need a documented pilot on the target GPU before claiming training support is validated.
- Do not create or push a GitHub repository or upload competition results unless the user authorizes it.
