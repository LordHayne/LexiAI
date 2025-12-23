# Repository Guidelines

This guide helps contributors navigate LexiAI and follow existing project conventions.

## Project Structure & Module Organization
- `backend/`: FastAPI service, memory system, Qdrant integration, and core logic.
- `frontend/`: Static UI pages plus build scripts.
- `tests/`: Pytest suite (unit, integration, performance, security).
- `docs/`: Project documentation and setup notes.
- Entry points: `main.py` (CLI) and `start_middleware.py` (API).
- Config lives in `backend/config/` (see `backend/config/persistent_config.json`).

## Build, Test, and Development Commands
- Install Python deps: `pip install -r requirements.txt`
- Start API: `python start_middleware.py` (API server)
- Frontend build: `cd frontend && npm install && npm run build`
- Frontend watch: `cd frontend && npm run watch`
- Test suite: `make test` (coverage), `make test-fast` (no coverage)
- Lint/format/typecheck: `make lint`, `make format`, `make typecheck`
- Dev API (uses venv + .env): `make dev`

## Coding Style & Naming Conventions
- Python: 4-space indentation, max line length 120.
- Formatters/linters: `black`, `isort`, `flake8`, `pylint`.
- Type checking: `mypy` (ignores missing imports).
- Tests follow `test_*.py` naming and live under `tests/`.

## Testing Guidelines
- Framework: `pytest` with markers for `integration`, `performance`, `security`.
- Coverage is produced by `make test` (HTML in `htmlcov/`).
- Run targeted tests with `pytest tests/test_profile_builder.py -v` or `make test-one TEST=...`.

## Commit & Pull Request Guidelines
- Recent history uses conventional prefixes like `feat:`, `fix:`, `chore:` plus occasional `Update ...`.
- Keep commits small and imperative (e.g., `fix: handle missing user on update`).
- PRs should include: summary, testing run, linked issue (if any), and screenshots for UI changes.

## Configuration & Secrets
- Use `.env` for local secrets; do not commit it.
- Qdrant and LLM endpoints are configured via `backend/config/persistent_config.json` or env vars.
