# LexiAI

Lokales Assistenzsystem mit Chat-API, Memory-System, optionaler Tool-Integration
(z. B. Home Assistant) und UI-Frontends.

## Features
- Chat-Verarbeitung mit optionalem Tool-Calling
- Langzeit-Memory in Qdrant (Retrieval, Filter, Feedback)
- Profil-/Kontext-Anreicherung
- Smart-Home-Integration (Home Assistant)
- Optionale Web-Search-Integration

## Struktur
- `backend/` FastAPI-Service, Chat-Processing, Memory, Qdrant-Anbindung
- `frontend/` Statische UI-Seiten und Build-Skripte
- `tests/` Pytest + Integrations-/Performance-Tests
- `docs/` Zentrale Dokumentation
- Entrypoints: `main.py` (CLI), `start_middleware.py` (API)

## Quickstart
Backend:
```bash
pip install -r requirements.txt
python start_middleware.py
```

Frontend:
```bash
cd frontend
npm install
npm run build
```

## Konfiguration
- Beispiel-ENV: `.env.example`
- Persistente Config: `backend/config/persistent_config.json`
- Qdrant Host/Port via Config oder ENV
- LLM/Embedding (z. B. Ollama) via Config oder ENV

## Tests
```bash
make test
make test-fast
make lint
make format
make typecheck
```

## Weitere Infos
Siehe `docs/project_dokumentation.md` fuer Details.
