# Project Dokumentation (Kurzfassung)

## Zweck
LexiAI ist ein lokales Assistenzsystem mit Chat‑API, Memory‑System, optionaler Tool‑Integration (z. B. Home Assistant) und UI‑Frontends.

## Struktur
- backend/: FastAPI‑Service, Chat‑Processing, Memory, Qdrant‑Anbindung
- frontend/: statische UI‑Seiten und Build‑Skripte
- tests/: pytest + Integrations‑/Performance‑Tests
- docs/: zentrale Dokumentation (diese Datei)
- Entrypoints: main.py (CLI), start_middleware.py (API)

## Hauptfunktionen
- Chat‑Verarbeitung mit optionalem Tool‑Calling
- Langzeit‑Memory in Qdrant (Retrieval, Filter, Feedback)
- Profil‑/Kontext‑Anreicherung
- Smart‑Home‑Integration (Home Assistant)
- Web‑Search Integration (optional)

## Wichtige Änderungen (letzte Runde)
- Tool‑Prompt: keine Tool‑Details/Counts in Antworten, Meta‑Antworten strikt 2 Sätze
- Memory‑Filter: Q/A‑Parsing, Identitäts‑Queries bevorzugen persönliche Fakten
- Meta‑Style‑Queries: Memory‑Retrieval übersprungen
- Multi‑Step‑Synthesis: Memory‑Inhalte eingebunden
- Qdrant‑Wrapper: Kompatibel mit `query_points`
- Self‑Reflection: fehlender Return behoben

Betroffene Dateien:
- backend/core/prompt_builder.py
- backend/core/memory_handler.py
- backend/core/llm_tool_calling.py
- backend/core/llm_multi_step_reasoning.py
- backend/qdrant/client_wrapper.py
- backend/core/llm_self_reflection.py
- backend/core/query_classifier.py

## Setup (Kurz)
- Python deps: `pip install -r requirements.txt`
- API starten: `python start_middleware.py`
- Frontend: `cd frontend && npm install && npm run build`

## Tests
- `make test` (voll)
- `make test-fast` (schnell)
- `make lint`, `make format`, `make typecheck`

## Betrieb / Konfiguration
- Konfig: `backend/config/persistent_config.json`
- ENV‑Beispiele: `.env`
- Qdrant: Host/Port in Config oder ENV
- Ollama: LLM/Embedding URL in Config oder ENV

## Notizen
Diese Datei ist eine komprimierte Zusammenfassung der entfernten MD‑Dokumente.
