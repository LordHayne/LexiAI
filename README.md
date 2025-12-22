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

## Quickstart MacOS
Easy Setup:
```bash
git clone https://github.com/LordHayne/LexiAI.git
sh setup.sh
```




## Weitere Infos
Siehe `docs/project_dokumentation.md` fuer Details.
