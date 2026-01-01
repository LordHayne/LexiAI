# LexiAI
<img width="1452" height="711" alt="Bildschirmfoto 2025-12-22 um 20 45 30" src="https://github.com/user-attachments/assets/9a08bb58-c3d8-4ba2-b4af-520537374a9d" />

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
Clone in your prefered Folder:
```bash
git clone https://github.com/LordHayne/LexiAI.git
```
cd in your folder and: 
```bash
sh setup.sh
```

## Lokale Spracherkennung (Whisper)
Fuer lokale Transkription wird `openai-whisper` genutzt und `ffmpeg` benoetigt.
- `setup.sh` installiert `ffmpeg` automatisch (Homebrew) und zieht Python-Dependencies.
- Bei bestehender Installation: `brew install ffmpeg` und `pip install -r requirements.txt`.



## Weitere Infos
Siehe `docs/project_dokumentation.md` fuer Details.

## Home Assistant (optional)
Home Assistant kann ueber ENV oder Config angebunden werden.

- ENV Variablen:
  - `LEXI_HA_URL` (z. B. `http://homeassistant.local:8123`)
  - `LEXI_HA_TOKEN` (Long-Lived Access Token)
- UI: `frontend/pages/config_ui.html` enthaelt einen Verbindungstest (`/v1/config/test-ha`).
