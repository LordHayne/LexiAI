# Feedback Learning Update

## Wichtigste Änderungen
- Feedback-Turn-IDs werden korrekt durchgereicht; UI erzeugt keine Fake-IDs mehr.
- Feedback und Turns werden in Qdrant persistiert, inkl. Lazy-Hydration.
- Self-Correction markiert Feedback als verarbeitet und vermeidet doppelte Korrekturen.
- Explizite Korrekturen werden sofort als `self_correction`-Memory gespeichert.
- Immediate-Feedback läuft über Qdrant-Queue (`immediate_pending`) und einen Background-Worker.
- Metrics Dashboard zeigt Feedback-Lernstatus.

## TODOs
- Alte Feedbacks ohne `ts`/`processed` migrieren, damit Hydration/Stats vollständig sind.
- Worker-Startstrategie prüfen (ein Prozess als Worker, keine Mehrfach-Worker in Multi-Proc).
- Optional: Positive Feedbacks in Relevanz/Ranking einfließen lassen.
- Tests ausführen, sobald `pytest` verfügbar ist.

## Relevante Endpoints
- `POST /v1/feedback/thumbs-up`
- `POST /v1/feedback/thumbs-down` (setzt `immediate_pending`)
- `POST /v1/feedback/correction` (speichert sofort Memory)
- `GET /v1/feedback/stats`
- `GET /debug/feedback/status`
