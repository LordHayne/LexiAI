# LexiAI Home Assistant Integration (Custom Component)

Minimal setup to push Home Assistant state changes to the Lexi backend.

## Install (manual)

1. Copy `home_assistant/custom_components/lexiai/` into your Home Assistant config:
   - `config/custom_components/lexiai/`
2. Restart Home Assistant.
3. Add integration via UI: Settings -> Devices & Services -> Add Integration -> "LexiAI"

## Configuration fields

- Base URL: Lexi backend URL (e.g. `http://localhost:8000`)
- API key: Lexi API key (header `X-API-Key`)
- Domains: Which entity domains to send
- Batch interval: Seconds to batch events before sending
- Batch size: Max events per batch
- Timeout: HTTP request timeout

## Connectivity check (UI)

After setup, a button entity "LexiAI Connectivity check" is added. Press it to test `/v1/health`.

## Backend endpoint

The integration sends `POST /v1/ha/events` with a batch payload:

```
{
  "source": "home_assistant",
  "events": [
    {
      "entity_id": "light.living_room",
      "domain": "light",
      "state": "on",
      "attributes": {"brightness": 255},
      "last_changed": "2025-01-01T12:00:00+00:00",
      "last_updated": "2025-01-01T12:00:00+00:00",
      "time_fired": "2025-01-01T12:00:00+00:00"
    }
  ]
}
```

## Notes

- Phase 1 only sends `state_changed` events for selected domains.
- Keep domains tight at first to avoid noise.

---

# LexiAI Home Assistant Integration (Custom Component) - Deutsch

Minimal-Setup, um Home Assistant State-Changes an das Lexi Backend zu senden.

## Installation (manuell)

1. Kopiere `home_assistant/custom_components/lexiai/` in dein Home Assistant Config-Verzeichnis:
   - `config/custom_components/lexiai/`
2. Home Assistant neu starten.
3. Integration im UI hinzufuegen: Einstellungen -> Geraete & Dienste -> Integration hinzufuegen -> "LexiAI"

## Konfigurationsfelder

- Base URL: Lexi Backend URL (z. B. `http://localhost:8000`)
- API Key: Lexi API Key (Header `X-API-Key`)
- Domains: Welche Domains gesendet werden
- Batch Interval: Sekunden zum Sammeln der Events
- Batch Size: Maximale Anzahl Events pro Batch
- Timeout: HTTP Request Timeout

## Connectivity Check (UI)

Nach dem Setup gibt es eine Button-Entity "LexiAI Connectivity check". Damit kannst du `/v1/health` testen.

## Backend-Endpoint

Die Integration sendet `POST /v1/ha/events` mit einem Batch-Payload:

```
{
  "source": "home_assistant",
  "events": [
    {
      "entity_id": "light.wohnzimmer",
      "domain": "light",
      "state": "on",
      "attributes": {"brightness": 255},
      "last_changed": "2025-01-01T12:00:00+00:00",
      "last_updated": "2025-01-01T12:00:00+00:00",
      "time_fired": "2025-01-01T12:00:00+00:00"
    }
  ]
}
```

## Hinweise

- Phase 1 sendet nur `state_changed` Events fuer ausgewaehlte Domains.
- Domains am Anfang klein halten, um Noise zu vermeiden.
