# Home Assistant Roadmap (LexiAI)

## Ziel
Lexi soll Smart-Home-Aktionen sicher steuern und schrittweise intelligente Automationsvorschlaege liefern.

## Phase 0: Grundlagen & Sicherheit
- [x] HA-Token/URL-Pruefung in der Konfiguration dokumentieren
- [x] Klarere Fehler bei fehlender Konfiguration (UI + API)
- [x] Logging fuer HA-Requests (ohne Tokens)
- [x] Feature-Flag-Check in allen HA-Entry-Points

## Phase 1: Robuste Steuerung
- [x] Entity-Aufloesung stabilisieren (Friendly Names + Domain-Praeferenzen)
- [x] Status-Verifikation nach Steuerung (Post-Action Check)
- [x] Area/Device-Info einbinden (fuer bessere Aufloesung)
- [x] Fehlertexte fuer User klarer formulieren

## Phase 2: Automations- und Script-Erstellung (Basics)
- [x] Neue Tools: home_assistant_create_automation
- [x] Neue Tools: home_assistant_create_script
- [x] Preview-Modus (YAML/JSON anzeigen, keine direkte Speicherung)
- [x] Validation: Entities existieren, Services sind erlaubt
- [x] Apply-Modus mit expliziter User-Bestaetigung
- [x] Rollback (vorherige Automation sichern und bei Fehler wiederherstellen)

## Phase 3: Automationsvorschlaege aus Mustern
- [ ] Muster-Collector (Qdrant + HA-History)
- [ ] Scoring: Haeufigkeit, Regelmaessigkeit, Tageszeit, Wochentag
- [ ] Vorschlags-API + UI-Ansicht
- [ ] Feedback-Loop (Daumen hoch/runter, "zu aggressiv")
- [ ] Dynamische Thresholds pro User

## Phase 3.5: Geraete-Gesundheit & Batterien
- [ ] Health-Checks: State fehlt/unknown/unavailable erkennen
- [ ] Batterie-Warnungen (sensor.*battery* oder device_class=battery)
- [ ] Vorschlagslogik: "Geraet pruefen" oder "Batterie ersetzen"
- [ ] UI-Panel fuer Geraetezustand (kritisch, warnung, ok)

## Phase 4: Kontext-Intelligenz (Roadmap)
- [ ] Praesenz/Geofencing als Trigger ("User zu Hause")
- [ ] Kontext-Features: Sonne, Wetter, Kalender, Medienstatus
- [ ] Adaptive Automationen (selbstanpassende Zeiten)
- [ ] Optional: Scene-Management fuer Praeferenzen

## Offene Fragen
- [ ] Welche Domains duerfen Lexi veraendern (light, switch, climate, etc.)?
- [ ] Soll Lexi Automationen automatisch aktivieren oder nur vorschlagen?
- [ ] Bevorzugst du YAML oder JSON fuer Automations-Preview?
