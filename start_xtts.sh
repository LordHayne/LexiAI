#!/bin/bash

# XTTS Server Startup Script für LexiAI
# Startet den XTTS Docker Container mit Lexi's Voice

set -e

echo "🎤 Starte XTTS Server für LexiAI..."
echo ""

# Prüfe ob Docker läuft
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker ist nicht gestartet!"
    echo "Bitte starte Docker Desktop und führe dieses Script erneut aus."
    exit 1
fi

# Prüfe ob Trainingsdatei existiert
if [ ! -f "lexi_voice_training.wav" ]; then
    echo "❌ Trainingsdatei nicht gefunden: lexi_voice_training.wav"
    echo "Bitte stelle sicher, dass die Datei im Hauptverzeichnis liegt."
    exit 1
fi

echo "✅ Docker läuft"
echo "✅ Trainingsdatei gefunden ($(ls -lh lexi_voice_training.wav | awk '{print $5}'))"
echo ""

# Erstelle Cache-Verzeichnis
mkdir -p xtts_cache

# Stoppe alten Container falls vorhanden
if docker ps -a --format '{{.Names}}' | grep -q "^lexi-xtts-server$"; then
    echo "🔄 Stoppe alten XTTS Container..."
    docker stop lexi-xtts-server > /dev/null 2>&1 || true
    docker rm lexi-xtts-server > /dev/null 2>&1 || true
fi

# Starte XTTS Server
echo "🚀 Starte XTTS Server (Port 8020)..."
echo "⏳ Erster Start kann 30-60 Sekunden dauern (Model-Download)..."
echo ""

docker-compose -f docker-compose-xtts.yml up -d

# Warte auf Health Check
echo "⏳ Warte auf Server-Initialisierung..."
sleep 5

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8020/health > /dev/null 2>&1; then
        echo ""
        echo "✅ XTTS Server läuft erfolgreich!"
        echo ""
        echo "📊 Server-Informationen:"
        echo "   URL: http://localhost:8020"
        echo "   Swagger UI: http://localhost:8020/docs"
        echo "   Voice: Lexi (aus lexi_voice_training.wav)"
        echo ""
        echo "🔧 Nützliche Kommandos:"
        echo "   Logs anzeigen:  docker-compose -f docker-compose-xtts.yml logs -f"
        echo "   Server stoppen: docker-compose -f docker-compose-xtts.yml down"
        echo "   Neu starten:    docker-compose -f docker-compose-xtts.yml restart"
        echo ""

        # Test-Request
        echo "🧪 Führe Test-Request aus..."
        curl -s -X POST http://localhost:8020/tts_to_audio/ \
          -H "Content-Type: application/json" \
          -d '{
            "text": "Hallo, ich bin Lexi. Der XTTS Server läuft erfolgreich!",
            "language": "de",
            "speaker_wav": "/app/voices/lexi.wav"
          }' \
          -o /tmp/xtts_test.wav

        if [ -f /tmp/xtts_test.wav ] && [ -s /tmp/xtts_test.wav ]; then
            echo "✅ Test-Audio generiert ($(ls -lh /tmp/xtts_test.wav | awk '{print $5}'))"
            echo "   Gespeichert unter: /tmp/xtts_test.wav"
            echo ""
            echo "🎉 XTTS ist bereit für LexiAI!"
        else
            echo "⚠️  Test-Audio konnte nicht generiert werden"
            echo "   Prüfe die Logs: docker-compose -f docker-compose-xtts.yml logs"
        fi

        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

echo ""
echo "❌ Server konnte nicht gestartet werden (Timeout nach ${MAX_RETRIES} Versuchen)"
echo ""
echo "📋 Logs anzeigen:"
docker-compose -f docker-compose-xtts.yml logs --tail=50

exit 1
