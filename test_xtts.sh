#!/bin/bash

# XTTS Integration Test Script für LexiAI
# Testet die komplette XTTS-Pipeline

set -e

echo "🧪 XTTS Integration Test"
echo "========================"
echo ""

# Farben für Output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test Counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Test 1: Docker läuft
echo "Test 1: Docker Status"
if docker info > /dev/null 2>&1; then
    print_success "Docker läuft"
else
    print_error "Docker ist nicht gestartet"
    echo "Bitte starte Docker Desktop und führe das Script erneut aus."
    exit 1
fi
echo ""

# Test 2: XTTS Container läuft
echo "Test 2: XTTS Container Status"
if docker ps --format '{{.Names}}' | grep -q "^lexi-xtts-server$"; then
    print_success "XTTS Container läuft"

    # Container Details
    CONTAINER_STATUS=$(docker inspect lexi-xtts-server --format='{{.State.Status}}')
    CONTAINER_UPTIME=$(docker inspect lexi-xtts-server --format='{{.State.StartedAt}}')
    print_info "Status: $CONTAINER_STATUS"
    print_info "Gestartet: $CONTAINER_UPTIME"
else
    print_error "XTTS Container läuft nicht"
    echo ""
    echo "Starte Container mit:"
    echo "  ./start_xtts.sh"
    echo ""
    exit 1
fi
echo ""

# Test 3: XTTS Health Endpoint
echo "Test 3: XTTS Health Endpoint"
HEALTH_RESPONSE=$(curl -s http://localhost:8020/health)
if echo "$HEALTH_RESPONSE" | grep -q "ok\|healthy\|running"; then
    print_success "XTTS Server antwortet"
    print_info "Response: $HEALTH_RESPONSE"
else
    print_error "XTTS Health Check fehlgeschlagen"
    echo "Response: $HEALTH_RESPONSE"
fi
echo ""

# Test 4: Speaker-Datei im Container verfügbar
echo "Test 4: Speaker-Datei (Trainingsdatei)"
if docker exec lexi-xtts-server test -f /app/voices/lexi.wav 2>/dev/null; then
    print_success "Speaker-Datei im Container vorhanden"

    # Datei-Info
    FILE_SIZE=$(docker exec lexi-xtts-server stat -f%z /app/voices/lexi.wav 2>/dev/null || docker exec lexi-xtts-server stat -c%s /app/voices/lexi.wav 2>/dev/null)
    FILE_SIZE_MB=$(echo "scale=2; $FILE_SIZE / 1024 / 1024" | bc)
    print_info "Größe: ${FILE_SIZE_MB} MB"
else
    print_error "Speaker-Datei nicht gefunden im Container"
    echo ""
    echo "Prüfe ob lexi_voice_training.wav im Hauptverzeichnis liegt"
    echo "Starte Container neu: ./start_xtts.sh"
fi
echo ""

# Test 5: TTS Generierung (kurzer Text)
echo "Test 5: TTS Generierung (Test-Audio)"
TTS_REQUEST=$(cat <<'EOF'
{
    "text": "Hallo, ich bin Lexi. Dies ist ein Test der XTTS-Integration.",
    "language": "de",
    "speaker_wav": "/app/voices/lexi.wav"
}
EOF
)

TTS_START=$(date +%s)
HTTP_CODE=$(curl -s -o /tmp/xtts_test_output.wav -w "%{http_code}" \
    -X POST http://localhost:8020/tts_to_audio/ \
    -H "Content-Type: application/json" \
    -d "$TTS_REQUEST")
TTS_END=$(date +%s)
TTS_DURATION=$((TTS_END - TTS_START))

if [ "$HTTP_CODE" = "200" ] && [ -f /tmp/xtts_test_output.wav ] && [ -s /tmp/xtts_test_output.wav ]; then
    print_success "TTS-Audio erfolgreich generiert"

    FILE_SIZE=$(stat -f%z /tmp/xtts_test_output.wav 2>/dev/null || stat -c%s /tmp/xtts_test_output.wav 2>/dev/null)
    FILE_SIZE_KB=$(echo "scale=2; $FILE_SIZE / 1024" | bc)

    print_info "Verarbeitungszeit: ${TTS_DURATION}s"
    print_info "Audio-Größe: ${FILE_SIZE_KB} KB"
    print_info "Gespeichert unter: /tmp/xtts_test_output.wav"

    # Optinal: Audio abspielen (macOS)
    if command -v afplay &> /dev/null; then
        print_info "Audio wird abgespielt..."
        afplay /tmp/xtts_test_output.wav &
    fi
else
    print_error "TTS-Generierung fehlgeschlagen (HTTP $HTTP_CODE)"
fi
echo ""

# Test 6: LexiAI Backend API (falls läuft)
echo "Test 6: LexiAI Backend Integration"
if curl -s http://localhost:8000/api/audio/health > /dev/null 2>&1; then
    BACKEND_HEALTH=$(curl -s http://localhost:8000/api/audio/health)

    if echo "$BACKEND_HEALTH" | grep -q "xtts_server"; then
        print_success "Backend erkennt XTTS Server"

        # XTTS Status aus Backend
        XTTS_STATUS=$(echo "$BACKEND_HEALTH" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('xtts_server', 'unknown'))" 2>/dev/null || echo "unknown")
        print_info "XTTS Status (Backend): $XTTS_STATUS"
    else
        print_error "Backend erkennt XTTS Server nicht"
    fi
else
    print_info "LexiAI Backend läuft nicht (optional)"
    echo "  Starte Backend mit: python start_middleware.py"
fi
echo ""

# Test 7: XTTS Konfiguration in persistent_config.json
echo "Test 7: Konfiguration"
if [ -f backend/config/persistent_config.json ]; then
    if grep -q "xtts_url" backend/config/persistent_config.json; then
        print_success "XTTS-Konfiguration vorhanden"

        # Config-Werte extrahieren
        XTTS_URL=$(python3 -c "import json; print(json.load(open('backend/config/persistent_config.json')).get('xtts_url', 'N/A'))" 2>/dev/null || echo "N/A")
        XTTS_LANG=$(python3 -c "import json; print(json.load(open('backend/config/persistent_config.json')).get('xtts_language', 'N/A'))" 2>/dev/null || echo "N/A")

        print_info "XTTS URL: $XTTS_URL"
        print_info "Sprache: $XTTS_LANG"
    else
        print_error "XTTS-Konfiguration fehlt in persistent_config.json"
    fi
else
    print_error "persistent_config.json nicht gefunden"
fi
echo ""

# Test 8: Container Resource Usage
echo "Test 8: Resource-Nutzung"
CONTAINER_STATS=$(docker stats lexi-xtts-server --no-stream --format "{{.MemUsage}}\t{{.CPUPerc}}" 2>/dev/null || echo "N/A\tN/A")
if [ "$CONTAINER_STATS" != "N/A\tN/A" ]; then
    MEM_USAGE=$(echo "$CONTAINER_STATS" | awk '{print $1}')
    CPU_USAGE=$(echo "$CONTAINER_STATS" | awk '{print $2}')

    print_success "Resource-Monitoring aktiv"
    print_info "RAM-Nutzung: $MEM_USAGE"
    print_info "CPU-Nutzung: $CPU_USAGE"
else
    print_info "Resource-Stats nicht verfügbar"
fi
echo ""

# Zusammenfassung
echo "================================"
echo "📊 Test-Zusammenfassung"
echo "================================"
echo -e "${GREEN}Erfolgreich: $TESTS_PASSED${NC}"
echo -e "${RED}Fehlgeschlagen: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 Alle Tests bestanden!${NC}"
    echo ""
    echo "XTTS ist betriebsbereit für LexiAI!"
    echo ""
    echo "Nächste Schritte:"
    echo "  1. Starte LexiAI Backend: python start_middleware.py"
    echo "  2. Öffne UI: http://localhost:8000/ui"
    echo "  3. Teste Audio-Pipeline mit Spracheingabe"
    echo ""
    exit 0
else
    echo -e "${YELLOW}⚠️  Einige Tests sind fehlgeschlagen${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  - Prüfe Logs: docker-compose -f docker-compose-xtts.yml logs"
    echo "  - Neustart: ./start_xtts.sh"
    echo "  - Dokumentation: cat XTTS_INTEGRATION.md"
    echo ""
    exit 1
fi
