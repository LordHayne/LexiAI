#!/bin/bash

# Test Voice Integration End-to-End
# ==================================

echo "🧪 Testing LexiAI Voice Integration..."
echo ""

# Check if server is running
if ! lsof -i:8000 | grep -q LISTEN; then
    echo "❌ Server not running on port 8000"
    echo "   Start with: ./start_voice_server.sh"
    exit 1
fi

echo "✅ Server is running on port 8000"
echo ""

# Test 1: Health Endpoint
echo "Test 1: Health Check"
response=$(curl -s http://localhost:8000/health)
if [ $? -eq 0 ]; then
    echo "✅ Health endpoint OK"
    echo "   Response: $response"
else
    echo "❌ Health endpoint failed"
    exit 1
fi
echo ""

# Test 2: TTS Settings
echo "Test 2: TTS Settings"
response=$(curl -s http://localhost:8000/api/tts/settings)
if [ $? -eq 0 ]; then
    echo "✅ TTS settings endpoint OK"
    echo "   $response" | python3 -m json.tool 2>/dev/null || echo "   $response"
else
    echo "❌ TTS settings failed"
fi
echo ""

# Test 3: TTS Health
echo "Test 3: TTS Health Check"
response=$(curl -s http://localhost:8000/api/tts/health)
if [ $? -eq 0 ]; then
    echo "✅ TTS health OK"
    echo "   $response" | python3 -m json.tool 2>/dev/null || echo "   $response"
else
    echo "❌ TTS health failed"
fi
echo ""

# Test 4: Simple TTS Test (create a test audio file)
echo "Test 4: TTS Synthesis"
curl -s -X POST "http://localhost:8000/api/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hallo, ich bin Lexi und teste die Voice Integration!"}' \
  -o /tmp/test_tts.mp3

if [ -f /tmp/test_tts.mp3 ] && [ -s /tmp/test_tts.mp3 ]; then
    size=$(wc -c < /tmp/test_tts.mp3)
    echo "✅ TTS synthesis OK"
    echo "   Generated audio: $size bytes"
    echo "   Saved to: /tmp/test_tts.mp3"
    echo "   Play with: afplay /tmp/test_tts.mp3"
else
    echo "❌ TTS synthesis failed"
fi
echo ""

# Test 5: Frontend Accessibility
echo "Test 5: Frontend UI"
response=$(curl -s -I http://localhost:8000/chat_ui.html | head -1)
if echo "$response" | grep -q "200"; then
    echo "✅ Frontend UI accessible"
    echo "   Open: http://localhost:8000/chat_ui.html"
else
    echo "❌ Frontend UI not accessible"
    echo "   Response: $response"
fi
echo ""

echo "==========================================="
echo "✅ All tests completed!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Open http://localhost:8000/chat_ui.html in browser"
echo "2. Click microphone button to start recording"
echo "3. Speak your question"
echo "4. Listen to Lexi's voice response"
echo ""
echo "Pipeline:"
echo "  🎤 Voice → Whisper STT → Ollama LLM → gTTS → 🔊 Audio"
echo ""

