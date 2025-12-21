#!/bin/bash

#  Start LexiAI Voice Server mit gTTS Integration
# ================================================

echo "🚀 Starting LexiAI Voice Server..."
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found (.venv/)"
    echo "   Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate venv
source .venv/bin/activate

# Check if gTTS is installed
python3 -c "import gtts" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing gTTS..."
    pip install gtts
fi

echo "✅ Dependencies OK"
echo ""

# Kill any existing server
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "🔊 Voice Integration Features:"
echo "  - STT: Whisper (OpenAI)"
echo "  - LLM: Ollama (gemma3:4b-it-qat)"
echo "  - TTS: gTTS (Google Text-to-Speech)"
echo ""

echo "📡 Starting server on http://localhost:8000"
echo "   - Frontend: http://localhost:8000/chat_ui.html"
echo "   - Health: http://localhost:8000/health"
echo "   - Docs: http://localhost:8000/docs"
echo ""

# Start server
python3 -m uvicorn backend.api.api_server:app --host 0.0.0.0 --port 8000

