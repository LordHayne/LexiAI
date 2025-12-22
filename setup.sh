#!/bin/bash
# LexiAI Development Setup Script
# This script sets up the complete development environment

set -e  # Exit on error

echo "🚀 LexiAI Development Setup"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python 3.11 on macOS (install via Homebrew if missing)
echo "1️⃣ Checking Python 3.11..."
if [ "$(uname)" != "Darwin" ]; then
    echo -e "${RED}❌ This setup script currently targets macOS only${NC}"
    exit 1
fi

PYTHON_BIN="python3.11"
if ! command -v "${PYTHON_BIN}" &> /dev/null; then
    if command -v brew &> /dev/null; then
        echo -e "${YELLOW}⚠️  Python 3.11 not found. Installing via Homebrew...${NC}"
        brew install python@3.11
    else
        echo -e "${RED}❌ Homebrew not found. Install Python 3.11 manually.${NC}"
        exit 1
    fi
fi

PYTHON_VERSION=$("${PYTHON_BIN}" --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "2️⃣ Creating virtual environment..."
    "${PYTHON_BIN}" -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo ""
    echo "2️⃣ Virtual environment exists"
    echo -e "${GREEN}✅ Using existing .venv${NC}"
fi

# Activate virtual environment
echo ""
echo "3️⃣ Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "4️⃣ Upgrading pip..."
pip install --quiet --upgrade pip
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install dependencies (skip if requirements.txt unchanged)
echo ""
echo "5️⃣ Installing dependencies from requirements.txt..."
echo -e "${YELLOW}   This may take a few minutes...${NC}"
REQ_FILE="requirements.txt"
REQ_HASH_FILE=".requirements.hash"
REQ_HASH="$(shasum -a 256 "${REQ_FILE}" | awk '{print $1}')"
if [ -f "${REQ_HASH_FILE}" ] && [ "$(cat "${REQ_HASH_FILE}")" = "${REQ_HASH}" ]; then
    echo -e "${GREEN}✅ requirements.txt unchanged, skipping install${NC}"
else
    pip install --quiet -r "${REQ_FILE}"
    echo "${REQ_HASH}" > "${REQ_HASH_FILE}"
    echo -e "${GREEN}✅ All dependencies installed${NC}"
fi

# Check if .env exists
echo ""
echo "6️⃣ Checking configuration files..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    if [ -f ".env.example" ]; then
        echo "   Creating .env from .env.example..."
        cp .env.example .env
        echo -e "${GREEN}✅ .env created${NC}"
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
fi

# Check persistent_config.json
if [ -f "backend/config/persistent_config.json" ]; then
    echo -e "${GREEN}✅ persistent_config.json exists${NC}"
else
    echo -e "${YELLOW}⚠️  persistent_config.json not found${NC}"
    if [ -f "backend/config/persistent_config.json.example" ]; then
        echo "   Creating persistent_config.json from example..."
        cp backend/config/persistent_config.json.example backend/config/persistent_config.json
        echo -e "${GREEN}✅ persistent_config.json created${NC}"
    else
        echo -e "${RED}❌ persistent_config.json.example not found${NC}"
        exit 1
    fi
fi

# Load environment defaults
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

LEXI_MEMORY_COLLECTION=${LEXI_MEMORY_COLLECTION:-lexi_memory}
LEXI_MEMORY_DIMENSION=${LEXI_MEMORY_DIMENSION:-768}
QDRANT_HOST=${QDRANT_HOST:-localhost}
QDRANT_PORT=${QDRANT_PORT:-6333}
OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
LLM_MODEL=${LLM_MODEL:-gemma3:4b}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-nomic-embed-text}

if [[ "${QDRANT_HOST}" == http* ]]; then
    QDRANT_URL="${QDRANT_HOST}:${QDRANT_PORT}"
else
    QDRANT_URL="http://${QDRANT_HOST}:${QDRANT_PORT}"
fi

# Ensure auth storage files exist
if [ ! -f "backend/config/users_db.json" ]; then
    echo "{}" > backend/config/users_db.json
fi
if [ ! -f "backend/config/refresh_tokens.json" ]; then
    echo "{}" > backend/config/refresh_tokens.json
fi
chmod 600 backend/config/users_db.json backend/config/refresh_tokens.json 2>/dev/null || true

# Test configuration
echo ""
echo "7️⃣ Testing configuration..."
ENV=development LEXI_API_KEY_ENABLED=False "${PYTHON_BIN}" -c "
from backend.config.auth_config import SecurityConfig
from backend.config.cors_config import CORSConfig
print('✅ Configuration modules loaded successfully')
" || {
    echo -e "${RED}❌ Configuration test failed${NC}"
    exit 1
}

# Install/Start Ollama and download models (macOS)
echo ""
echo "8️⃣ Setting up Ollama..."
if ! command -v ollama > /dev/null 2>&1; then
    if command -v brew > /dev/null 2>&1; then
        brew install ollama
    else
        echo -e "${RED}❌ Homebrew not found. Install Ollama manually: https://ollama.com${NC}"
        exit 1
    fi
fi

if ! curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama not running, starting...${NC}"
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
fi

if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running${NC}"
    if [ -n "${LLM_MODEL}" ]; then
        echo "   Pulling LLM model in background: ${LLM_MODEL}"
        ollama pull "${LLM_MODEL}" > /tmp/ollama_pull_llm.log 2>&1 &
    fi
    if [ -n "${EMBEDDING_MODEL}" ]; then
        echo "   Pulling embedding model in background: ${EMBEDDING_MODEL}"
        ollama pull "${EMBEDDING_MODEL}" > /tmp/ollama_pull_embed.log 2>&1 &
    fi
else
    echo -e "${RED}❌ Ollama is not accessible at ${OLLAMA_URL}${NC}"
    exit 1
fi

# Install/Start Qdrant via Docker Compose
echo ""
echo "9️⃣ Setting up Qdrant (Docker Compose)..."
mkdir -p qdrant_storage
if ! command -v docker > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker not found. Install Docker Desktop to run Qdrant.${NC}"
    exit 1
fi
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi
if ! docker compose version > /dev/null 2>&1; then
    echo -e "${RED}❌ docker compose not found. Update Docker Desktop.${NC}"
    exit 1
fi

docker compose up -d qdrant

if ! curl -s "${QDRANT_URL}/collections" > /dev/null 2>&1; then
    echo -e "${RED}❌ Qdrant is not accessible at ${QDRANT_HOST}:${QDRANT_PORT}${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Qdrant is running${NC}"

# Create collection if needed
if ! curl -s "${QDRANT_URL}/collections/${LEXI_MEMORY_COLLECTION}" | grep -q '"status":"ok"'; then
    echo "   Creating collection: ${LEXI_MEMORY_COLLECTION} (${LEXI_MEMORY_DIMENSION} dims)"
    curl -s -X PUT "${QDRANT_URL}/collections/${LEXI_MEMORY_COLLECTION}" \
        -H "Content-Type: application/json" \
        -d "{\"vectors\":{\"size\":${LEXI_MEMORY_DIMENSION},\"distance\":\"Cosine\"}}"
fi

# Create necessary directories
echo ""
echo "10️⃣ Creating necessary directories..."
mkdir -p logs
mkdir -p models
mkdir -p backend/logs
echo -e "${GREEN}✅ Directories created${NC}"

# Final summary
echo ""
echo "======================================"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Starting middleware server..."
python start_middleware.py
