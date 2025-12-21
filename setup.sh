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

# Check Python version
echo "1️⃣ Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "2️⃣ Creating virtual environment..."
    python3 -m venv .venv
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

# Install dependencies
echo ""
echo "5️⃣ Installing dependencies from requirements.txt..."
echo -e "${YELLOW}   This may take a few minutes...${NC}"
pip install --quiet -r requirements.txt
echo -e "${GREEN}✅ All dependencies installed${NC}"

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

# Test configuration
echo ""
echo "7️⃣ Testing configuration..."
ENV=development LEXI_API_KEY_ENABLED=False python3 -c "
from backend.config.auth_config import SecurityConfig
from backend.config.cors_config import CORSConfig
print('✅ Configuration modules loaded successfully')
" || {
    echo -e "${RED}❌ Configuration test failed${NC}"
    exit 1
}

# Check if Ollama is running
echo ""
echo "8️⃣ Checking external services..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama is not running or not accessible${NC}"
    echo "   Please start Ollama: https://ollama.ai"
fi

# Check if Qdrant is running
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant is running${NC}"
else
    echo -e "${YELLOW}⚠️  Qdrant is not running or not accessible${NC}"
    echo "   Please start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
fi

# Create necessary directories
echo ""
echo "9️⃣ Creating necessary directories..."
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
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Start the middleware server:"
echo "   python start_middleware.py"
echo ""
echo "   OR start the CLI:"
echo "   python main.py"
echo ""
echo "3. Check if external services are running:"
echo "   - Ollama: http://localhost:11434"
echo "   - Qdrant: http://localhost:6333"
echo ""
echo "For more information, see SECURITY.md"
echo ""
