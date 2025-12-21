#!/bin/bash
# LexiAI Setup Verification Script
# Prüft ob alle Voraussetzungen für den Start erfüllt sind

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 LexiAI Setup Verification${NC}"
echo "======================================"
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Virtual Environment
echo "1️⃣ Virtual Environment"
if [ -d ".venv" ]; then
    echo -e "${GREEN}✅ .venv exists${NC}"
    if [ -f ".venv/bin/python3" ]; then
        VERSION=$(.venv/bin/python3 --version)
        echo -e "${GREEN}✅ Python: $VERSION${NC}"
    else
        echo -e "${RED}❌ .venv/bin/python3 not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}❌ .venv directory not found${NC}"
    echo "   Run: ./setup.sh"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: Configuration Files
echo "2️⃣ Configuration Files"
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env exists${NC}"
    if grep -q "LEXI_API_KEY_ENABLED=False" .env; then
        echo -e "${GREEN}   → API Key authentication disabled (dev mode)${NC}"
    elif grep -q "LEXI_API_KEY=" .env; then
        echo -e "${GREEN}   → API Key configured${NC}"
    else
        echo -e "${YELLOW}⚠️  API Key not configured${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${RED}❌ .env not found${NC}"
    echo "   Run: cp .env.example .env"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "backend/config/persistent_config.json" ]; then
    echo -e "${GREEN}✅ persistent_config.json exists${NC}"
else
    echo -e "${RED}❌ persistent_config.json not found${NC}"
    echo "   Run: cp backend/config/persistent_config.json.example backend/config/persistent_config.json"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 3: Dependencies
echo "3️⃣ Python Dependencies"
if [ -f ".venv/bin/pip" ]; then
    # Check key packages
    source .venv/bin/activate

    PACKAGES=("fastapi" "uvicorn" "langchain" "qdrant-client" "ollama")
    ALL_INSTALLED=true

    for pkg in "${PACKAGES[@]}"; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo -e "${GREEN}✅ $pkg installed${NC}"
        else
            echo -e "${RED}❌ $pkg not installed${NC}"
            ALL_INSTALLED=false
            ERRORS=$((ERRORS + 1))
        fi
    done

    if [ "$ALL_INSTALLED" = false ]; then
        echo ""
        echo "   Run: ./setup.sh or pip install -r requirements.txt"
    fi
else
    echo -e "${RED}❌ pip not found in .venv${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 4: External Services
echo "4️⃣ External Services"

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running (http://localhost:11434)${NC}"

    # Check if models are available
    MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")
    if echo "$MODELS" | grep -q "gemma"; then
        echo -e "${GREEN}   → gemma model available${NC}"
    else
        echo -e "${YELLOW}⚠️  gemma model not found${NC}"
        echo "      Run: ollama pull gemma3:4b-it-qat"
        WARNINGS=$((WARNINGS + 1))
    fi

    if echo "$MODELS" | grep -q "nomic"; then
        echo -e "${GREEN}   → nomic-embed-text available${NC}"
    else
        echo -e "${YELLOW}⚠️  nomic-embed-text not found${NC}"
        echo "      Run: ollama pull nomic-embed-text"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠️  Ollama not running or not accessible${NC}"
    echo "   Start: ollama serve"
    echo "   Install: https://ollama.ai"
    WARNINGS=$((WARNINGS + 1))
fi

# Check Qdrant
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant is running (http://localhost:6333)${NC}"
else
    echo -e "${YELLOW}⚠️  Qdrant not running or not accessible${NC}"
    echo "   Start: docker run -p 6333:6333 qdrant/qdrant"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check 5: Directories
echo "5️⃣ Required Directories"
DIRS=("logs" "models" "backend/logs")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅ $dir/${NC}"
    else
        echo -e "${YELLOW}⚠️  $dir/ not found (will be created automatically)${NC}"
    fi
done
echo ""

# Check 6: Permissions
echo "6️⃣ Permissions"
if [ -x "setup.sh" ]; then
    echo -e "${GREEN}✅ setup.sh is executable${NC}"
else
    echo -e "${YELLOW}⚠️  setup.sh not executable${NC}"
    echo "   Run: chmod +x setup.sh"
    WARNINGS=$((WARNINGS + 1))
fi

if [ -x "verify_setup.sh" ]; then
    echo -e "${GREEN}✅ verify_setup.sh is executable${NC}"
else
    echo -e "${YELLOW}⚠️  verify_setup.sh not executable${NC}"
fi
echo ""

# Summary
echo "======================================"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 Setup Complete - Ready to Start!${NC}"
    echo ""
    echo "Start LexiAI with:"
    echo "  source .venv/bin/activate"
    echo "  python start_middleware.py"
    echo ""
    echo "Or read QUICK_START.md for more options"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Setup OK with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can start LexiAI, but some features may not work."
    echo "See warnings above for details."
    echo ""
    echo "Start with:"
    echo "  source .venv/bin/activate"
    echo "  python start_middleware.py"
else
    echo -e "${RED}❌ Setup Incomplete - $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before starting."
    echo "Run: ./setup.sh to auto-fix most issues"
    exit 1
fi
echo "======================================"
