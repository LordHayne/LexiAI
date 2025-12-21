#!/bin/bash
#
# LexiAI Performance Test Runner
#
# This script automates the setup and execution of performance tests
# with proper environment preparation and result logging.
#
# Usage:
#   ./scripts/run_performance_tests.sh [--skip-cleanup] [--skip-warmup]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_URL="${LEXI_OLLAMA_URL:-http://localhost:11434}"
QDRANT_HOST="${LEXI_QDRANT_HOST:-localhost}"
QDRANT_PORT="${LEXI_QDRANT_PORT:-6333}"
MODEL="${LEXI_LLM_MODEL:-gemma3:4b-it-qat}"

# Parse arguments
SKIP_CLEANUP=false
SKIP_WARMUP=false

for arg in "$@"; do
    case $arg in
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        --skip-warmup)
            SKIP_WARMUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-cleanup    Skip Qdrant database cleanup"
            echo "  --skip-warmup     Skip Ollama model warmup"
            echo "  --help           Show this help message"
            exit 0
            ;;
    esac
done

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="docs/performance_results"
LOG_FILE="${LOG_DIR}/performance_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LexiAI Performance Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Timestamp: ${TIMESTAMP}"
echo -e "Log file: ${LOG_FILE}"
echo ""

# Step 1: Check Prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check Ollama
echo -n "  Checking Ollama... "
if curl -s "${OLLAMA_URL}/api/tags" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: Cannot connect to Ollama at ${OLLAMA_URL}${NC}"
    echo "Please start Ollama first."
    exit 1
fi

# Check Qdrant
echo -n "  Checking Qdrant... "
if curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" > /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: Cannot connect to Qdrant at ${QDRANT_HOST}:${QDRANT_PORT}${NC}"
    echo "Please start Qdrant first:"
    echo "  docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant"
    exit 1
fi

# Check Python environment
echo -n "  Checking Python environment... "
if python -c "import asyncio, logging" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: Python environment not properly configured${NC}"
    exit 1
fi

echo ""

# Step 2: Clean Qdrant Database
if [ "$SKIP_CLEANUP" = false ]; then
    echo -e "${YELLOW}Step 2: Cleaning Qdrant database...${NC}"

    echo -n "  Clearing memory collection... "
    python -c "
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.config.middleware_config import MiddlewareConfig
import logging
logging.basicConfig(level=logging.ERROR)

config = MiddlewareConfig()
qm = QdrantMemoryInterface(config.qdrant_host, config.qdrant_port)
try:
    qm.clear_collection()
    print('✓', end='')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
" && echo -e "${GREEN}${NC}" || echo -e "${RED}${NC}"

    echo -n "  Creating bootstrap memories... "
    python -c "
from backend.memory.bootstrap_memories import create_bootstrap_memories
import logging
logging.basicConfig(level=logging.ERROR)

try:
    create_bootstrap_memories()
    print('✓', end='')
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
" && echo -e "${GREEN}${NC}" || echo -e "${RED}${NC}"

    echo -n "  Verifying memory count... "
    MEMORY_COUNT=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/lexi_memory" | python -c "import sys, json; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || echo "0")

    if [ "$MEMORY_COUNT" -eq 4 ]; then
        echo -e "${GREEN}✓ (4 entries)${NC}"
    else
        echo -e "${YELLOW}⚠ (${MEMORY_COUNT} entries, expected 4)${NC}"
    fi

    echo ""
else
    echo -e "${YELLOW}Step 2: Skipping database cleanup (--skip-cleanup)${NC}"
    echo ""
fi

# Step 3: Warm Up Ollama Model
if [ "$SKIP_WARMUP" = false ]; then
    echo -e "${YELLOW}Step 3: Warming up Ollama model...${NC}"

    echo -n "  Loading ${MODEL}... "
    WARMUP_RESPONSE=$(curl -s -X POST "${OLLAMA_URL}/api/chat" \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}], \"keep_alive\": \"30m\"}" 2>&1)

    if echo "$WARMUP_RESPONSE" | grep -q "message"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        echo -e "${RED}Error: Failed to load model${NC}"
        echo "$WARMUP_RESPONSE"
        exit 1
    fi

    echo -n "  Verifying model status... "
    MODEL_STATUS=$(curl -s "${OLLAMA_URL}/api/ps" | python -c "import sys, json; data=json.load(sys.stdin); models=[m['name'] for m in data.get('models', [])]; print('✓' if any('${MODEL}' in m for m in models) else '✗')" 2>/dev/null || echo "✗")

    if [ "$MODEL_STATUS" = "✓" ]; then
        echo -e "${GREEN}✓ (loaded)${NC}"
    else
        echo -e "${YELLOW}⚠ (status unknown)${NC}"
    fi

    echo ""
else
    echo -e "${YELLOW}Step 3: Skipping model warmup (--skip-warmup)${NC}"
    echo ""
fi

# Step 4: Run Performance Tests
echo -e "${YELLOW}Step 4: Running performance tests...${NC}"
echo ""
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Test execution starting...${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
echo ""

# Run tests with logging
python tests/performance_test_optimized.py 2>&1 | tee "$LOG_FILE"

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${BLUE}----------------------------------------${NC}"
echo -e "${BLUE}Test execution complete${NC}"
echo -e "${BLUE}----------------------------------------${NC}"
echo ""

# Step 5: Summary
echo -e "${YELLOW}Step 5: Test Summary${NC}"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "  Status: ${GREEN}✓ PASSED${NC}"

    # Extract key metrics from log
    if [ -f "$LOG_FILE" ]; then
        echo -e "  Results saved to: ${LOG_FILE}"

        # Try to extract average time
        AVG_TIME=$(grep "Average Response Time:" "$LOG_FILE" | head -1 | grep -oE "[0-9]+\.[0-9]+s" | sed 's/s//' || echo "N/A")
        if [ "$AVG_TIME" != "N/A" ]; then
            echo -e "  Average Time: ${AVG_TIME}s"

            # Check if target met
            if (( $(echo "$AVG_TIME < 6.0" | bc -l) )); then
                echo -e "  Target: ${GREEN}✓ Phase 2 target met (<6s)${NC}"
            elif (( $(echo "$AVG_TIME < 8.0" | bc -l) )); then
                echo -e "  Target: ${YELLOW}⚠ Phase 1 target met (<8s)${NC}"
            else
                echo -e "  Target: ${RED}✗ Target not met (>8s)${NC}"
            fi
        fi

        # Extract grade
        GRADE=$(grep "Overall Grade:" "$LOG_FILE" | head -1 | grep -oE "(A\+|A|B|C|D)" || echo "N/A")
        if [ "$GRADE" != "N/A" ]; then
            case $GRADE in
                A+|A)
                    echo -e "  Grade: ${GREEN}${GRADE}${NC}"
                    ;;
                B)
                    echo -e "  Grade: ${YELLOW}${GRADE}${NC}"
                    ;;
                *)
                    echo -e "  Grade: ${RED}${GRADE}${NC}"
                    ;;
            esac
        fi
    fi
else
    echo -e "  Status: ${RED}✗ FAILED${NC}"
    echo -e "  Exit code: ${TEST_EXIT_CODE}"
    echo -e "  Check log file for details: ${LOG_FILE}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Performance Test Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Full results: ${LOG_FILE}"
echo ""

exit $TEST_EXIT_CODE
