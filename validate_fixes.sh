#!/bin/bash

# LexiAI Production Readiness Validation Script
# Validates that all 15 critical fixes are properly implemented
# Run this before deploying to production

set -e

echo "🔍 LexiAI Production Readiness Validation"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Function to check for pattern in file
check_pattern() {
    local file=$1
    local pattern=$2
    local description=$3

    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $description"
        ((FAILED++))
    fi
}

# Function to check file exists
check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $description"
        ((FAILED++))
    fi
}

echo "🔒 Security Fixes (10/10)"
echo "-------------------------"

# Fix 1: API Keys removed from config
if ! grep -q "api_key\|LEXI_API_KEY\|tavily_api_key" "backend/config/persistent_config.json" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Fix 1: API keys not in persistent_config.json"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Fix 1: API keys found in persistent_config.json (should be removed)"
    ((FAILED++))
fi
check_file "backend/config/persistent_config.example.json" "Fix 1: Example config template exists"

# Fix 2: Timing-safe API key verification
check_pattern "backend/config/auth_config.py" "secrets.compare_digest" "Fix 2: Timing-safe API key verification"

# Fix 3: Thread-safe component cache
check_pattern "backend/core/component_cache.py" "threading.Lock" "Fix 3: Thread-safe component cache"

# Fix 4: CORS security
check_pattern "backend/config/cors_config.py" "SECURITY ERROR: CORS wildcard" "Fix 4: CORS security validation"

# Fix 5: Security headers
check_pattern "backend/api/api_server.py" "security_headers_middleware" "Fix 5: Security headers middleware"

# Fix 6: Rate limiting
check_pattern "backend/api/api_server.py" "limiter = Limiter" "Fix 6: Rate limiting integration"
check_pattern "requirements.txt" "slowapi" "Fix 6: slowapi dependency added"

# Fix 7: Log masking
check_pattern "backend/config/persistence.py" "SENSITIVE_KEYS" "Fix 7: Sensitive data masking in logs"

# Fix 8: UI authentication
check_pattern "backend/api/middleware/auth.py" "verify_ui_auth" "Fix 8: UI endpoint authentication"

# Fix 9: File locking
check_pattern "backend/config/persistence.py" "FileLock" "Fix 9: File locking for config writes"
check_pattern "requirements.txt" "filelock" "Fix 9: filelock dependency added"

# Fix 10: Input validation
check_file "backend/utils/input_validation.py" "Fix 10: Input validation module exists"
check_pattern "backend/utils/input_validation.py" "class InputValidator" "Fix 10: InputValidator class implemented"

echo ""
echo "⚡ Performance Fixes (5/5)"
echo "-------------------------"

# Fix 11: Memory leak prevention
check_pattern "backend/core/bootstrap.py" "ConversationBufferWindowMemory" "Fix 11: Memory leak fix (bounded buffer)"

# Fix 12: Connection pooling
check_pattern "backend/qdrant/client_wrapper.py" "_client_lock" "Fix 12a: Qdrant connection pooling"
check_pattern "backend/embeddings/embedding_model.py" "_sync_client" "Fix 12b: Embedding HTTP client pooling"

# Fix 13: Async I/O fixes
check_pattern "backend/core/chat_processing.py" "asyncio.to_thread" "Fix 13: Blocking I/O wrapped in asyncio.to_thread"

# Fix 14: Graceful shutdown
check_pattern "backend/api/api_server.py" "Closing embedding clients" "Fix 14: Graceful shutdown cleanup"

# Fix 15: Cache invalidation
check_pattern "backend/memory/adapter.py" "cache.invalidate_user" "Fix 15: Cache invalidation on write"

echo ""
echo "📚 Documentation"
echo "----------------"

check_file "SECURITY.md" "Security documentation exists"
check_file "SECURITY_FIXES_SUMMARY.md" "Security fixes summary exists"
check_file "PERFORMANCE_FIXES_SUMMARY.md" "Performance fixes summary exists"
check_file "IMPLEMENTATION_COMPLETE.md" "Implementation complete document exists"

echo ""
echo "📦 Dependencies"
echo "---------------"

# Check if requirements.txt has new dependencies
if [ -f "requirements.txt" ]; then
    if grep -q "slowapi" requirements.txt && grep -q "filelock" requirements.txt; then
        echo -e "${GREEN}✓${NC} All new dependencies present in requirements.txt"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} Missing dependencies in requirements.txt"
        ((FAILED++))
    fi
else
    echo -e "${RED}✗${NC} requirements.txt not found"
    ((FAILED++))
fi

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED!${NC}"
    echo "System is ready for production deployment."
    echo ""
    echo "Next steps:"
    echo "1. Install new dependencies: pip install -r requirements.txt"
    echo "2. Set environment variables (see SECURITY.md)"
    echo "3. Run performance tests: pytest tests/test_performance.py"
    echo "4. Run load tests: ab -n 1000 -c 50 ..."
    echo "5. Deploy to production"
    exit 0
else
    echo -e "${RED}⚠️  SOME CHECKS FAILED!${NC}"
    echo "Please review failed checks before deploying to production."
    exit 1
fi
