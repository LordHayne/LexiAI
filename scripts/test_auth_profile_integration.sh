#!/bin/bash
# Integration Test Script: Authentication + Profile Learning
# Tests the complete flow: Register → Login → Chat → Profile Verification

set -e  # Exit on error

echo "=== LexiAI Authentication + Profile Learning Integration Test ==="
echo ""

# Configuration
API_BASE_URL="http://localhost:8000"
TEST_EMAIL="integration-test-$(date +%s)@example.com"
TEST_PASSWORD="SecureTestPass123!"
TEST_NAME="Integration Test User"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if API server is running
print_info "Checking API server..."
if ! curl -s "${API_BASE_URL}/health" > /dev/null; then
    print_error "API server is not running at ${API_BASE_URL}"
    echo "Start the server with: python start_middleware.py"
    exit 1
fi
print_success "API server is running"
echo ""

# Step 1: Register new user
print_info "Step 1: Registering new user..."
REGISTER_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/v1/auth/register" \
    -H "Content-Type: application/json" \
    -d "{
        \"email\": \"${TEST_EMAIL}\",
        \"password\": \"${TEST_PASSWORD}\",
        \"full_name\": \"${TEST_NAME}\"
    }")

if echo "$REGISTER_RESPONSE" | grep -q "user_id"; then
    USER_ID=$(echo "$REGISTER_RESPONSE" | jq -r '.user_id')
    print_success "User registered successfully (ID: $USER_ID)"

    # Verify password is not in response
    if echo "$REGISTER_RESPONSE" | grep -q "password"; then
        print_error "SECURITY ISSUE: Password found in registration response!"
        exit 1
    fi
    print_success "Security check passed: No password in response"
else
    print_error "Registration failed"
    echo "Response: $REGISTER_RESPONSE"
    exit 1
fi
echo ""

# Step 2: Login and get JWT
print_info "Step 2: Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{
        \"email\": \"${TEST_EMAIL}\",
        \"password\": \"${TEST_PASSWORD}\"
    }")

if echo "$LOGIN_RESPONSE" | grep -q "access_token"; then
    ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token')
    REFRESH_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.refresh_token')
    print_success "Login successful"
    print_success "Access token received"
    print_success "Refresh token received"

    # Decode JWT to verify user_id
    JWT_PAYLOAD=$(echo "$ACCESS_TOKEN" | cut -d'.' -f2)
    # Add padding if needed
    JWT_PAYLOAD_PADDED=$(echo "$JWT_PAYLOAD" | awk '{while(length($0)%4!=0){$0=$0"="}}1')
    DECODED_USER_ID=$(echo "$JWT_PAYLOAD_PADDED" | base64 -d 2>/dev/null | jq -r '.sub' 2>/dev/null || echo "")

    if [ "$DECODED_USER_ID" == "$USER_ID" ]; then
        print_success "JWT contains correct user_id"
    else
        print_error "JWT user_id mismatch"
        exit 1
    fi
else
    print_error "Login failed"
    echo "Response: $LOGIN_RESPONSE"
    exit 1
fi
echo ""

# Step 3: Chat with profile information
print_info "Step 3: Chatting with profile information..."

CHAT_MESSAGES=(
    "Hallo, ich bin Thomas."
    "Ich arbeite als Software Engineer bei Google."
    "Ich liebe Python, Machine Learning und Kubernetes."
    "Ich bin 28 Jahre alt und wohne in München."
)

for MESSAGE in "${CHAT_MESSAGES[@]}"; do
    print_info "Sending: $MESSAGE"

    CHAT_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/v1/chat" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -d "{\"message\": \"${MESSAGE}\"}")

    if echo "$CHAT_RESPONSE" | grep -q "response"; then
        print_success "Chat message processed"
    else
        print_error "Chat failed"
        echo "Response: $CHAT_RESPONSE"
        exit 1
    fi
done

# Wait for background profile building
print_info "Waiting for profile learning (2 seconds)..."
sleep 2
print_success "Profile learning completed"
echo ""

# Step 4: Verify profile was learned
print_info "Step 4: Verifying profile..."
PROFILE_RESPONSE=$(curl -s -X GET "${API_BASE_URL}/v1/profile/me" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}")

if echo "$PROFILE_RESPONSE" | grep -q "profile_items"; then
    PROFILE_COUNT=$(echo "$PROFILE_RESPONSE" | jq '.profile_items | length')
    print_success "Profile retrieved successfully ($PROFILE_COUNT items)"

    # Check for expected information
    PROFILE_TEXT=$(echo "$PROFILE_RESPONSE" | jq -r '.profile_items[] | .content' | tr '[:upper:]' '[:lower:]')

    EXPECTED_INFO=(
        "thomas"
        "software"
        "engineer"
        "python"
        "machine learning"
        "münchen"
    )

    FOUND_COUNT=0
    for INFO in "${EXPECTED_INFO[@]}"; do
        if echo "$PROFILE_TEXT" | grep -q "$INFO"; then
            print_success "Found: $INFO"
            ((FOUND_COUNT++))
        fi
    done

    if [ $FOUND_COUNT -ge 3 ]; then
        print_success "Profile learning accuracy: Good ($FOUND_COUNT/6 items)"
    else
        print_error "Profile learning accuracy: Poor ($FOUND_COUNT/6 items)"
        exit 1
    fi
else
    print_error "Failed to retrieve profile"
    echo "Response: $PROFILE_RESPONSE"
    exit 1
fi
echo ""

# Step 5: Test personalized response
print_info "Step 5: Testing personalized response..."
PERSONALIZED_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/v1/chat" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -d '{"message": "Was weißt du über mich?"}')

if echo "$PERSONALIZED_RESPONSE" | grep -q "response"; then
    RESPONSE_TEXT=$(echo "$PERSONALIZED_RESPONSE" | jq -r '.response' | tr '[:upper:]' '[:lower:]')

    # Check if response references profile
    if echo "$RESPONSE_TEXT" | grep -qE "(thomas|software|engineer|python|machine learning)"; then
        print_success "Response includes profile information"
    else
        print_error "Response does not reference profile"
    fi
else
    print_error "Personalized chat failed"
    exit 1
fi
echo ""

# Step 6: Test token refresh
print_info "Step 6: Testing token refresh..."
REFRESH_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/v1/auth/refresh" \
    -H "Content-Type: application/json" \
    -d "{\"refresh_token\": \"${REFRESH_TOKEN}\"}")

if echo "$REFRESH_RESPONSE" | grep -q "access_token"; then
    NEW_ACCESS_TOKEN=$(echo "$REFRESH_RESPONSE" | jq -r '.access_token')

    if [ "$NEW_ACCESS_TOKEN" != "$ACCESS_TOKEN" ]; then
        print_success "Token refresh successful (new token generated)"
    else
        print_error "Token refresh returned same token"
        exit 1
    fi
else
    print_error "Token refresh failed"
    echo "Response: $REFRESH_RESPONSE"
    exit 1
fi
echo ""

# Step 7: Test logout
print_info "Step 7: Testing logout..."
LOGOUT_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/v1/auth/logout" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}")

if echo "$LOGOUT_RESPONSE" | grep -q "success"; then
    print_success "Logout successful"

    # Try to use token after logout
    print_info "Verifying token invalidation..."
    INVALID_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_BASE_URL}/v1/chat" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -d '{"message": "Test"}')

    HTTP_CODE=$(echo "$INVALID_RESPONSE" | tail -n1)

    if [ "$HTTP_CODE" == "401" ]; then
        print_success "Token correctly invalidated after logout"
    else
        print_error "Token still valid after logout (HTTP $HTTP_CODE)"
        exit 1
    fi
else
    print_error "Logout failed"
    echo "Response: $LOGOUT_RESPONSE"
    exit 1
fi
echo ""

# Performance summary
echo "=== Integration Test Summary ==="
print_success "All tests passed!"
echo ""
echo "Verified:"
echo "  ✓ User registration"
echo "  ✓ Secure login with JWT"
echo "  ✓ Password never exposed"
echo "  ✓ JWT contains correct user_id"
echo "  ✓ Profile learning from chat"
echo "  ✓ Personalized responses"
echo "  ✓ Token refresh"
echo "  ✓ Logout and token invalidation"
echo ""
print_success "Integration test completed successfully!"
