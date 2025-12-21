#!/bin/bash

# Test Memory Management Operations

echo "🧪 Testing Memory Management Functions"
echo "========================================"
echo ""

# 1. Check Heartbeat Status
echo "1️⃣ Checking Heartbeat Status..."
curl -s http://localhost:8000/v1/debug/heartbeat/status | python3 -m json.tool | head -30
echo ""
echo ""

# 2. Test Consolidation Mode
echo "2️⃣ Testing CONSOLIDATION mode..."
curl -s -X POST http://localhost:8000/v1/debug/heartbeat/ui-trigger \
  -H 'Content-Type: application/json' \
  -d '{"mode": "consolidation"}' | python3 -m json.tool
echo ""
echo ""

# Wait 65 seconds for cooldown
echo "⏳ Waiting 65 seconds for rate limit cooldown..."
sleep 65

# 3. Test Synthesis Mode
echo "3️⃣ Testing SYNTHESIS mode..."
curl -s -X POST http://localhost:8000/v1/debug/heartbeat/ui-trigger \
  -H 'Content-Type: application/json' \
  -d '{"mode": "synthesis"}' | python3 -m json.tool
echo ""
echo ""

# Wait 65 seconds for cooldown
echo "⏳ Waiting 65 seconds for rate limit cooldown..."
sleep 65

# 4. Test Cleanup Mode
echo "4️⃣ Testing CLEANUP mode..."
curl -s -X POST http://localhost:8000/v1/debug/heartbeat/ui-trigger \
  -H 'Content-Type: application/json' \
  -d '{"mode": "cleanup"}' | python3 -m json.tool
echo ""
echo ""

echo "✅ All tests completed!"
