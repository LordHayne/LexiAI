#!/bin/bash
# LexiAI Authentication & Profile Learning Setup Script

set -e  # Exit on error

echo "🚀 LexiAI Authentication Setup"
echo "================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "start_middleware.py" ]; then
    echo "❌ Error: Must run from LexiAI root directory"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install PyJWT==2.8.0 bcrypt==4.1.2 --quiet

if [ $? -eq 0 ]; then
    echo "   ✅ PyJWT and bcrypt installed successfully"
else
    echo "   ❌ Failed to install dependencies"
    exit 1
fi

# Generate JWT secret if not exists
echo ""
echo "🔐 Setting up JWT secret..."

if [ -z "$LEXI_JWT_SECRET" ]; then
    JWT_SECRET=$(openssl rand -hex 32)
    echo "   Generated new JWT secret"

    # Add to .env file
    if [ ! -f ".env" ]; then
        touch .env
    fi

    # Check if already in .env
    if grep -q "LEXI_JWT_SECRET" .env 2>/dev/null; then
        echo "   ⚠️  LEXI_JWT_SECRET already in .env - not overwriting"
    else
        echo "LEXI_JWT_SECRET=$JWT_SECRET" >> .env
        echo "   ✅ Added LEXI_JWT_SECRET to .env"
    fi

    # Export for current session
    export LEXI_JWT_SECRET=$JWT_SECRET
else
    echo "   ✅ LEXI_JWT_SECRET already set in environment"
fi

# Create logs directory if not exists
echo ""
echo "📁 Setting up directories..."
mkdir -p logs
echo "   ✅ Logs directory ready"

# Test import
echo ""
echo "🧪 Testing imports..."
python3 << EOF
try:
    import jwt
    import bcrypt
    from backend.api.v1.routes.auth import router
    from backend.services.profile_builder import ProfileBuilder
    from backend.services.profile_context import ProfileContextBuilder
    print("   ✅ All imports successful")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "   ❌ Import test failed"
    exit 1
fi

# Display summary
echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "📋 Summary:"
echo "   • PyJWT and bcrypt installed"
echo "   • JWT secret generated and saved to .env"
echo "   • Authentication routes registered"
echo "   • Profile learning integrated"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Start the server:"
echo "   python start_middleware.py"
echo ""
echo "2. Test registration:"
echo "   curl -X POST http://localhost:8000/v1/auth/register \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"email\":\"test@example.com\",\"password\":\"Test1234\"}'"
echo ""
echo "3. View API docs:"
echo "   Open http://localhost:8000/docs in browser"
echo ""
echo "📚 Documentation:"
echo "   • Full docs: docs/AUTHENTICATION_AND_PROFILE_LEARNING.md"
echo "   • Installation: docs/INSTALLATION_AUTH.md"
echo ""
echo "🔒 Security Notes:"
echo "   • JWT secret is in .env (DO NOT commit to git!)"
echo "   • Add .env to .gitignore"
echo "   • Use HTTPS in production"
echo ""
