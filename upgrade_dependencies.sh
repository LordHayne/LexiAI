#!/bin/bash
# LexiAI Dependency Upgrade Script
# This script upgrades all dependencies to the latest compatible versions

set -e  # Exit on error

echo "🔄 LexiAI Dependency Upgrade Script"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found (.venv)"
    echo "Please create one first: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Backup current requirements
if [ -f "requirements.txt" ]; then
    echo "💾 Backing up current requirements..."
    cp requirements.txt requirements.txt.backup.$(date +%Y%m%d_%H%M%S)
fi

# Upgrade pip itself
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Uninstall old langchain if present
echo "🧹 Cleaning old dependencies..."
pip uninstall -y langchain-community langchain-core langchain-ollama langchain-qdrant langchain 2>/dev/null || true

# Install core dependencies first
echo "📥 Installing core dependencies..."
pip install --upgrade \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0

# Install LangChain ecosystem
echo "📥 Installing LangChain ecosystem..."
pip install --upgrade \
    langchain>=0.1.0 \
    langchain-core>=0.1.0 \
    langchain-community>=0.0.10 \
    langchain-ollama>=0.1.0 \
    langchain-qdrant>=0.1.0

# Install vector DB and ML libraries
echo "📥 Installing vector DB and ML libraries..."
pip install --upgrade \
    qdrant-client>=1.7.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0

# Install remaining dependencies
echo "📥 Installing remaining dependencies..."
pip install --upgrade -r requirements.txt

# Verify critical imports
echo ""
echo "✅ Verifying installations..."
python3 << 'EOF'
import sys

def check_import(module, name=None):
    try:
        __import__(module)
        print(f"✅ {name or module}")
        return True
    except ImportError as e:
        print(f"❌ {name or module}: {e}")
        return False

all_ok = True
all_ok &= check_import("fastapi", "FastAPI")
all_ok &= check_import("pydantic", "Pydantic")
all_ok &= check_import("langchain", "LangChain")
all_ok &= check_import("langchain_ollama", "LangChain-Ollama")
all_ok &= check_import("qdrant_client", "Qdrant Client")
all_ok &= check_import("sklearn", "scikit-learn")
all_ok &= check_import("numpy", "NumPy")
all_ok &= check_import("backoff", "Backoff")

if not all_ok:
    print("\n⚠️  Some imports failed. Check the errors above.")
    sys.exit(1)
else:
    print("\n✅ All critical dependencies installed successfully!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Dependency upgrade completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review MIGRATION.md for breaking changes"
    echo "2. Run tests: pytest tests/"
    echo "3. Start the server: python start_middleware.py"
else
    echo ""
    echo "❌ Some dependencies failed to install or import."
    echo "Please check the errors above and fix manually."
    exit 1
fi
