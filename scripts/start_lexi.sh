#!/bin/bash
# Script to start Lexi AI with proper virtual environment activation

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Banner
echo -e "\n============================================================"
echo "Lexi AI with Intelligent Memory System"
echo "============================================================"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo -e "\nActivating virtual environment..."
    
    # Activate the virtual environment (works for bash/zsh)
    source .venv/bin/activate
    
    # Check if activation was successful
    if [ $? -ne 0 ]; then
        echo "Failed to activate virtual environment. Please check .venv directory."
        exit 1
    fi
    
    echo "Virtual environment activated successfully."
    
    # Pass all arguments to the Python script
    echo -e "\nStarting Lexi AI...\n"
    python start_lexi.py "$@"
    
    # Store the exit code
    EXIT_CODE=$?
    
    # Deactivate the virtual environment
    deactivate
    
    # Exit with the same code as the Python script
    exit $EXIT_CODE
else
    echo -e "\nERROR: Virtual environment not found in .venv directory."
    echo "Please create a virtual environment and install required dependencies:"
    echo -e "\nmkdir -p .venv"
    echo "python -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    echo -e "\nThen try running this script again."
    exit 1
fi
