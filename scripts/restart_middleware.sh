#!/bin/bash
# Restart script for Lexi Middleware

echo "=== Restarting Lexi Middleware with OpenWebUI compatibility ==="
echo "    Configuration will now persist between restarts"

# Find and stop any running middleware processes
echo "Stopping any running middleware processes..."
pkill -f "uvicorn api_server:app" || echo "No running middleware found"
sleep 2  # Give time for processes to stop

# Make sure the config directory exists
mkdir -p config

# Check if config persistence file exists and show status
if [ -f "config/persistent_config.json" ]; then
    echo "Found existing persistent configuration"
    echo "Configuration will be loaded from: config/persistent_config.json"
else
    echo "No persistent configuration found. A new one will be created when you update settings."
fi

# Start the middleware with the updated compatibility features
echo "Starting middleware with OpenWebUI compatibility and configuration persistence..."
python3 start_middleware.py

# Exit with the same code as the middleware process
exit $?
