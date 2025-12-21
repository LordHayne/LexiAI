#!/usr/bin/env python3
"""
Startup script for the Lexi Middleware API
"""
import argparse
import os
import sys
import subprocess
import time
import webbrowser
import logging
import shutil
import urllib.request
import urllib.error

# Load .env file FIRST before any other imports
from pathlib import Path
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

from backend.utils.version import get_version



def setup_logging():
    """Setup basic logging for the startup script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger("lexi_middleware_startup")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Lexi Middleware for OpenWebUI")
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to (default: 8000)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with auto-reload"
    )
    
    parser.add_argument(
        "--no-browser", 
        action="store_true", 
        help="Don't open browser automatically"
    )
    
    parser.add_argument(
        "--api-key", 
        help="Set a custom API key for authentication"
    )
    
    parser.add_argument(
        "--ollama-url",
        help="URL for the Ollama LLM instance (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--embedding-url",
        help="URL for the Ollama embedding instance (default: same as --ollama-url)"
    )
    
    parser.add_argument(
        "--qdrant-host",
        help="Host for the Qdrant vector database (default: localhost)"
    )
    
    parser.add_argument(
        "--qdrant-port",
        type=int,
        help="Port for the Qdrant vector database (default: 6333)"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of the vector collection"
    )

    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only start the API server (skip Qdrant/Ollama/XTTS)"
    )

    parser.add_argument(
        "--dev-remote",
        action="store_true",
        help="Dev mode with remote Qdrant/Ollama (skip starting local services)"
    )

    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip starting Qdrant"
    )

    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip starting Ollama"
    )

    parser.add_argument(
        "--skip-xtts",
        action="store_true",
        help="Skip starting XTTS"
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        return True
    except ImportError as e:
        return False

def check_service(url, timeout_seconds=2):
    """Return True if an HTTP service responds."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds):
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return False

def run_command(cmd, logger, check=True, background=False):
    """Run a subprocess command."""
    if background:
        logger.info("Starting: %s", " ".join(cmd))
        return subprocess.Popen(cmd)
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check)

def ensure_docker(logger):
    """Ensure docker is installed and running."""
    if not shutil.which("docker"):
        logger.error("Docker not found. Please install Docker.")
        return False
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logger.error("Docker does not appear to be running.")
        return False

def start_qdrant(logger):
    """Start Qdrant via Docker if needed."""
    if check_service("http://localhost:6333"):
        logger.info("Qdrant already running.")
        return True

    if not ensure_docker(logger):
        return False

    storage_dir = os.path.join(os.getcwd(), "qdrant_storage")
    os.makedirs(storage_dir, exist_ok=True)

    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        container_names = result.stdout.splitlines()
        if "qdrant_lexiai" in container_names:
            run_command(["docker", "start", "qdrant_lexiai"], logger)
        else:
            run_command([
                "docker", "run", "-d",
                "--name", "qdrant_lexiai",
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", f"{storage_dir}:/qdrant/storage",
                "qdrant/qdrant"
            ], logger)
    except subprocess.CalledProcessError:
        logger.error("Failed to start Qdrant container.")
        return False

    for _ in range(30):
        if check_service("http://localhost:6333"):
            logger.info("Qdrant is ready.")
            return True
        time.sleep(1)

    logger.error("Qdrant did not become ready in time.")
    return False

def start_ollama(logger):
    """Start Ollama serve if needed."""
    if check_service("http://localhost:11434/api/tags"):
        logger.info("Ollama already running.")
        return None, True

    if not shutil.which("ollama"):
        logger.error("Ollama not found. Please install Ollama.")
        return None, False

    process = run_command(["ollama", "serve"], logger, background=True)
    for _ in range(30):
        if check_service("http://localhost:11434/api/tags"):
            logger.info("Ollama is ready.")
            return process, True
        time.sleep(1)

    logger.error("Ollama did not become ready in time.")
    return process, False

def detect_docker_compose():
    """Return docker compose command list."""
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    if shutil.which("docker"):
        return ["docker", "compose"]
    return None

def start_xtts(logger):
    """Start XTTS via docker compose if needed."""
    if check_service("http://localhost:8020/health"):
        logger.info("XTTS already running.")
        return True

    if not ensure_docker(logger):
        return False

    compose_cmd = detect_docker_compose()
    if not compose_cmd:
        logger.error("docker-compose not found.")
        return False

    compose_file = os.path.join(os.getcwd(), "docker-compose-xtts.yml")
    if not os.path.isfile(compose_file):
        logger.error("Missing docker-compose-xtts.yml")
        return False

    run_command(compose_cmd + ["-f", compose_file, "up", "-d"], logger)

    for _ in range(30):
        if check_service("http://localhost:8020/health"):
            logger.info("XTTS is ready.")
            return True
        time.sleep(2)

    logger.error("XTTS did not become ready in time.")
    return False

def main():
    """Main entry point for starting the middleware"""
    logger = setup_logging()
    args = parse_arguments()
    
    # Show banner
    print("\n" + "=" * 60)
    print(f"Lexi Middleware for OpenWebUI v{get_version()}")
    print("=" * 60 + "\n")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("Missing dependencies. Please run: pip install -r requirements.txt")
        return 1
    
    # Handle API key
    if args.api_key:
        os.environ["LEXI_API_KEY"] = args.api_key
        logger.info(f"Using custom API key: {args.api_key[:3]}{'*' * (len(args.api_key) - 6)}{args.api_key[-3:]}")
    else:
        # Check if default API key is being used
        from backend.config.auth_config import SecurityConfig
        if SecurityConfig.API_KEY == "dev_api_key_change_me_in_production":
            logger.warning("Using default API key. For production, set a custom key with --api-key")
    
    # Set Ollama URL if provided
    if args.ollama_url:
        os.environ["LEXI_OLLAMA_URL"] = args.ollama_url
        logger.info(f"Using custom Ollama LLM URL: {args.ollama_url}")
    
    # Set Embedding URL if provided, otherwise use Ollama URL
    if args.embedding_url:
        os.environ["LEXI_EMBEDDING_URL"] = args.embedding_url
        logger.info(f"Using custom Embedding URL: {args.embedding_url}")
    elif args.ollama_url:
        # If only ollama-url is provided, use it for embeddings too
        os.environ["LEXI_EMBEDDING_URL"] = args.ollama_url
        logger.info(f"Using Ollama LLM URL for embeddings: {args.ollama_url}")
    
    # Set Qdrant host if provided
    if args.qdrant_host:
        os.environ["LEXI_QDRANT_HOST"] = args.qdrant_host
        logger.info(f"Using custom Qdrant host: {args.qdrant_host}")
    
    # Set Qdrant port if provided
    if args.qdrant_port:
        os.environ["LEXI_QDRANT_PORT"] = str(args.qdrant_port)
        logger.info(f"Using custom Qdrant port: {args.qdrant_port}")
    
    # Set force recreate flag if provided
    if args.force_recreate:
        os.environ["LEXI_FORCE_RECREATE"] = "True"
        logger.info("Force recreation of vector collection enabled")
    
    # Start dependent services unless skipped
    start_qdrant_service = not args.api_only and not args.skip_qdrant and not args.dev_remote
    start_ollama_service = not args.api_only and not args.skip_ollama and not args.dev_remote
    start_xtts_service = not args.api_only and not args.skip_xtts and not args.dev_remote

    if args.dev_remote:
        if not (args.qdrant_host or os.environ.get("LEXI_QDRANT_HOST")):
            logger.warning("Dev remote mode: no Qdrant host provided (use --qdrant-host or LEXI_QDRANT_HOST).")
        if not (args.ollama_url or os.environ.get("LEXI_OLLAMA_URL")):
            logger.warning("Dev remote mode: no Ollama URL provided (use --ollama-url or LEXI_OLLAMA_URL).")

    ollama_process = None
    if start_qdrant_service:
        if not start_qdrant(logger):
            return 1

    if start_ollama_service:
        ollama_process, ollama_ready = start_ollama(logger)
        if not ollama_ready:
            if ollama_process and ollama_process.poll() is None:
                ollama_process.terminate()
            return 1

    if start_xtts_service:
        if not start_xtts(logger):
            return 1

    # Starting server
    logger.info(f"Starting Lexi Middleware on http://{args.host}:{args.port}")
    
    # Start uvicorn
    reload_arg = ["--reload"] if args.debug else []
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.api.api_server:app", 
        "--host", args.host, 
        "--port", str(args.port)
    ] + reload_arg
    
    # Open browser if not disabled
    if not args.no_browser:
        # Wait a moment before opening browser
        def open_browser():
            time.sleep(3)  # Give the server a moment to start
            url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
            logger.info(f"Opening browser at {url}")
            webbrowser.open(url)
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    # Execute the uvicorn command
    try:
        subprocess.run(cmd, check=True)
        return 0
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return 1
    finally:
        if ollama_process and ollama_process.poll() is None:
            ollama_process.terminate()

if __name__ == "__main__":
    sys.exit(main())
