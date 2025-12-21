"""
Environment variable loader.
Loads .env file if present - MUST be called before any config imports!
"""
import os
from pathlib import Path


def load_env_file():
    """
    Load .env file from project root.
    This should be called BEFORE any config modules are imported.
    """
    # Find project root (.env should be there)
    current = Path(__file__).resolve()
    while current.parent != current:
        env_path = current / '.env'
        if env_path.exists():
            _load_dotenv(env_path)
            return
        current = current.parent

    # If not found, try current working directory
    env_path = Path.cwd() / '.env'
    if env_path.exists():
        _load_dotenv(env_path)


def _load_dotenv(env_path: Path):
    """Load environment variables from .env file."""
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Only set if not already in environment
                os.environ.setdefault(key, value)


# Auto-load on import (but only if not already loaded)
if not os.environ.get('_LEXI_ENV_LOADED'):
    load_env_file()
    os.environ['_LEXI_ENV_LOADED'] = 'true'
