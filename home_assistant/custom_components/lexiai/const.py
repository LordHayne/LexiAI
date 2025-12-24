"""Constants for the LexiAI Home Assistant integration."""

DOMAIN = "lexiai"

CONF_BASE_URL = "base_url"
CONF_API_KEY = "api_key"
CONF_DOMAINS = "domains"
CONF_BATCH_INTERVAL = "batch_interval"
CONF_BATCH_SIZE = "batch_size"
CONF_TIMEOUT = "timeout"

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_DOMAINS = [
    "light",
    "switch",
    "climate",
    "cover",
    "media_player",
    "lock",
    "binary_sensor",
    "sensor",
]
DEFAULT_BATCH_INTERVAL = 0.5
DEFAULT_BATCH_SIZE = 50
DEFAULT_TIMEOUT = 10.0

DOMAIN_OPTIONS = {domain: domain for domain in DEFAULT_DOMAINS}
