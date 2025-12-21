import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'persistent_config.json')

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Fehler beim Laden der Konfiguration: {e}")
        return {}

CONFIG = load_config()

def get_config_value(key, default=None):
    return CONFIG.get(key, default)
