"""
Utilities for extracting content from LLM responses.
"""
import json
import logging

# Setup logging
logger = logging.getLogger("lexi_middleware.utils.content_extraction")

def extract_content_from_response(response_text):
    """
    Extrahiert den tatsächlichen Content aus einer LLM-Antwort.
    Behandelt den Fall, dass die Antwort ein JSON-String mit einem 'content'-Feld ist.
    
    Args:
        response_text (str): Die ursprüngliche Antwort vom LLM
        
    Returns:
        str: Der extrahierte Content oder der ursprüngliche Text, wenn keine Extraktion nötig ist
    """
    if not response_text:
        return ""
    
    # Überprüfe, ob der Text ein JSON-String ist
    try:
        # Versuche, den Text als JSON zu parsen
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            json_data = json.loads(response_text)
            
            # Überprüfe, ob ein 'content'-Feld vorhanden ist
            if isinstance(json_data, dict) and 'content' in json_data:
                logger.info("JSON Objekt mit 'content'-Feld erkannt - extrahiere Inhalt")
                return json_data['content']
    except json.JSONDecodeError:
        # Wenn es kein gültiges JSON ist, verwende den ursprünglichen Text
        logger.debug("Antwort ist kein gültiges JSON, verwende den ursprünglichen Text")
    except Exception as e:
        # Bei anderen Fehlern, logge und verwende den ursprünglichen Text
        logger.warning(f"Fehler beim Extrahieren des Contents: {str(e)}")
    
    # Wenn keine Extraktion erfolgt ist, gib den ursprünglichen Text zurück
    return response_text
