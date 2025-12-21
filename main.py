from backend.core.bootstrap import initialize_components, ConfigurationError
from backend.core import process_chat_message
from backend.services.heartbeat_memory import start_heartbeat
from langchain_ollama import OllamaEmbeddings
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.config.persistence import ConfigPersistence
import sys
import os
import argparse
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Optional: CORS freischalten
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/config")
def get_config():
    return {
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "embedding_url": os.getenv("EMBEDDING_URL", "http://localhost:11434"),
        "llm_model": os.getenv("LLM_MODEL", "gemma3:4b-it-qat"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": os.getenv("QDRANT_PORT", "6333"),
    }

def main():
    # Argument-Parser für Kommandozeilenoptionen
    parser = argparse.ArgumentParser(description="Lexi KI mit intelligentem Gedächtnissystem")
    parser.add_argument('--no-feedback', action='store_true', help="Deaktiviert das Sammeln von Feedback")
    parser.add_argument('--test', action='store_true', help="Führt den Intelligenztest durch")
    parser.add_argument('--force-recreate', action='store_true', help="Erzwingt die Neuanlage der Vektor-Sammlung")
    args = parser.parse_args()
    
    # Sicherstellen, dass das Verzeichnis für ML-Modelle existiert
    os.makedirs("models", exist_ok=True)
    
    # Laden der persistenten Konfiguration
    print("Lade gespeicherte Konfiguration...")
    saved_config = ConfigPersistence.load_config()
    if saved_config:
        print(f"Gespeicherte Konfiguration gefunden mit {len(saved_config)} Einstellungen")
        ConfigPersistence.apply_config(saved_config)
    else:
        print("Keine gespeicherte Konfiguration gefunden, verwende Standardeinstellungen")
    
    # Initialisierung der Komponenten
    config_warning = None
    embeddings = None
    vectorstore = None
    memory = None
    chat_client = None
    
    try:
        embeddings, vectorstore, memory, chat_client, config_warning = initialize_components(force_recreate=args.force_recreate)
    except ConfigurationError as e:
        error_message = str(e)
        print(f"\nKonfigurationsfehler: {error_message}")
        
        # Check if this is a dimension mismatch error and provide more helpful guidance
        if "dimension" in error_message.lower():
            print("\nDIMENSION MISMATCH DETECTED:")
            print("  1. Starten Sie mit --force-recreate, um die Vektor-Kollektion neu zu erstellen:")
            print("     python main.py --force-recreate")
            print("\n  2. ODER setzen Sie ein anderes Embedding-Modell in .env, z.B.:")
            print("     EMBEDDING_MODEL=nomic-embed-text")
            print("\n  3. ODER konfigurieren Sie die Dimensionen direkt:")
            print("     LEXI_MEMORY_DIMENSION=768  # für nomic-embed-text")
            print("     LEXI_MEMORY_DIMENSION=4096 # für qwen3")
        else:
            print("Bitte überprüfen Sie Ihre Konfiguration und starten Sie erneut.")
        return
    except Exception as e:
        print(f"\nUnerwarteter Fehler bei der Initialisierung: {str(e)}")
        return
        
    # Konfigurationswarnung anzeigen, falls vorhanden
    if config_warning:
        print(f"\nKonfigurationswarnung: {config_warning}")
        print("Sie können die Konfiguration über das Web-Interface anpassen oder mit --force-recreate neu starten.")
    
    
    # Starten der gewichteten Gedächtnisbereinigung (alle 24 Stunden)
    cleanup_thread = start_heartbeat()
    
    # Informationen über den Modus anzeigen
    feedback_status = "deaktiviert" if args.no_feedback else "aktiviert"
    print(f"\n=== Lexi mit intelligentem Gedächtnissystem gestartet ===")
    print(f"- Maschinelles Lernen für Gedächtnisentscheidungen: AKTIV")
    print(f"- LLM-basierte Bewertung für wichtige Informationen: AKTIV")
    print(f"- Benutzerfeedback-Sammlung: {feedback_status}")
    print(f"- ML-Modelle werden in './models/' gespeichert")
    print("=========================================================")

    # Hauptloop
    while True:
        try:
            user_input = input("\nBitte gib deine Nachricht ein (oder 'exit' zum Beenden): ")
            if user_input.lower() == 'exit':
                break

            # Verarbeitung der Chat-Nachricht
            response = process_chat_message(
                user_input, 
                chat_client, 
                vectorstore, 
                memory, 
                embeddings, 
                collect_feedback=not args.no_feedback
            )
            
            print("\nLexi:")
            print(response)

        except KeyboardInterrupt:
            print("\nProgramm wird beendet...")
            break
        except Exception as e:
            print(f"\nFehler bei der Verarbeitung der Nachricht: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramm wird beendet...")
    except Exception as e:
        print(f"\nUnerwarteter Fehler: {str(e)}", file=sys.stderr)
