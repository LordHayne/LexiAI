from langchain_ollama import OllamaEmbeddings
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import requests

# Simple HTTP check
def check_http(name, url):
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print(f"[OK] {name} HTTP erreichbar unter {url}")
        else:
            print(f"[WARNUNG] {name} antwortet, aber mit Statuscode {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[FEHLER] {name} nicht erreichbar unter {url}: {e}")

# Teste Embedding-Modell direkt
def test_embedding_model():
    try:
        embeddings = OllamaEmbeddings(
            base_url="http://192.168.1.2:11434",
            model="nomic-embed-text:latest"
        )
        vector = embeddings.embed_query("Testverbindung Lexi")
        if isinstance(vector, list) and len(vector) > 0:
            print("[OK] Embedding-Modell funktioniert und liefert Vektoren")
        else:
            print("[FEHLER] Embedding-Modell gibt keine gültigen Vektoren zurück")
    except Exception as e:
        print(f"[FEHLER] Embedding-Modell konnte nicht genutzt werden: {e}")

# Teste Qdrant mit Vektor
def test_qdrant_with_embedding():
    try:
        embeddings = OllamaEmbeddings(
            base_url="http://192.168.1.2:11434",
            model="nomic-embed-text:latest"
        )
        client = QdrantClient(host="192.168.1.2", port=6333)

        # Erstelle temporäre Test-Collection (falls noch nicht vorhanden)
        test_collection = "lexi_health_test"
        if test_collection not in [c.name for c in client.get_collections().collections]:
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        vectorstore = QdrantMemoryInterface(
            client=client,
            collection_name=test_collection,
            embedding=embeddings
        )

        # Einfügen und Suchen testen
        vectorstore.add_texts(["Lexi Testeintrag"])
        result = vectorstore.similarity_search("Lexi Testeintrag", k=1)
        if result:
            print("[OK] Qdrant ist erreichbar und verarbeitet Embeddings")
        else:
            print("[FEHLER] Qdrant verarbeitet Embeddings nicht korrekt")
    except Exception as e:
        print(f"[FEHLER] Verbindung zu Qdrant oder Embedding fehlgeschlagen: {e}")

if __name__ == "__main__":
    check_http("Ollama (Chat)", "http://192.168.1.146:11434")
    check_http("Ollama (Embedding)", "http://192.168.1.2:11434")
    check_http("Qdrant", "http://192.168.1.2:6333")

    print("\n--- Funktionstest Embedding-Modell ---")
    test_embedding_model()

    print("\n--- Funktionstest Qdrant + Embedding ---")
    test_qdrant_with_embedding()
