#!/usr/bin/env python3
"""
Qdrant Hard Reset
Löscht alle Kollektionen und erstellt lexi_memory neu
"""

import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.config.middleware_config import MiddlewareConfig

def hard_reset_qdrant():
    """Setzt Qdrant komplett zurück"""

    config = MiddlewareConfig()
    qdrant_host = config.get_qdrant_host()
    qdrant_port = config.get_qdrant_port()
    memory_collection = config.get_memory_collection()
    memory_dimension = config.get_memory_dimension()

    print("=" * 60)
    print("⚠️  QDRANT HARD RESET")
    print("=" * 60)
    print(f"\nQdrant Host: {qdrant_host}:{qdrant_port}")
    print(f"Ziel-Kollektion: {memory_collection}")
    print(f"Dimensionen: {memory_dimension}")

    print("\n⚠️  WARNUNG: Dies löscht ALLE Kollektionen in Qdrant!")
    response = input("\nFortfahren? (HARD RESET eingeben): ").strip()

    if response != "HARD RESET":
        print("\n❌ Abgebrochen (Sicherheitsprüfung fehlgeschlagen)")
        return False

    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10.0)
        print("\n✅ Verbunden mit Qdrant")

        # Alle Kollektionen abrufen
        collections = client.get_collections().collections
        print(f"\nGefundene Kollektionen: {[c.name for c in collections]}")

        # Alle Kollektionen löschen
        for collection in collections:
            print(f"Lösche Kollektion: {collection.name}...")
            client.delete_collection(collection_name=collection.name)
            print(f"   ✅ {collection.name} gelöscht")

        print(f"\n🏗️  Erstelle neue Kollektion: {memory_collection}")
        client.create_collection(
            collection_name=memory_collection,
            vectors_config=VectorParams(
                size=memory_dimension,
                distance=Distance.COSINE
            )
        )

        # Verifiziere
        collection_info = client.get_collection(memory_collection)
        print(f"\n✅ Kollektion erstellt:")
        print(f"   Name: {memory_collection}")
        print(f"   Vektor-Größe: {collection_info.config.params.vectors.size}")
        print(f"   Distanz-Metrik: {collection_info.config.params.vectors.distance}")
        print(f"   Anzahl Punkte: {collection_info.points_count}")

        print("\n" + "=" * 60)
        print("✅ HARD RESET ERFOLGREICH!")
        print("=" * 60)
        print("\n✨ Qdrant wurde zurückgesetzt")
        print("✨ Die lexi_memory Kollektion ist bereit")

        return True

    except Exception as e:
        print(f"\n❌ FEHLER: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = hard_reset_qdrant()
    sys.exit(0 if result else 1)
