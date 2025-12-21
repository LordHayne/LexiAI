#!/usr/bin/env python3
"""
Qdrant Fix-Script
Erstellt die Qdrant-Kollektion neu und behebt korrupte Daten
"""

import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.config.middleware_config import MiddlewareConfig

def fix_qdrant():
    """Erstellt die Qdrant-Kollektion neu"""

    config = MiddlewareConfig()

    # Config-Werte abrufen
    qdrant_host = config.get_qdrant_host()
    qdrant_port = config.get_qdrant_port()
    memory_collection = config.get_memory_collection()
    memory_dimension = config.get_memory_dimension()

    print("=" * 60)
    print("🔧 QDRANT FIX - KOLLEKTION NEU ERSTELLEN")
    print("=" * 60)
    print(f"\n📍 Qdrant Host: {qdrant_host}:{qdrant_port}")
    print(f"📦 Kollektion: {memory_collection}")
    print(f"📐 Dimensionen: {memory_dimension}")

    print("\n⚠️  WARNUNG: Alle gespeicherten Memories werden gelöscht!")
    print("⚠️  Dies behebt korrupte Daten, aber löscht alle Einträge.")

    response = input("\n❓ Möchten Sie fortfahren? (ja/nein): ").strip().lower()

    if response not in ['ja', 'j', 'yes', 'y']:
        print("\n❌ Abgebrochen.")
        return False

    try:
        # Verbindung zu Qdrant herstellen
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            timeout=10.0
        )
        print("\n✅ Verbindung zu Qdrant erfolgreich")

        # Überprüfe ob Kollektion existiert
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if memory_collection in collection_names:
            print(f"\n🗑️  Lösche alte Kollektion '{memory_collection}'...")
            client.delete_collection(collection_name=memory_collection)
            print("   ✅ Alte Kollektion gelöscht")
        else:
            print(f"\n ℹ️  Kollektion '{memory_collection}' existiert nicht")

        # Erstelle neue Kollektion
        print(f"\n🏗️  Erstelle neue Kollektion '{memory_collection}'...")
        client.create_collection(
            collection_name=memory_collection,
            vectors_config=VectorParams(
                size=memory_dimension,
                distance=Distance.COSINE
            )
        )
        print("   ✅ Neue Kollektion erstellt")

        # Verifiziere Kollektion
        print(f"\n🔍 Verifiziere neue Kollektion...")
        collection_info = client.get_collection(memory_collection)
        print(f"   ✅ Kollektion verifiziert:")
        print(f"      - Vektor-Größe: {collection_info.config.params.vectors.size}")
        print(f"      - Distanz-Metrik: {collection_info.config.params.vectors.distance}")
        print(f"      - Anzahl Punkte: {collection_info.points_count}")

        print("\n" + "=" * 60)
        print("✅ ERFOLG!")
        print("=" * 60)
        print("\n✨ Die Qdrant-Kollektion wurde erfolgreich neu erstellt.")
        print("✨ Das Memory-System ist jetzt wieder einsatzbereit.")
        print("\n💡 Hinweis: Lexi wird neue Informationen lernen, sobald Sie")
        print("   mit ihr chatten.")

        return True

    except Exception as e:
        print(f"\n❌ FEHLER: {str(e)}")
        print(f"   Fehlertyp: {type(e).__name__}")
        print("\n💡 Mögliche Lösungen:")
        print("   - Stelle sicher, dass Qdrant läuft")
        print("   - Überprüfe die Netzwerkverbindung")
        print("   - Starte Qdrant neu: docker restart qdrant")
        return False

if __name__ == "__main__":
    result = fix_qdrant()
    sys.exit(0 if result else 1)
