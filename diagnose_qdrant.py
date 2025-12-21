#!/usr/bin/env python3
"""
Qdrant Diagnose-Script
Überprüft den Zustand der Qdrant-Kollektion und identifiziert Probleme
"""

import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from backend.config.middleware_config import MiddlewareConfig

def diagnose_qdrant():
    """Diagnostiziert Probleme mit der Qdrant-Kollektion"""

    config = MiddlewareConfig()

    # Config-Werte abrufen
    qdrant_host = config.get_qdrant_host()
    qdrant_port = config.get_qdrant_port()
    memory_collection = config.get_memory_collection()
    memory_dimension = config.get_memory_dimension()

    print("=" * 60)
    print("🔍 QDRANT DIAGNOSE")
    print("=" * 60)
    print(f"\n📍 Qdrant Host: {qdrant_host}:{qdrant_port}")
    print(f"📦 Kollektion: {memory_collection}")
    print(f"📐 Erwartete Dimensionen: {memory_dimension}")

    try:
        # Verbindung zu Qdrant herstellen
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            timeout=10.0
        )
        print("\n✅ Verbindung zu Qdrant erfolgreich")

        # Kollektionen auflisten
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        print(f"\n📋 Verfügbare Kollektionen: {collection_names}")

        # Überprüfe ob unsere Kollektion existiert
        if memory_collection not in collection_names:
            print(f"\n⚠️  Kollektion '{memory_collection}' existiert nicht!")
            print("💡 Lösung: Starte den Server mit --force-recreate")
            return False

        # Kollektion-Info abrufen
        collection_info = client.get_collection(memory_collection)
        print(f"\n📊 Kollektion-Info:")
        print(f"   - Vektor-Größe: {collection_info.config.params.vectors.size}")
        print(f"   - Distanz-Metrik: {collection_info.config.params.vectors.distance}")
        print(f"   - Anzahl Punkte: {collection_info.points_count}")
        print(f"   - Anzahl Segmente: {collection_info.segments_count}")

        # Dimensionsprüfung
        if collection_info.config.params.vectors.size != memory_dimension:
            print(f"\n❌ DIMENSIONSFEHLER!")
            print(f"   Erwartet: {memory_dimension}")
            print(f"   Gefunden: {collection_info.config.params.vectors.size}")
            print("💡 Lösung: Starte den Server mit --force-recreate")
            return False

        # Versuche ein paar Punkte abzurufen
        print(f"\n🔍 Überprüfe Datenintegrität...")

        try:
            # Scroll durch die ersten 10 Punkte (ohne Vektoren, da die sehr groß sind)
            result = client.scroll(
                collection_name=memory_collection,
                limit=10,
                with_payload=True,
                with_vectors=False  # Vektoren nicht abrufen - zu groß und nicht nötig
            )

            points = result[0]

            if len(points) == 0:
                print("   ℹ️  Keine Punkte in der Kollektion")
                return True

            print(f"   ✅ {len(points)} Punkte erfolgreich abgerufen")

            # Überprüfe jeden Punkt
            corrupted_points = []
            for point in points:
                try:
                    # Überprüfe Payload
                    if not hasattr(point, 'payload') or not point.payload:
                        corrupted_points.append({
                            'id': point.id,
                            'reason': 'Kein Payload'
                        })

                    # Überprüfe ob Content vorhanden ist
                    payload = point.payload or {}
                    if not payload.get('content'):
                        corrupted_points.append({
                            'id': point.id,
                            'reason': 'Kein Content im Payload'
                        })

                except Exception as e:
                    corrupted_points.append({
                        'id': getattr(point, 'id', 'unknown'),
                        'reason': f'Fehler beim Lesen: {str(e)}'
                    })

            if corrupted_points:
                print(f"\n❌ KORRUPTE DATEN GEFUNDEN!")
                print(f"   Anzahl korrupter Punkte: {len(corrupted_points)}")
                for cp in corrupted_points[:5]:  # Zeige max 5
                    print(f"   - ID: {cp['id']}, Grund: {cp['reason']}")
                print("\n💡 Lösung: Kollektion muss neu erstellt werden")
                return False
            else:
                print("   ✅ Alle überprüften Punkte sind intakt")

        except Exception as e:
            print(f"\n❌ FEHLER beim Lesen der Daten: {str(e)}")
            print(f"   Fehlertyp: {type(e).__name__}")
            print("\n💡 Dies deutet auf korrupte Daten hin")
            print("💡 Lösung: Kollektion muss neu erstellt werden")
            return False

        print("\n✅ Diagnose abgeschlossen - Kollektion ist in Ordnung")
        return True

    except Exception as e:
        print(f"\n❌ FEHLER: {str(e)}")
        print(f"   Fehlertyp: {type(e).__name__}")
        return False

def offer_fix():
    """Bietet Lösungsoptionen an"""
    print("\n" + "=" * 60)
    print("🔧 LÖSUNGSOPTIONEN")
    print("=" * 60)
    print("\nOption 1: Kollektion über Config-UI neu erstellen")
    print("   1. Öffne http://localhost:8000/frontend/pages/config_ui.html")
    print("   2. Aktiviere 'Vektor-Kollektion neu erstellen'")
    print("   3. Speichere die Konfiguration")
    print("   ⚠️  ACHTUNG: Alle Memories werden gelöscht!")

    print("\nOption 2: Server mit --force-recreate starten")
    print("   python start_middleware.py --force-recreate")
    print("   ⚠️  ACHTUNG: Alle Memories werden gelöscht!")

    print("\nOption 3: Qdrant komplett neu starten")
    print("   docker restart qdrant  (falls Docker)")
    print("   oder Qdrant-Service neu starten")

if __name__ == "__main__":
    result = diagnose_qdrant()

    if not result:
        offer_fix()
        sys.exit(1)
    else:
        print("\n✅ Keine Probleme gefunden!")
        sys.exit(0)
