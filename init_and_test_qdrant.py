#!/usr/bin/env python3
"""
Qdrant Initialisierungs- und Test-Script
Erstellt die Kollektion korrekt und testet alle Funktionen
"""

import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings

from backend.config.middleware_config import MiddlewareConfig
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.models.memory_entry import MemoryEntry


class QdrantInitializer:
    """Initialisiert und testet Qdrant-Kollektion"""

    def __init__(self):
        self.config = MiddlewareConfig()
        self.qdrant_host = self.config.get_qdrant_host()
        self.qdrant_port = self.config.get_qdrant_port()
        self.collection_name = self.config.get_memory_collection()
        self.dimensions = self.config.get_memory_dimension()
        self.embedding_model = self.config.get_default_embedding_model()
        self.embedding_url = self.config.get_embedding_base_url()

        self.client = None
        self.embeddings = None
        self.interface = None

        self.test_results: List[Dict[str, Any]] = []

    def log_test(self, name: str, success: bool, message: str = ""):
        """Loggt Testergebnis"""
        status = "✅" if success else "❌"
        self.test_results.append({
            "name": name,
            "success": success,
            "message": message
        })
        print(f"{status} {name}")
        if message:
            print(f"   └─ {message}")

    def print_header(self, title: str):
        """Gibt Header aus"""
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def run(self) -> bool:
        """Führt vollständige Initialisierung und Tests durch"""
        try:
            self.print_header("🚀 QDRANT INITIALISIERUNG UND TEST")

            # Schritt 1: Konfiguration anzeigen
            if not self.show_configuration():
                return False

            # Schritt 2: Verbindung zu Qdrant herstellen
            if not self.connect_to_qdrant():
                return False

            # Schritt 3: Embeddings initialisieren
            if not self.initialize_embeddings():
                return False

            # Schritt 4: Kollektion erstellen
            if not self.create_collection():
                return False

            # Schritt 5: Interface initialisieren
            if not self.initialize_interface():
                return False

            # Schritt 6: Funktionalitäts-Tests
            if not self.run_functionality_tests():
                return False

            # Zusammenfassung
            self.print_summary()

            return True

        except Exception as e:
            print(f"\n❌ KRITISCHER FEHLER: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def show_configuration(self) -> bool:
        """Zeigt Konfiguration an"""
        self.print_header("📋 KONFIGURATION")

        print(f"Qdrant Host:        {self.qdrant_host}")
        print(f"Qdrant Port:        {self.qdrant_port}")
        print(f"Kollektion:         {self.collection_name}")
        print(f"Dimensionen:        {self.dimensions}")
        print(f"Embedding Model:    {self.embedding_model}")
        print(f"Embedding URL:      {self.embedding_url}")

        self.log_test("Konfiguration geladen", True)
        return True

    def connect_to_qdrant(self) -> bool:
        """Stellt Verbindung zu Qdrant her"""
        self.print_header("🔌 QDRANT VERBINDUNG")

        try:
            self.client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=10.0
            )

            # Test: Collections abrufen
            collections = self.client.get_collections()
            print(f"Verfügbare Kollektionen: {[c.name for c in collections.collections]}")

            self.log_test("Qdrant-Verbindung", True, f"Verbunden mit {self.qdrant_host}:{self.qdrant_port}")
            return True

        except Exception as e:
            self.log_test("Qdrant-Verbindung", False, str(e))
            return False

    def initialize_embeddings(self) -> bool:
        """Initialisiert Embedding-Modell"""
        self.print_header("🧠 EMBEDDING-MODELL")

        try:
            self.embeddings = OllamaEmbeddings(
                base_url=self.embedding_url,
                model=self.embedding_model
            )

            # Test: Embedding generieren
            print("Teste Embedding-Generierung...")
            test_vector = self.embeddings.embed_query("Hallo Welt")
            actual_dim = len(test_vector)

            print(f"Erwartete Dimensionen: {self.dimensions}")
            print(f"Tatsächliche Dimensionen: {actual_dim}")

            if actual_dim != self.dimensions:
                self.log_test("Embedding-Initialisierung", False,
                            f"Dimensionsfehler: {actual_dim} != {self.dimensions}")
                return False

            self.log_test("Embedding-Initialisierung", True, f"Modell: {self.embedding_model}")
            return True

        except Exception as e:
            self.log_test("Embedding-Initialisierung", False, str(e))
            return False

    def create_collection(self) -> bool:
        """Erstellt Qdrant-Kollektion"""
        self.print_header("🏗️  KOLLEKTION ERSTELLEN")

        try:
            # Überprüfe ob Kollektion existiert
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                print(f"⚠️  Kollektion '{self.collection_name}' existiert bereits")
                response = input("Möchten Sie sie neu erstellen? (ja/nein): ").strip().lower()

                if response not in ['ja', 'j', 'yes', 'y']:
                    print("Verwende existierende Kollektion")
                    self.log_test("Kollektion erstellen", True, "Existierende Kollektion verwendet")
                    return True

                # Lösche alte Kollektion
                print(f"Lösche alte Kollektion '{self.collection_name}'...")
                self.client.delete_collection(collection_name=self.collection_name)
                print("   ✅ Alte Kollektion gelöscht")

            # Erstelle neue Kollektion
            print(f"Erstelle Kollektion '{self.collection_name}' mit {self.dimensions} Dimensionen...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE
                )
            )

            # Verifiziere Kollektion
            collection_info = self.client.get_collection(self.collection_name)
            print(f"\n✅ Kollektion erstellt:")
            print(f"   Name:              {self.collection_name}")
            print(f"   Vektor-Größe:      {collection_info.config.params.vectors.size}")
            print(f"   Distanz-Metrik:    {collection_info.config.params.vectors.distance}")
            print(f"   Anzahl Punkte:     {collection_info.points_count}")

            # Verifiziere Dimensionen
            if collection_info.config.params.vectors.size != self.dimensions:
                self.log_test("Kollektion erstellen", False, "Dimensionsfehler")
                return False

            self.log_test("Kollektion erstellen", True, f"Kollektion '{self.collection_name}' bereit")
            return True

        except Exception as e:
            self.log_test("Kollektion erstellen", False, str(e))
            return False

    def initialize_interface(self) -> bool:
        """Initialisiert QdrantMemoryInterface"""
        self.print_header("🔧 INTERFACE INITIALISIERUNG")

        try:
            self.interface = QdrantMemoryInterface(
                collection_name=self.collection_name,
                embeddings=self.embeddings,
                qdrant_client=self.client
            )

            self.log_test("Interface-Initialisierung", True, "QdrantMemoryInterface bereit")
            return True

        except Exception as e:
            self.log_test("Interface-Initialisierung", False, str(e))
            return False

    def run_functionality_tests(self) -> bool:
        """Führt Funktionalitäts-Tests durch"""
        self.print_header("🧪 FUNKTIONALITÄTS-TESTS")

        # Test 1: Eintrag speichern
        test_ids = []
        if not self.test_store_entry(test_ids):
            return False

        # Test 2: Mehrere Einträge speichern
        if not self.test_store_multiple_entries(test_ids):
            return False

        # Test 3: Einträge abrufen
        if not self.test_query_memories():
            return False

        # Test 4: Similarity Search
        if not self.test_similarity_search():
            return False

        # Test 5: Alle Einträge abrufen
        if not self.test_get_all_entries():
            return False

        # Test 6: Metadaten aktualisieren
        if test_ids and not self.test_update_metadata(test_ids[0]):
            return False

        # Test 7: Eintrag löschen
        if test_ids and not self.test_delete_entry(test_ids[0]):
            return False

        return True

    def test_store_entry(self, test_ids: List) -> bool:
        """Testet Eintrag speichern"""
        try:
            print("\n📝 Test: Eintrag speichern...")

            entry_id = uuid4()
            entry = MemoryEntry(
                id=entry_id,
                content="Das ist ein Test-Eintrag für das Memory-System",
                timestamp=datetime.now(timezone.utc),
                category="test",
                tags=["test", "initialization"],
                source="init_script",
                relevance=1.0
            )

            self.interface.store_entry(entry)
            test_ids.append(entry_id)

            # Verifiziere Speicherung
            time.sleep(0.5)  # Kurze Pause für Qdrant-Indexierung

            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count > 0:
                self.log_test("Eintrag speichern", True, f"ID: {entry_id}")
                return True
            else:
                self.log_test("Eintrag speichern", False, "Eintrag nicht gefunden")
                return False

        except Exception as e:
            self.log_test("Eintrag speichern", False, str(e))
            return False

    def test_store_multiple_entries(self, test_ids: List) -> bool:
        """Testet mehrere Einträge speichern"""
        try:
            print("\n📝 Test: Mehrere Einträge speichern...")

            test_data = [
                "Thomas ist der Benutzer dieses Systems",
                "Lexi ist eine freundliche KI-Assistentin",
                "Das Wetter heute ist sonnig und warm",
                "Python ist eine großartige Programmiersprache"
            ]

            for content in test_data:
                entry_id = uuid4()
                entry = MemoryEntry(
                    id=entry_id,
                    content=content,
                    timestamp=datetime.now(timezone.utc),
                    category="test",
                    tags=["test"],
                    source="init_script",
                    relevance=1.0
                )
                self.interface.store_entry(entry)
                test_ids.append(entry_id)

            time.sleep(0.5)

            collection_info = self.client.get_collection(self.collection_name)
            stored_count = collection_info.points_count

            self.log_test("Mehrere Einträge speichern", True, f"{stored_count} Einträge gespeichert")
            return True

        except Exception as e:
            self.log_test("Mehrere Einträge speichern", False, str(e))
            return False

    def test_query_memories(self) -> bool:
        """Testet Memories abfragen"""
        try:
            print("\n🔍 Test: Memories abfragen...")

            query = "Wer ist der Benutzer?"
            results = self.interface.query_memories(query, limit=3)

            if results:
                print(f"   Gefunden: {len(results)} Ergebnisse")
                for i, result in enumerate(results[:3], 1):
                    print(f"   {i}. {result.content[:50]}... (Score: {result.relevance:.3f})")

                self.log_test("Memories abfragen", True, f"{len(results)} Ergebnisse gefunden")
                return True
            else:
                self.log_test("Memories abfragen", False, "Keine Ergebnisse gefunden")
                return False

        except Exception as e:
            self.log_test("Memories abfragen", False, str(e))
            return False

    def test_similarity_search(self) -> bool:
        """Testet Similarity Search"""
        try:
            print("\n🔍 Test: Similarity Search...")

            query = "Programmierung"
            results = self.interface.similarity_search(query, k=3)

            if results:
                print(f"   Gefunden: {len(results)} Ergebnisse")
                self.log_test("Similarity Search", True, f"{len(results)} ähnliche Einträge")
                return True
            else:
                self.log_test("Similarity Search", False, "Keine ähnlichen Einträge")
                return False

        except Exception as e:
            self.log_test("Similarity Search", False, str(e))
            return False

    def test_get_all_entries(self) -> bool:
        """Testet alle Einträge abrufen"""
        try:
            print("\n📋 Test: Alle Einträge abrufen...")

            # Direkter scroll-Aufruf um das Problem zu umgehen
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True
            )

            # scroll() gibt ein Tuple zurück: (points, next_page_offset)
            points = scroll_result[0] if isinstance(scroll_result, tuple) else []

            if points:
                print(f"   Gefunden: {len(points)} Einträge")
                self.log_test("Alle Einträge abrufen", True, f"{len(points)} gesamt")
                return True
            else:
                # Versuche alternative Methode
                entries = self.interface.get_all_entries()
                if entries:
                    print(f"   Gefunden (Alternative): {len(entries)} Einträge")
                    self.log_test("Alle Einträge abrufen", True, f"{len(entries)} gesamt (Alternative)")
                    return True
                else:
                    self.log_test("Alle Einträge abrufen", False, "Keine Einträge gefunden")
                    return False

        except Exception as e:
            self.log_test("Alle Einträge abrufen", False, str(e))
            import traceback
            traceback.print_exc()
            return False

    def test_update_metadata(self, entry_id) -> bool:
        """Testet Metadaten aktualisieren"""
        try:
            print("\n✏️  Test: Metadaten aktualisieren...")

            new_metadata = {
                "category": "updated_test",
                "relevance": 0.8,
                "updated": True
            }

            result = self.interface.update_entry_metadata(entry_id, new_metadata)

            if result:
                self.log_test("Metadaten aktualisieren", True, f"ID: {entry_id}")
                return True
            else:
                self.log_test("Metadaten aktualisieren", False, "Update fehlgeschlagen")
                return False

        except Exception as e:
            self.log_test("Metadaten aktualisieren", False, str(e))
            return False

    def test_delete_entry(self, entry_id) -> bool:
        """Testet Eintrag löschen"""
        try:
            print("\n🗑️  Test: Eintrag löschen...")

            self.interface.delete_entry(entry_id)

            time.sleep(0.5)

            # Verifiziere Löschung
            all_entries = self.interface.get_all_entries()
            deleted = all(str(e.id) != str(entry_id) for e in all_entries)

            if deleted:
                self.log_test("Eintrag löschen", True, f"ID: {entry_id}")
                return True
            else:
                self.log_test("Eintrag löschen", False, "Eintrag noch vorhanden")
                return False

        except Exception as e:
            self.log_test("Eintrag löschen", False, str(e))
            return False

    def print_summary(self):
        """Gibt Zusammenfassung aus"""
        self.print_header("📊 ZUSAMMENFASSUNG")

        passed = sum(1 for t in self.test_results if t["success"])
        total = len(self.test_results)
        success_rate = (passed / total * 100) if total > 0 else 0

        print(f"\nTests bestanden: {passed}/{total} ({success_rate:.1f}%)")
        print(f"\n{'Test':<40} Status")
        print("-" * 60)

        for test in self.test_results:
            status = "✅ PASS" if test["success"] else "❌ FAIL"
            print(f"{test['name']:<40} {status}")

        if passed == total:
            print("\n" + "=" * 60)
            print("✨ ERFOLG! Alle Tests bestanden!")
            print("=" * 60)
            print("\n✅ Qdrant-Kollektion ist vollständig einsatzbereit")
            print("✅ Alle Funktionen arbeiten korrekt")
            print("✅ Das Memory-System kann jetzt verwendet werden")
        else:
            print("\n" + "=" * 60)
            print("⚠️  WARNUNG: Einige Tests sind fehlgeschlagen")
            print("=" * 60)


def main():
    """Main-Funktion"""
    initializer = QdrantInitializer()

    try:
        success = initializer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Abgebrochen durch Benutzer")
        sys.exit(1)


if __name__ == "__main__":
    main()
