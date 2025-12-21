import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.qdrant.client_wrapper import create_qdrant_client
from backend.qdrant.qdrant_interface import QdrantMemoryInterface
from backend.embeddings.embedding_model import OllamaEmbeddingModel
from backend.memory.category_predictor import ClusteredCategoryPredictor
from datetime import datetime, timezone

# Setup
qdrant_client = create_qdrant_client()
embedding_model = OllamaEmbeddingModel()
collection_name = "lexi_memory"

qdrant_interface = QdrantMemoryInterface(
    qdrant_client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_model
)

predictor = ClusteredCategoryPredictor(
    qdrant=qdrant_interface,
    embedding_model=embedding_model
)

# Testdaten hinzufügen (ähnliche Aussagen)
texts = [
    "Ich heiße Thomas und bin männlich.",
    "Ich mag Schokolade und gehe gerne spazieren.",
    "Ich liebe Eis und treffe mich gerne mit Freunden.",
    "Meine Hobbys sind Lesen und Natur genießen.",
    "Ich bin gerne draußen unterwegs und mache Sport.",
    "Ich spiele gerne Gitarre.",
    "Musik ist meine Leidenschaft.",
    "Ich höre gerne Rockmusik.",
    "Sport mache ich fast täglich.",
    "Joggen am Morgen tut mir gut.",
    "Ich esse gerne Pizza.",
    "Italienisches Essen ist mein Favorit.",
    "Ich lese gerne Bücher über Psychologie.",
    "Romane entspannen mich.",
    "Natur und Wandern finde ich toll."
]
technik = [
    "Wie funktioniert ein Verbrennungsmotor?",
    "Was ist ein Katalysator?",
    "Wann sollte man das Motoröl wechseln?",
    "Unterschied zwischen Hybrid- und Elektromotor.",
    "Wie liest man einen Fehlerspeicher aus?"
]


for text in texts:
    predictor.assign_and_store(text, {"created_at": datetime.now(timezone.utc).isoformat()})


# Clustering neu aufbauen
predictor.rebuild_clusters()

# Neue Eingabe kategorisieren
sample_text = "Ich mag Schokolade und gehe gerne spazieren."
category = predictor.predict_category(sample_text)
print(f"Kategorie: {category}")
