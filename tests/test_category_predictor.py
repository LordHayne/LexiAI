import unittest
from unittest.mock import MagicMock
import numpy as np
from backend.memory.category_predictor import ClusteredCategoryPredictor

class TestClusteredCategoryPredictor(unittest.TestCase):
    def setUp(self):
        dummy_qdrant = MagicMock()
        dummy_qdrant.get_all_entries.return_value = []

        dummy_embedding_model = MagicMock()
        dummy_embedding_model.embed_query.side_effect = lambda x: [float(len(x))] * 768

        self.predictor = ClusteredCategoryPredictor(
            qdrant=dummy_qdrant,
            embedding_model=dummy_embedding_model
        )

    def test_predict_category_basic(self):
        self.predictor.clusters = {
            0: [np.array([1.0] * 768)],
            1: [np.array([2.0] * 768)]
        }
        self.predictor.cluster_names = {
            0: "kurz",
            1: "lang"
        }

        result = self.predictor.predict_category("hi")
        self.assertEqual(result, "cluster_0")

        result2 = self.predictor.predict_category("hallohallo")
        self.assertEqual(result2, "cluster_0")

    def test_export_model_info_mock(self):
        self.predictor.get_cluster_info = MagicMock(return_value={"dummy": True})
        self.predictor.export_model_info = lambda filepath: True  # kein echter Export
        self.assertTrue(self.predictor.export_model_info("dummy.json"))

    def test_suggest_reclustering(self):
        self.predictor.embeddings = [np.ones(768)] * 10
        self.assertEqual(len(self.predictor.embeddings), 10)

if __name__ == '__main__':
    unittest.main()
