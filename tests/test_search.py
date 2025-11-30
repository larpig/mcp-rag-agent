import unittest
from src.mongodb.client import MongoDBClient
from src.mongodb.semantic_search import SemanticSearch

class TestSemanticSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = MongoDBClient()
        cls.client.connect()
        cls.collection = cls.client.get_collection('test_collection')
        cls.semantic_search = SemanticSearch(cls.collection)

    @classmethod
    def tearDownClass(cls):
        cls.client.disconnect()

    def test_index_documents(self):
        documents = [
            {"text": "This is a test document."},
            {"text": "This document is for testing semantic search."}
        ]
        self.semantic_search.index_documents(documents)
        # Verify that documents are indexed correctly
        indexed_count = self.collection.count_documents({})
        self.assertEqual(indexed_count, len(documents))

    def test_search(self):
        query = "test document"
        results = self.semantic_search.search(query)
        # Verify that the search returns expected results
        self.assertGreater(len(results), 0)
        self.assertIn("This is a test document.", [result['text'] for result in results])

if __name__ == '__main__':
    unittest.main()