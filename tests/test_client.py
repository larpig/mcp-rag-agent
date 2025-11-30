import unittest
from src.mongodb.client import MongoDBClient

class TestMongoDBClient(unittest.TestCase):

    def setUp(self):
        self.client = MongoDBClient()

    def test_connect(self):
        result = self.client.connect()
        self.assertTrue(result)

    def test_disconnect(self):
        self.client.connect()
        result = self.client.disconnect()
        self.assertTrue(result)

    def test_get_collection(self):
        self.client.connect()
        collection = self.client.get_collection('test_collection')
        self.assertIsNotNone(collection)

if __name__ == '__main__':
    unittest.main()