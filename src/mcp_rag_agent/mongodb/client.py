"""MongoDB client for database operations."""

from typing import Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class MongoDBClient:
    """MongoDB client wrapper for semantic search operations."""
    
    def __init__(self, uri: str, database_name: str):
        """Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection URI.
            database_name: Name of the database to use.
        """
        self._uri = uri
        self._database_name = database_name
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
    
    def connect(self) -> None:
        """Establish connection to MongoDB."""
        if self._client is None:
            self._client = MongoClient(self._uri)
            self._db = self._client[self._database_name]
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
    
    @property
    def db(self) -> Database:
        """Get the database instance."""
        if self._db is None:
            self.connect()
        return self._db
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection by name.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            MongoDB collection instance.
        """
        return self.db[collection_name]
    
    def list_collections(self) -> list[str]:
        """List all collections in the database.
        
        Returns:
            List of collection names.
        """
        return self.db.list_collection_names()
    
    def create_collection(self, collection_name: str) -> Collection:
        """Create a new collection in the database.
        
        Args:
            collection_name: Name of the collection to create.
            
        Returns:
            MongoDB collection instance.
        """
        return self.db.create_collection(collection_name)
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the database.
        
        Args:
            collection_name: Name of the collection to check.
            
        Returns:
            True if collection exists, False otherwise.
        """
        return collection_name in self.list_collections()
    
    def insert_document(
        self, 
        collection_name: str, 
        document: dict[str, Any]
    ) -> str:
        """Insert a document into a collection.
        
        Args:
            collection_name: Name of the collection.
            document: Document to insert.
            
        Returns:
            Inserted document ID as string.
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_documents(
        self, 
        collection_name: str, 
        documents: list[dict[str, Any]]
    ) -> list[str]:
        """Insert multiple documents into a collection.
        
        Args:
            collection_name: Name of the collection.
            documents: List of documents to insert.
            
        Returns:
            List of inserted document IDs as strings.
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def find_documents(
        self, 
        collection_name: str, 
        query: dict[str, Any], 
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find documents matching a query.
        
        Args:
            collection_name: Name of the collection.
            query: MongoDB query filter.
            limit: Maximum number of documents to return.
            
        Returns:
            List of matching documents.
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find(query).limit(limit)
        return list(cursor)
    
    def delete_documents(
        self,
        collection_name: str,
        query: dict[str, Any]
    ) -> int:
        """Delete documents matching a query.
        
        Args:
            collection_name: Name of the collection.
            query: MongoDB query filter for documents to delete.
            
        Returns:
            Number of documents deleted.
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_many(query)
        return result.deleted_count
    
    def create_vector_search_index(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        dimensions: int,
        similarity: str = "cosine"
    ) -> None:
        """Create a vector search index on a collection.
        
        Args:
            collection_name: Name of the collection.
            index_name: Name for the search index.
            vector_field: Field containing the vector embeddings.
            dimensions: Number of dimensions in the vectors.
            similarity: Similarity metric (cosine, euclidean, dotProduct).
        """
        collection = self.get_collection(collection_name)
        
        index_definition = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_field,
                        "numDimensions": dimensions,
                        "similarity": similarity
                    }
                ]
            }
        }
        
        collection.create_search_index(index_definition)
    
    def vector_search(
        self,
        collection_name: str,
        index_name: str,
        vector_field: str,
        query_vector: list[float],
        limit: int = 10,
        num_candidates: int = 100,
        filter_query: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            collection_name: Name of the collection.
            index_name: Name of the vector search index.
            vector_field: Field containing the vector embeddings.
            query_vector: Query vector for similarity search.
            limit: Maximum number of results to return.
            num_candidates: Number of candidates to consider.
            filter_query: Optional filter to apply to results.
            
        Returns:
            List of matching documents with similarity scores.
        """
        collection = self.get_collection(collection_name)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_field,
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        if filter_query:
            pipeline[0]["$vectorSearch"]["filter"] = filter_query
        
        return list(collection.aggregate(pipeline))


def main():
    """Main function to demonstrate list_collections method."""
    from mcp_rag_agent.core.config import config
    
    # Get MongoDB configuration from config
    uri = config.db_url
    database_name = config.db_name
    
    # Create client instance
    client = MongoDBClient(uri=uri, database_name=database_name)
    
    try:
        # Connect and list collections
        print(f"Connecting to MongoDB database: {database_name}")
        client.connect()
        
        collections = client.list_collections()
        
        print(f"\nFound {len(collections)} collection(s):")
        for collection in collections:
            print(f"  - {collection}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure cleanup
        client.disconnect()
        print("\nDisconnected from MongoDB")


if __name__ == "__main__":
    main()
