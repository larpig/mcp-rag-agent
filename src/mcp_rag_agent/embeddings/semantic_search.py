"""Semantic search functionality for MongoDB."""

from typing import Any, Optional

from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator


class SemanticSearch:
    """Semantic search engine using MongoDB vector search."""
    
    def __init__(
        self,
        mongo_client: MongoDBClient,
        embedding_generator: EmbeddingGenerator,
        default_collection: str = "vectors",
        default_index: str = "vector_index",
        vector_field: str = "embedding",
        text_field: str = "content"
    ):
        """Initialize semantic search.
        
        Args:
            mongo_client: MongoDB client instance.
            embedding_generator: Embedding generator instance.
            default_collection: Default collection name.
            default_index: Default vector search index name.
            vector_field: Field name for vector embeddings.
            text_field: Field name for text content.
        """
        self._mongo_client = mongo_client
        self._embedding_generator = embedding_generator
        self._default_collection = default_collection
        self._default_index = default_index
        self._vector_field = vector_field
        self._text_field = text_field
    
    async def index_document(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> str:
        """Index a document for semantic search.
        
        Args:
            content: Text content to index.
            metadata: Optional metadata to store with the document.
            collection_name: Collection to store the document.
            
        Returns:
            Inserted document ID.
        """
        collection = collection_name or self._default_collection
        
        # Generate embedding for the content
        embedding = await self._embedding_generator.generate(content)
        
        # Create document with embedding
        document = {
            self._text_field: content,
            self._vector_field: embedding,
            "metadata": metadata or {}
        }
        
        return self._mongo_client.insert_document(collection, document)
    
    async def index_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> list[str]:
        """Index multiple documents for semantic search.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'.
            collection_name: Collection to store the documents.
            
        Returns:
            List of inserted document IDs.
        """
        collection = collection_name or self._default_collection
        
        # Generate embeddings for all documents
        contents = [doc.get("content", "") for doc in documents]
        embeddings = await self._embedding_generator.generate_batch(contents)
        
        # Create documents with embeddings
        docs_with_embeddings = []
        for doc, embedding in zip(documents, embeddings):
            docs_with_embeddings.append({
                self._text_field: doc.get("content", ""),
                self._vector_field: embedding,
                "metadata": doc.get("metadata", {})
            })
        
        return self._mongo_client.insert_documents(collection, docs_with_embeddings)
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        collection_name: Optional[str] = None,
        index_name: Optional[str] = None,
        filter_query: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """Perform semantic search.
        
        Args:
            query: Search query text.
            limit: Maximum number of results.
            collection_name: Collection to search.
            index_name: Vector search index name.
            filter_query: Optional filter to apply.
            
        Returns:
            List of matching documents with scores.
        """
        collection = collection_name or self._default_collection
        index = index_name or self._default_index
        
        # Generate embedding for the query
        query_vector = await self._embedding_generator.generate(query)
        
        # Perform vector search
        results = self._mongo_client.vector_search(
            collection_name=collection,
            index_name=index,
            vector_field=self._vector_field,
            query_vector=query_vector,
            limit=limit,
            filter_query=filter_query
        )
        
        # Format results (remove embedding from response)
        formatted_results = []
        for result in results:
            result.pop(self._vector_field, None)
            result["_id"] = str(result.get("_id", ""))
            formatted_results.append(result)
        
        return formatted_results
    
    def setup_index(
        self,
        collection_name: Optional[str] = None,
        index_name: Optional[str] = None,
        dimensions: Optional[int] = None
    ) -> None:
        """Create vector search index for a collection.
        
        Args:
            collection_name: Collection to create index on.
            index_name: Name for the index.
            dimensions: Vector dimensions (defaults to embedding model dimensions).
        """
        collection = collection_name or self._default_collection
        index = index_name or self._default_index
        dims = dimensions or self._embedding_generator.dimensions
        
        self._mongo_client.create_vector_search_index(
            collection_name=collection,
            index_name=index,
            vector_field=self._vector_field,
            dimensions=dims
        )

async def main():
    """Main function to demonstrate SemanticSearch setup and usage."""
    import asyncio
    from mcp_rag_agent.core.config import config
    
    # Initialize MongoDB client
    print("Initializing MongoDB client...")
    mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
    mongo_client.connect()
    
    # Initialize embedding generator
    print("Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        api_key=config.model_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimension
    )
    
    # Initialize semantic search
    print("Initializing semantic search...")
    semantic_search = SemanticSearch(
        mongo_client=mongo_client,
        embedding_generator=embedding_generator,
        default_collection=config.db_documents_collection,
        default_index="vector_index"
    )
    
    try:
        # Check if collection exists, if not create it
        print(f"\nChecking collection '{config.db_vector_collection}'...")
        
        if not mongo_client.collection_exists(config.db_vector_collection):
            print(f"Collection '{config.db_vector_collection}' does not exist. Creating collection...")
            mongo_client.create_collection(config.db_vector_collection)
            print(f"Collection '{config.db_vector_collection}' created successfully!")
        else:
            print(f"Collection '{config.db_vector_collection}' exists.")

        # Create vector search index
        print(f"Setting up vector search index '{config.db_vector_index_name}'...")
        try:
            semantic_search.setup_index(
                collection_name=config.db_vector_collection,
                index_name=config.db_vector_index_name,
                dimensions=config.embedding_dimension
            )
            print("Vector search index created successfully!")
            print("Note: MongoDB Atlas vector search indexes may take a few minutes to become active.")
        except Exception as e:
            print(f"Index may already exist or error occurred: {e}")
        
        # Index a dummy document
        print("\nIndexing dummy documents...")
        dummy_docs = [
            {
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "metadata": {"source": "dummy", "category": "AI"}
            },
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"source": "dummy", "category": "Programming"}
            },
            {
                "content": "Cloud computing provides on-demand access to computing resources over the internet.",
                "metadata": {"source": "dummy", "category": "Cloud"}
            }
        ]
        
        doc_ids = await semantic_search.index_documents(
            dummy_docs,
            collection_name=config.db_vector_collection
        )
        print(f"Indexed {len(doc_ids)} documents with IDs: {doc_ids}")
        
        # Perform a dummy vector search
        print("\nPerforming dummy vector search...")
        query = "What is artificial intelligence?"
        print(f"Query: '{query}'")
        
        results = await semantic_search.search(
            query=query,
            limit=3,
            collection_name=config.db_vector_collection,
            index_name=config.db_vector_index_name,
        )
        
        print(f"\nFound {len(results)} results:")
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. Score: {result.get('score', 'N/A'):.4f}")
            print(f"   Content: {result.get('content', 'N/A')[:100]}...")
            print(f"   Metadata: {result.get('metadata', {})}")
            print(f"   ID: {result.get('_id', 'N/A')}")
        
        # Delete the dummy documents
        print("\nCleaning up dummy documents...")
        deleted_count = mongo_client.delete_documents(
            collection_name=config.db_vector_collection,
            query={"metadata.source": "dummy"}
        )
        print(f"Deleted {deleted_count} dummy document(s)")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nDisconnecting from MongoDB...")
        mongo_client.disconnect()
        print("Done!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
