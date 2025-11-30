"""MCP server for RAG agent with semantic search capabilities."""
import logging

from mcp_rag_agent.core.config import config
from mcp_rag_agent.core.log_setup import setup_logging
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch

setup_logging()
logger = logging.getLogger("Retriever")


# Initialize MongoDB client
logger.info("Initializing MongoDB client...")
mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
mongo_client.connect()

# Initialize embedding generator
logger.info("Initializing embedding generator...")
embedding_generator = EmbeddingGenerator(
    api_key=config.model_api_key,
    model=config.embedding_model,
    dimensions=config.embedding_dimension
)

# Initialize semantic search
logger.info("Initializing semantic search...")
semantic_search = SemanticSearch(
    mongo_client=mongo_client,
    embedding_generator=embedding_generator,
    default_collection=config.db_documents_collection,
    default_index=config.db_vector_index_name
)

async def search_documents(
    query: str,
    top_k: int = 3
) -> list[dict]:
    """
    Perform semantic search on indexed documents using vector embeddings.

    This tool searches through the document collection using semantic similarity
    to find the most relevant documents matching the user's query. It converts
    the query text into embeddings and uses MongoDB's vector search capabilities
    to retrieve similar documents. These documents can be used to ground the 
    user's answer.

    Args:
        query (str): The search query text to find relevant documents.
        top_k (int, optional): The maximum number of results to return.
            Defaults to 3.

    Returns:
        list[dict]: A list of matching documents, ordered by relevance.
    """
    logger.info(f"Semantic search started: {query}")

    results = await semantic_search.search(
        query=query,
        limit=top_k,
        collection_name=config.db_vector_collection,
        index_name=config.db_vector_index_name,
    )

    logger.info(f"Found {len(results)} relevant documents")

    return results
