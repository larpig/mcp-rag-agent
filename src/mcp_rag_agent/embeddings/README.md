# Embeddings Module

This module provides semantic search functionality using MongoDB Atlas Vector Search and OpenAI embeddings.

## Overview

The embeddings module consists of three main components:

1. **`embedding_generator.py`** - Generates vector embeddings using OpenAI's embedding models
2. **`semantic_search.py`** - Provides semantic search capabilities using MongoDB vector search
3. **`index_documents.py`** - Script to index documents from disk into MongoDB

## Components

### EmbeddingGenerator

Generates embeddings using OpenAI or compatible APIs.

**Features:**
- Supports OpenAI embedding models (e.g., `text-embedding-3-small`)
- Configurable dimensions (default: 1536)
- Batch processing support
- Compatible with OpenAI-compatible APIs

**Usage:**
```python
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

# Initialize
generator = EmbeddingGenerator(
    api_key="your-api-key",
    model="text-embedding-3-small",
    dimensions=1536
)

# Generate single embedding
embedding = await generator.generate("Your text here")

# Generate batch embeddings
embeddings = await generator.generate_batch(["Text 1", "Text 2", "Text 3"])
```

### SemanticSearch

Semantic search engine using MongoDB vector search.

**Features:**
- Document indexing with embeddings
- Vector similarity search
- Metadata filtering
- Automatic index creation
- Batch document processing

**Usage:**
```python
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch

# Initialize components
mongo_client = MongoDBClient(uri="mongodb://...", database_name="mydb")
embedding_generator = EmbeddingGenerator(api_key="your-key")

semantic_search = SemanticSearch(
    mongo_client=mongo_client,
    embedding_generator=embedding_generator,
    default_collection="vectors",
    default_index="vector_index"
)

# Create index
semantic_search.setup_index()

# Index a document
doc_id = await semantic_search.index_document(
    content="Your document content",
    metadata={"source": "example", "category": "docs"}
)

# Search
results = await semantic_search.search(
    query="search query",
    limit=10
)
```

**Demo Script:**

Run the semantic search demo:
```bash
python src/mcp_rag_agent/embeddings/semantic_search.py
```

This demo script will:
- Set up a vector search index
- Index 3 dummy documents
- Perform a test search
- Clean up dummy documents

### Document Indexing Script

The `index_documents.py` script provides a complete pipeline for indexing documents from disk into MongoDB.

## Document Indexing

### Quick Start

**Index documents:**
```bash
python src/mcp_rag_agent/embeddings/index_documents.py
```

**Clear existing data and re-index:**
```bash
python src/mcp_rag_agent/embeddings/index_documents.py --clear
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--clear` | Delete all existing documents and vectors before indexing |

### How It Works

The indexing script performs the following workflow:

1. **Initialize Components**
   - Connect to MongoDB
   - Initialize embedding generator
   - Set up semantic search

2. **Clear Data (Optional)**
   - If `--clear` flag is used:
     - Delete all documents from `documents` collection
     - Delete all vectors from `vectors` collection

3. **Ensure Collections Exist**
   - Create `documents` collection if needed
   - Create `vectors` collection if needed

4. **Set Up Vector Search Index**
   - Create vector search index on `vectors` collection
   - Configure dimensions and similarity metric

5. **Index Documents**
   - Scan `data/ingested_documents` folder recursively
   - For each `.txt` file:
     - Read content
     - Save metadata to `documents` collection
     - Generate embedding
     - Save vector to `vectors` collection with cross-reference

### Document Structure

**Documents Collection:**
```json
{
  "_id": "ObjectId(...)",
  "name": "1 - Remote Working.txt",
  "folder": "policies",
  "relative_path": "policies/1 - Remote Working.txt",
  "absolute_path": "D:/Projects/mcp-rag-agent/data/ingested_documents/policies/1 - Remote Working.txt",
  "content": "Full document content...",
  "size": 172
}
```

**Vectors Collection:**
```json
{
  "_id": "ObjectId(...)",
  "content": "Full document content...",
  "embedding": [0.123, -0.456, ...],  // 1536-dimensional vector
  "metadata": {
    "document_id": "ObjectId(...)",
    "document_name": "1 - Remote Working.txt",
    "folder_name": "policies",
    "relative_path": "policies/1 - Remote Working.txt",
    "content_length": 172
  }
}
```

### Output Example

```
============================================================
Document Indexing Script
============================================================

📊 Initializing MongoDB client...
🤖 Initializing embedding generator...
🔍 Initializing semantic search...

🗑️  Clearing existing data...
   ✅ Deleted 5 document(s) from 'documents'
   ✅ Deleted 5 vector(s) from 'vectors'

🗂️  Checking collections...
   ✅ Collection 'documents' exists
   ✅ Collection 'vectors' exists

🔧 Setting up vector search index...
   ✅ Vector search index 'vector_index' ready
   ℹ️  Note: New indexes may take a few minutes to become fully active

Found 5 document(s) to index
============================================================

📄 Processing: 1 - Remote Working.txt
   Folder: policies
   Content length: 172 characters
   ✅ Saved to documents collection (ID: 692af121...)
   ✅ Saved to vectors collection (ID: 692af122...)

... (more documents)

============================================================
✨ Indexing complete! Successfully indexed 5/5 document(s)

🔌 Disconnecting from MongoDB...
✅ Done!
```

## Configuration

The module uses settings from `config.py`:

```python
# MongoDB settings
db_url: str                        # MongoDB connection URI
db_name: str                       # Database name
db_documents_collection: str       # Collection for document metadata
db_vector_collection: str          # Collection for vectors
db_vector_index_name: str         # Name of vector search index

# Embedding settings
model_api_key: str                 # OpenAI API key
embedding_model: str               # Model name (e.g., "text-embedding-3-small")
embedding_dimension: int           # Vector dimensions (default: 1536)
```

## Requirements

- MongoDB Atlas cluster with vector search support
- OpenAI API key
- Python 3.11+
- Required packages:
  - `pymongo`
  - `openai`
  - `pydantic-settings`
  - `python-dotenv`

## Notes

### Vector Search Index

MongoDB Atlas vector search indexes may take a few minutes to become fully active after creation. If searches return no results immediately after indexing, wait a few minutes and try again.

### Embedding Costs

Generating embeddings uses the OpenAI API and incurs costs based on:
- Model used (e.g., `text-embedding-3-small`)
- Number of tokens processed
- Refer to OpenAI pricing for current rates

### Best Practices

1. **Use `--clear` flag judiciously** - It deletes all existing data
2. **Monitor embedding costs** - Batch operations are more efficient
3. **Wait for index activation** - New indexes need time to build
4. **Use meaningful metadata** - Helps with filtering and organization
5. **Handle large files** - Consider chunking for very large documents

## Troubleshooting

**Issue: Search returns no results**
- Wait a few minutes for the index to become active
- Verify documents were indexed successfully
- Check that the index name matches configuration

**Issue: Import errors**
- Ensure the package is installed: `pip install -e .`
- Check that all dependencies are installed

**Issue: OpenAI API errors**
- Verify API key is correct in `.env` file
- Check API rate limits and quotas
- Ensure sufficient credits in OpenAI account

## Examples

### Basic Indexing Workflow

```python
import asyncio
from mcp_rag_agent.config import config
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch

async def main():
    # Initialize
    mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
    mongo_client.connect()
    
    embedding_generator = EmbeddingGenerator(
        api_key=config.model_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimension
    )
    
    semantic_search = SemanticSearch(
        mongo_client=mongo_client,
        embedding_generator=embedding_generator
    )
    
    # Setup index
    semantic_search.setup_index()
    
    # Index documents
    docs = [
        {"content": "Document 1", "metadata": {"type": "policy"}},
        {"content": "Document 2", "metadata": {"type": "guide"}}
    ]
    
    doc_ids = await semantic_search.index_documents(docs)
    print(f"Indexed {len(doc_ids)} documents")
    
    # Search
    results = await semantic_search.search("policy information", limit=5)
    for result in results:
        print(f"Score: {result['score']}, Content: {result['content'][:100]}...")
    
    # Cleanup
    mongo_client.disconnect()

asyncio.run(main())
```

### Searching with Filters

```python
# Search with metadata filter
results = await semantic_search.search(
    query="remote work policy",
    limit=10,
    filter_query={"metadata.folder_name": "policies"}
)
```

## License

This module is part of the MCP RAG Agent project.
