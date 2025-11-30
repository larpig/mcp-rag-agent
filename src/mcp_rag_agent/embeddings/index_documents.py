"""Script to index documents from the ingested_documents folder into MongoDB."""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp_rag_agent.core.config import config
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch


async def index_documents_from_folder(
    folder_path: str,
    mongo_client: MongoDBClient,
    semantic_search: SemanticSearch,
    documents_collection: str,
    vectors_collection: str
) -> None:
    """Index all documents from a folder into MongoDB.
    
    Args:
        folder_path: Path to the folder containing documents.
        mongo_client: MongoDB client instance.
        semantic_search: SemanticSearch instance.
        documents_collection: Collection name for storing document metadata.
        vectors_collection: Collection name for storing vectors.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Get all text files recursively
    text_files = list(folder.rglob("*.txt"))
    
    if not text_files:
        print(f"No .txt files found in '{folder_path}'")
        return
    
    print(f"\nFound {len(text_files)} document(s) to index")
    print("=" * 60)
    
    indexed_count = 0
    
    for file_path in text_files:
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                print(f"⚠️  Skipping empty file: {file_path.name}")
                continue
            
            # Get relative path from base folder
            relative_path = file_path.relative_to(folder)
            folder_name = relative_path.parent.name if relative_path.parent != Path('.') else "root"
            
            print(f"\n📄 Processing: {file_path.name}")
            print(f"   Folder: {folder_name}")
            print(f"   Content length: {len(content)} characters")
            
            # First, save document metadata to documents collection
            document_metadata = {
                "name": file_path.name,
                "folder": folder_name,
                "relative_path": str(relative_path),
                "absolute_path": str(file_path.absolute()),
                "content": content,
                "size": len(content),
                "created_at": datetime.utcnow()
            }
            
            doc_id = mongo_client.insert_document(
                collection_name=documents_collection,
                document=document_metadata
            )
            print(f"   ✅ Saved to documents collection (ID: {doc_id})")
            
            # Now create embedding and save to vectors collection
            vector_metadata = {
                "document_id": doc_id,
                "document_name": file_path.name,
                "folder_name": folder_name,
                "relative_path": str(relative_path),
                "content_length": len(content),
                "created_at": datetime.utcnow()
            }
            
            # Use semantic_search to index the document with embedding
            vector_id = await semantic_search.index_document(
                content=content,
                metadata=vector_metadata,
                collection_name=vectors_collection
            )
            print(f"   ✅ Saved to vectors collection (ID: {vector_id})")
            
            indexed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✨ Indexing complete! Successfully indexed {indexed_count}/{len(text_files)} document(s)")


async def main(clear_existing: bool = False):
    """Main function to index documents.
    
    Args:
        clear_existing: If True, deletes all existing documents before indexing.
    """
    print("=" * 60)
    print("Document Indexing Script")
    print("=" * 60)
    
    # Initialize MongoDB client
    print("\n📊 Initializing MongoDB client...")
    mongo_client = MongoDBClient(uri=config.db_url, database_name=config.db_name)
    mongo_client.connect()
    
    # Initialize embedding generator
    print("🤖 Initializing embedding generator...")
    embedding_generator = EmbeddingGenerator(
        api_key=config.model_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimension
    )
    
    # Initialize semantic search
    print("🔍 Initializing semantic search...")
    semantic_search = SemanticSearch(
        mongo_client=mongo_client,
        embedding_generator=embedding_generator,
        default_collection=config.db_vector_collection,
        default_index=config.db_vector_index_name
    )
    
    try:
        # Clear existing data if requested
        if clear_existing:
            print(f"\n🗑️  Clearing existing data...")
            
            if mongo_client.collection_exists(config.db_documents_collection):
                deleted_docs = mongo_client.delete_documents(
                    collection_name=config.db_documents_collection,
                    query={}
                )
                print(f"   ✅ Deleted {deleted_docs} document(s) from '{config.db_documents_collection}'")
            
            if mongo_client.collection_exists(config.db_vector_collection):
                deleted_vectors = mongo_client.delete_documents(
                    collection_name=config.db_vector_collection,
                    query={}
                )
                print(f"   ✅ Deleted {deleted_vectors} vector(s) from '{config.db_vector_collection}'")
        
        # Ensure collections exist
        print(f"\n🗂️  Checking collections...")
        
        # Check documents collection
        if not mongo_client.collection_exists(config.db_documents_collection):
            print(f"   Creating '{config.db_documents_collection}' collection...")
            mongo_client.create_collection(config.db_documents_collection)
            print(f"   ✅ Collection '{config.db_documents_collection}' created")
        else:
            print(f"   ✅ Collection '{config.db_documents_collection}' exists")
        
        # Check vectors collection
        if not mongo_client.collection_exists(config.db_vector_collection):
            print(f"   Creating '{config.db_vector_collection}' collection...")
            mongo_client.create_collection(config.db_vector_collection)
            print(f"   ✅ Collection '{config.db_vector_collection}' created")
        else:
            print(f"   ✅ Collection '{config.db_vector_collection}' exists")
        
        # Ensure vector search index exists
        print(f"\n🔧 Setting up vector search index...")
        try:
            semantic_search.setup_index(
                collection_name=config.db_vector_collection,
                index_name=config.db_vector_index_name,
                dimensions=config.embedding_dimension
            )
            print(f"   ✅ Vector search index '{config.db_vector_index_name}' ready")
            print("   ℹ️  Note: New indexes may take a few minutes to become fully active")
        except Exception as e:
            print(f"   ⚠️  Index may already exist: {e}")
        
        # Index documents from the ingested_documents folder
        documents_folder = r"D:\Projects\mcp-rag-agent\data\ingested_documents"
        
        await index_documents_from_folder(
            folder_path=documents_folder,
            mongo_client=mongo_client,
            semantic_search=semantic_search,
            documents_collection=config.db_documents_collection,
            vectors_collection=config.db_vector_collection
        )
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🔌 Disconnecting from MongoDB...")
        mongo_client.disconnect()
        print("✅ Done!")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Index documents from ingested_documents folder into MongoDB"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents and vectors before indexing"
    )
    
    args = parser.parse_args()
    
    # Run the main function with the clear_existing parameter
    asyncio.run(main(clear_existing=args.clear))
