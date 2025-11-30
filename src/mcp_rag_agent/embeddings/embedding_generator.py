"""Embedding generator using OpenAI or other providers."""

import os
from typing import Optional

import openai


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or compatible APIs."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        base_url: Optional[str] = None
    ):
        """Initialize embedding generator.
        
        Args:
            api_key: API key for the embedding service.
            model: Model name to use for embeddings.
            dimensions: Number of dimensions for embeddings.
            base_url: Optional base URL for API (for compatible services).
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model
        self._dimensions = dimensions
        
        client_kwargs = {"api_key": self._api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self._client = openai.AsyncOpenAI(**client_kwargs)
    
    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self._dimensions
    
    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model
    
    async def generate(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            List of floats representing the embedding.
        """
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions
        )
        return response.data[0].embedding
    
    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embeddings.
        """
        if not texts:
            return []
        
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions
        )
        
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

async def main():
    """Main function to demonstrate EmbeddingGenerator usage."""
    import asyncio
    from mcp_rag_agent.core.config import config
    
    print("=" * 60)
    print("Embedding Generator Demo")
    print("=" * 60)
    
    # Initialize embedding generator
    print("\n🤖 Initializing embedding generator...")
    generator = EmbeddingGenerator(
        api_key=config.model_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimension
    )
    
    print(f"   Model: {generator.model}")
    print(f"   Dimensions: {generator.dimensions}")
    
    try:
        # Generate single embedding
        print("\n📝 Generating single embedding...")
        text = "Artificial intelligence is transforming the world of technology."
        print(f"   Text: \"{text}\"")
        
        embedding = await generator.generate(text)
        print(f"   ✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Generate batch embeddings
        print("\n📚 Generating batch embeddings...")
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Cloud computing enables scalable infrastructure."
        ]
        
        print(f"   Processing {len(texts)} texts:")
        for i, t in enumerate(texts, 1):
            print(f"   {i}. \"{t}\"")
        
        embeddings = await generator.generate_batch(texts)
        print(f"   ✅ Generated {len(embeddings)} embeddings")
        
        for i, emb in enumerate(embeddings, 1):
            print(f"   Embedding {i}: {len(emb)} dimensions, first 3 values: {emb[:3]}")
        
        # Calculate similarity between first two embeddings (simple dot product)
        print("\n🔍 Calculating similarity...")
        if len(embeddings) >= 2:
            dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
            print(f"   Dot product similarity between text 1 and 2: {dot_product:.4f}")
            print("   (Higher values indicate more similarity)")
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
