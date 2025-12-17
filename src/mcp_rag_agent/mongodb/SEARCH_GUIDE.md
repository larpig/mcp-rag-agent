# MongoDB Search Implementation Guide

This guide explains the three search methods available in the MongoDB RAG Agent: Vector Search, Text Search, and Hybrid Search.

## Table of Contents
- [Overview](#overview)
- [Vector Search](#vector-search)
- [Text Search](#text-search)
- [Hybrid Search](#hybrid-search)
- [Comparison & Use Cases](#comparison--use-cases)
- [Best Practices](#best-practices)

---

## Overview

The MongoDB RAG Agent provides three complementary search approaches:

| Search Type | Method | Best For | Requires |
|-------------|--------|----------|----------|
| **Vector Search** | Semantic similarity | Conceptual queries, synonyms | Vector embeddings |
| **Text Search** | Keyword matching | Exact terms, technical terms | Text index |
| **Hybrid Search** | Combined (RRF) | Best overall results | Both indexes |

---

## Vector Search

### What It Does
Vector search finds documents based on **semantic similarity** using embedding vectors. It can find relevant documents even when they don't contain the exact query terms.

### How It Works
1. Converts query text to embedding vector (1536 dimensions for text-embedding-3-small)
2. Uses cosine similarity to find nearest vectors in MongoDB Atlas
3. Returns documents ranked by similarity score (0-1 scale)

### Key Features
- ✅ **Semantic understanding**: Finds conceptually related content
- ✅ **Synonym handling**: "AI" matches "artificial intelligence" 
- ✅ **No exact matches needed**: Finds relevant content without keywords
- ✅ **Cross-lingual**: Can work across languages with multilingual models
- ❌ **May miss specific terms**: Doesn't guarantee keyword presence

### Example

```python
from mcp_rag_agent.mongodb.client import MongoDBClient
from mcp_rag_agent.embeddings.semantic_search import SemanticSearch
from mcp_rag_agent.embeddings.embedding_generator import EmbeddingGenerator

# Setup
client = MongoDBClient(uri="mongodb://...", database_name="mydb")
embedder = EmbeddingGenerator(api_key="...", model="text-embedding-3-small")
search = SemanticSearch(client, embedder)

# Search
results = await search.search(
    query="What is artificial intelligence?",
    limit=5
)

# Results might include documents about:
# - "Machine learning and AI"  ✓ (high similarity)
# - "Neural networks and deep learning"  ✓ (conceptually related)
# - "Python programming"  ✓ (often associated with AI)
# - "Cloud computing"  ✗ (low similarity)
```

### When to Use
- General knowledge questions
- Conceptual queries
- When synonyms/related terms are acceptable
- Cross-domain search (e.g., "how to improve code quality" matches various topics)

---

## Text Search

### What It Does
Text search finds documents containing **specific keywords** from your query using MongoDB's full-text search capabilities.

### How It Works
1. Processes query: removes stop words, applies stemming
2. Searches text index for matching terms
3. Returns documents containing query terms, ranked by relevance score

### Key Features Built-In

#### 1. **Stop Word Removal** ✅
Automatically removes common words like "the", "is", "what", "a":
```
Query: "What is machine learning?"
Processed: "machine learning"  (removed: "what", "is")
```

#### 2. **Stemming/Lemmatization** ✅
Matches word roots automatically:
```
"running" matches "run", "runs", "ran"
"machines" matches "machine"
"learning" matches "learn", "learned"
```

#### 3. **Case-Insensitive** ✅
```
"AI" matches "ai", "Ai", "AI"
```

### Features NOT Supported

#### ❌ Fuzzy Matching (Typo Tolerance)
```
"machne" does NOT match "machine"
"lerning" does NOT match "learning"
```
**Solution**: Use MongoDB Atlas Search with `use_atlas_search=True`

#### ❌ Wildcard/Partial Matches
```
Cannot search: "mach*" or "learn?"
```

#### ❌ Automatic Synonym Expansion
```
"AI" does NOT automatically match "artificial intelligence"
"ML" does NOT automatically match "machine learning"
```
**Solution**: Use hybrid search (vector component handles synonyms)

### Example

```python
from mcp_rag_agent.mongodb.client import MongoDBClient

client = MongoDBClient(uri="mongodb://...", database_name="mydb")

# Standard text search (self-hosted MongoDB)
results = client.text_search(
    collection_name="documents",
    index_name="text_index",
    query_text="machine learning artificial intelligence",
    limit=5,
    use_atlas_search=False  # Default: standard text search
)

# Results ONLY include documents containing:
# - "machine" or "learning" or "artificial" or "intelligence"
# - With stemming: "machines", "learned", "artificially", etc.
```

### Text Search Behavior Examples

**Query:** "What is machine learning and AI?"

**Processing:**
```
1. Stop word removal: "machine learning AI"
2. Stemming applied: "machin learn ai"
3. Search for: documents containing any of these terms
```

**Sample Results:**

| Document | Contains Terms | Match? | Score |
|----------|---------------|--------|-------|
| "Machine learning is a subset of AI..." | machine, learning, AI | ✅ Yes | 2.45 |
| "Artificial intelligence and ML..." | AI (variant) | ✅ Yes | 1.20 |
| "Python programming language..." | none | ❌ No | - |
| "Cloud computing services..." | none | ❌ No | - |

### When to Use
- Searching for specific technical terms
- Finding exact product names, codes, or identifiers
- When precision is more important than recall
- Compliance/legal searches requiring exact terms

---

## Hybrid Search

### What It Does
Hybrid search **combines vector and text search** using Reciprocal Rank Fusion (RRF) to provide the best of both worlds: semantic understanding with keyword precision.

### How It Works

1. **Parallel Execution**:
   ```
   Vector Search: Find top N semantically similar documents
   Text Search:   Find top N keyword-matching documents
   ```

2. **Reciprocal Rank Fusion (RRF)**:
   ```python
   For each document:
       rrf_score = vector_weight / (k + vector_rank) + text_weight / (k + text_rank)
   
   Where:
       k = 60 (default constant, reduces high-rank impact)
       vector_rank = position in vector results (1, 2, 3, ...)
       text_rank = position in text results (1, 2, 3, ...)
       vector_weight = 0.7 (default, adjustable)
       text_weight = 0.3 (default, adjustable)
   ```

3. **Score Combination**:
   - Documents in both result sets get combined scores
   - Documents in only one result set get partial scores
   - Final ranking by RRF score (highest first)

### Why RRF (Not Simple Score Addition)?

RRF is superior to simple score normalization because:
- ✅ **Scale-independent**: Works regardless of score magnitudes
- ✅ **Robust**: Not affected by outlier scores
- ✅ **Rank-based**: Focuses on relative positions, not absolute scores
- ✅ **Industry standard**: Used by Elasticsearch, Weaviate, Vespa

### Example

```python
from mcp_rag_agent.embeddings.hybrid_search import HybridSearch

hybrid = HybridSearch(
    mongo_client=client,
    embedding_generator=embedder
)

results = await hybrid.search(
    query="What is machine learning and AI?",
    limit=5,
    vector_weight=0.7,  # Favor semantic similarity
    text_weight=0.3,    # Boost keyword matches
    rrf_k=60           # Standard RRF constant
)

# Each result includes:
for doc in results:
    print(f"RRF Score: {doc['rrf_score']:.4f}")
    print(f"Vector Rank: {doc['vector_rank']} (similarity: {doc.get('vector_score', 'N/A')})")
    print(f"Text Rank: {doc['text_rank']} (relevance: {doc.get('text_score', 'N/A')})")
    print(f"Content: {doc['content'][:100]}...")
```

### Hybrid Search Example Walkthrough

**Query:** "machine learning and AI"

**Step 1: Vector Search Results**
```
Rank 1: "Artificial intelligence and ML revolutionize..." (score: 0.89)
Rank 2: "Machine learning models learn from data..." (score: 0.85)
Rank 3: "Python programming for data science..." (score: 0.72)
```

**Step 2: Text Search Results**
```
Rank 1: "Machine learning is a subset of AI..." (score: 2.45)
Rank 2: "AI and machine learning applications..." (score: 2.20)
```

**Step 3: RRF Calculation** (k=60, vector_weight=0.7, text_weight=0.3)

| Document | Vector Rank | Text Rank | RRF Calculation | Final Score |
|----------|-------------|-----------|-----------------|-------------|
| "AI and ML revolutionize..." | 1 | 2 | 0.7/(60+1) + 0.3/(60+2) = 0.0115 + 0.0048 | **0.0163** ⭐ |
| "Machine learning is subset..." | 2 | 1 | 0.7/(60+2) + 0.3/(60+1) = 0.0113 + 0.0049 | **0.0162** |
| "Python for data science..." | 3 | None | 0.7/(60+3) + 0 = 0.0111 | **0.0111** |

**Step 4: Final Ranked Results**
```
1. "AI and ML revolutionize..." (RRF: 0.0163) - Found in BOTH searches
2. "Machine learning is subset..." (RRF: 0.0162) - Found in BOTH searches  
3. "Python for data science..." (RRF: 0.0111) - Found in vector only
```

### Tuning Weights

Adjust `vector_weight` and `text_weight` based on your use case:

```python
# Favor semantic similarity (exploratory search)
results = await hybrid.search(
    query="innovative AI solutions",
    vector_weight=0.8,  # Higher weight on concepts
    text_weight=0.2     # Lower weight on keywords
)

# Favor exact keywords (precise search)
results = await hybrid.search(
    query="Python API documentation",
    vector_weight=0.5,  # Equal weight
    text_weight=0.5
)

# Strong keyword preference (legal/compliance)
results = await hybrid.search(
    query="GDPR Article 17 data erasure",
    vector_weight=0.3,  # Lower semantic weight
    text_weight=0.7     # Higher keyword weight
)
```

### When to Use
- **Default choice for most applications** ✅
- General question answering
- Document retrieval systems
- When you want both precision and recall
- Production RAG systems

---

## Comparison & Use Cases

### Search Method Comparison

| Scenario | Vector Search | Text Search | Hybrid Search |
|----------|--------------|-------------|---------------|
| "What is AI?" | ⭐⭐⭐ Best | ⭐⭐ Good | ⭐⭐⭐ Best |
| "Find GDPR Article 17" | ⭐ Poor | ⭐⭐⭐ Best | ⭐⭐⭐ Best |
| "How to improve code quality?" | ⭐⭐⭐ Best | ⭐ Poor | ⭐⭐⭐ Best |
| "Python documentation" | ⭐⭐ Good | ⭐⭐⭐ Best | ⭐⭐⭐ Best |
| Exploratory research | ⭐⭐⭐ Best | ⭐ Poor | ⭐⭐ Good |
| Exact term search | ⭐ Poor | ⭐⭐⭐ Best | ⭐⭐ Good |

### Performance Characteristics

| Metric | Vector Search | Text Search | Hybrid Search |
|--------|--------------|-------------|---------------|
| Latency | ~100-200ms | ~10-50ms | ~100-250ms |
| Precision | Medium | High | High |
| Recall | High | Medium | Very High |
| Storage | High (vectors) | Low (index) | High |
| Setup Complexity | Medium | Low | Medium |

---

## Best Practices

### 1. Choose the Right Search Method

```python
# Use vector search for:
results = await semantic_search.search("concepts related to neural networks")

# Use text search for:
results = client.text_search("specific-product-code-XYZ-123")

# Use hybrid search for (RECOMMENDED DEFAULT):
results = await hybrid_search.search("general question about anything")
```

### 2. Create Proper Indexes

```python
# For hybrid search, create BOTH indexes
hybrid.setup_indexes(
    collection_name="documents",
    vector_index_name="vector_idx",
    text_index_name="text_idx",
    text_fields=["content", "title"],  # Index multiple fields
    text_field_weights={"title": 10, "content": 1},  # Title more important
    dimensions=1536  # text-embedding-3-small
)
```

### 3. Optimize Text Queries

```python
# ❌ BAD: Too many stop words
query = "What is the best way to do machine learning?"

# ✅ GOOD: Focused keywords
query = "machine learning best practices"

# ✅ BETTER: Include specific terms
query = "machine learning model optimization techniques"
```

### 4. Monitor and Adjust

```python
# Log search results for analysis
results = await hybrid.search(query, limit=10)

for i, doc in enumerate(results, 1):
    logger.info(f"Rank {i}: RRF={doc['rrf_score']:.4f}, "
                f"Vector={doc['vector_rank']}, Text={doc['text_rank']}")
    
# Adjust weights based on which component performs better
if text_matches_are_better:
    text_weight = 0.5  # Increase from 0.3
    vector_weight = 0.5  # Decrease from 0.7
```

### 5. Handle Edge Cases

```python
# Empty results from text search (no keyword matches)
results = await hybrid.search(query="very specific unusual terminology")
# Hybrid search will still return vector results

# Documents with minimal text
# Ensure meaningful content in indexed fields
document = {
    "content": "Detailed explanation here...",  # Good
    # Not: "Document"  # Too short for meaningful search
}
```

### 6. Language Considerations

```python
# MongoDB text search supports multiple languages
# Set language at index level or document level
db.documents.create_index(
    [("content", "text")],
    default_language="english",  # or "spanish", "french", etc.
    language_override="language"  # field name for per-document language
)

# Document with language specification
document = {
    "content": "Contenido en español",
    "language": "spanish"
}
```

---

## Conclusion

**For most use cases, use Hybrid Search** - it provides the best balance of semantic understanding and keyword precision through the battle-tested RRF algorithm.

- **Vector Search**: When you need pure semantic similarity
- **Text Search**: When you need exact keyword matching  
- **Hybrid Search**: When you want the best overall results (recommended default)

The current implementation is production-ready and follows industry best practices used by major search platforms like Elasticsearch, Weaviate, and Vespa.
