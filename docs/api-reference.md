# API Reference

This document provides detailed API reference for the RAG (Retrieval-Augmented Generation) application.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Key Functions](#key-functions)
3. [Configuration Parameters](#configuration-parameters)
4. [Response Formats](#response-formats)
5. [Error Handling](#error-handling)

## Core Classes

### VectorSearchClient
Main client for interacting with Databricks Vector Search.

```python
class VectorSearchClient:
    def __init__(self, workspace_url: str, token: str)
    def create_endpoint(self, endpoint_name: str)
    def create_vector_index(self, endpoint_name: str, index_name: str, primary_key: str, embedding_dimension: int)
```

**Methods:**
- `create_endpoint()`: Creates a vector search endpoint
- `create_vector_index()`: Creates a vector index for document embeddings
- `similarity_search()`: Performs similarity search on indexed documents

### ChatCompletionsClient
Client for chat completions using foundation models.

```python
class ChatCompletionsClient:
    def __init__(self, base_url: str, api_key: str)
    def complete(self, messages: List[Dict], model: str, max_tokens: int)
```

## Key Functions

### Document Processing

#### `load_and_split_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200)`
Loads PDF documents and splits them into chunks.

**Parameters:**
- `file_path` (str): Path to the PDF file
- `chunk_size` (int): Size of each text chunk (default: 1000)
- `chunk_overlap` (int): Overlap between chunks (default: 200)

**Returns:**
- `List[Document]`: List of document chunks

**Example:**
```python
documents = load_and_split_documents("data/pdf/document.pdf")
```

#### `generate_embeddings(texts: List[str], model_name: str = "databricks-bge-large-en")`
Generates embeddings for text chunks.

**Parameters:**
- `texts` (List[str]): List of text strings to embed
- `model_name` (str): Name of the embedding model

**Returns:**
- `List[List[float]]`: List of embedding vectors

### Vector Search Operations

#### `create_vector_index(endpoint_name: str, index_name: str, source_table: str)`
Creates a vector search index.

**Parameters:**
- `endpoint_name` (str): Name of the vector search endpoint
- `index_name` (str): Name of the index to create
- `source_table` (str): Source table containing documents and embeddings

**Returns:**
- `dict`: Index creation response

#### `similarity_search(query: str, index_name: str, k: int = 5)`
Performs similarity search on the vector index.

**Parameters:**
- `query` (str): Search query text
- `index_name` (str): Name of the vector index
- `k` (int): Number of similar documents to return

**Returns:**
- `List[dict]`: List of similar documents with scores

### RAG Pipeline

#### `rag_query(question: str, context_documents: List[str], model: str = "databricks-dbrx-instruct")`
Performs RAG query with retrieved context.

**Parameters:**
- `question` (str): User's question
- `context_documents` (List[str]): Retrieved relevant documents
- `model` (str): Language model to use for generation

**Returns:**
- `str`: Generated answer

**Example:**
```python
answer = rag_query(
    question="What are the key components of a RAG system?",
    context_documents=retrieved_docs,
    model="databricks-dbrx-instruct"
)
```

## Configuration Parameters

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DATABRICKS_HOST` | Databricks workspace URL | Yes | - |
| `DATABRICKS_TOKEN` | Databricks access token | Yes | - |
| `VECTOR_SEARCH_ENDPOINT` | Vector search endpoint name | Yes | - |
| `VECTOR_INDEX_NAME` | Vector index name | Yes | - |
| `EMBEDDING_MODEL` | Embedding model name | No | `databricks-bge-large-en` |
| `CHAT_MODEL` | Chat completion model | No | `databricks-dbrx-instruct` |

### Model Parameters

#### Embedding Model Parameters
```python
embedding_config = {
    "model": "databricks-bge-large-en",
    "dimension": 1024,
    "max_sequence_length": 512
}
```

#### Chat Model Parameters
```python
chat_config = {
    "model": "databricks-dbrx-instruct",
    "max_tokens": 500,
    "temperature": 0.1,
    "top_p": 0.95
}
```

### Document Processing Parameters
```python
chunk_config = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", " ", ""]
}
```

## Response Formats

### Similarity Search Response
```json
{
    "results": [
        {
            "document_id": "doc_123",
            "content": "Document text content...",
            "metadata": {
                "source": "file.pdf",
                "page": 1
            },
            "score": 0.85
        }
    ],
    "total_results": 5
}
```

### RAG Query Response
```json
{
    "answer": "Generated answer text...",
    "sources": [
        {
            "document_id": "doc_123",
            "relevance_score": 0.85,
            "source": "file.pdf"
        }
    ],
    "model_used": "databricks-dbrx-instruct",
    "processing_time": 1.23
}
```

### Index Creation Response
```json
{
    "index_name": "my_vector_index",
    "status": "ONLINE",
    "endpoint_name": "my_endpoint",
    "dimension": 1024,
    "total_documents": 1500
}
```

## Error Handling

### Common Error Codes

#### Authentication Errors
- **401 Unauthorized**: Invalid or missing Databricks token
- **403 Forbidden**: Insufficient permissions for the operation

#### Vector Search Errors
- **404 Not Found**: Vector index or endpoint does not exist
- **400 Bad Request**: Invalid query parameters or malformed request
- **429 Too Many Requests**: Rate limit exceeded

#### Model Errors
- **500 Internal Server Error**: Model inference failure
- **503 Service Unavailable**: Model temporarily unavailable

### Error Response Format
```json
{
    "error": {
        "code": "INVALID_REQUEST",
        "message": "The request is invalid",
        "details": "Specific error details here"
    },
    "request_id": "req_123456"
}
```

### Exception Classes

#### `VectorSearchError`
Raised when vector search operations fail.
```python
class VectorSearchError(Exception):
    def __init__(self, message: str, error_code: str = None)
```

#### `EmbeddingError`
Raised when embedding generation fails.
```python
class EmbeddingError(Exception):
    def __init__(self, message: str, model_name: str = None)
```

#### `RAGError`
Raised when RAG pipeline operations fail.
```python
class RAGError(Exception):
    def __init__(self, message: str, stage: str = None)
```

## Usage Examples

### Basic RAG Pipeline
```python
# Initialize clients
vs_client = VectorSearchClient(workspace_url, token)
chat_client = ChatCompletionsClient(base_url, api_key)

# Process documents
documents = load_and_split_documents("document.pdf")
embeddings = generate_embeddings([doc.content for doc in documents])

# Create index
create_vector_index("my_endpoint", "my_index", "source_table")

# Perform RAG query
query = "What is machine learning?"
similar_docs = similarity_search(query, "my_index", k=3)
answer = rag_query(query, similar_docs)
```

### Advanced Configuration
```python
# Custom configuration
config = {
    "embedding": {
        "model": "databricks-bge-large-en",
        "batch_size": 32
    },
    "retrieval": {
        "top_k": 5,
        "score_threshold": 0.7
    },
    "generation": {
        "model": "databricks-dbrx-instruct",
        "max_tokens": 512,
        "temperature": 0.1
    }
}

# Initialize with config
rag_system = RAGPipeline(config)
```

## Rate Limits

- **Embedding API**: 100 requests per minute
- **Chat Completions**: 50 requests per minute
- **Vector Search**: 1000 queries per minute

## Versioning

Current API version: `v1`

For version-specific changes, see the [CHANGELOG.md](../CHANGELOG.md).
