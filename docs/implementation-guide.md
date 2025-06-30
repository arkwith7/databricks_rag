# Implementation Guide

This guide provides detailed implementation instructions for building the RAG (Retrieval-Augmented Generation) application using Databricks and VS Code.

## Table of Contents

1. [Implementation Overview](#implementation-overview)
2. [Core Components](#core-components)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Code Structure](#code-structure)
5. [Advanced Features](#advanced-features)
6. [Testing and Validation](#testing-and-validation)
7. [Deployment Considerations](#deployment-considerations)

## Implementation Overview

### System Architecture
The RAG system consists of several key components:
- **Document Processing Pipeline**: PDF loading and text chunking
- **Embedding Generation**: Converting text to vector representations
- **Vector Search Index**: Storing and querying document embeddings
- **Retrieval System**: Finding relevant document chunks
- **Generation Pipeline**: Creating answers using retrieved context

### Technology Stack
- **Databricks**: Platform for data processing and ML
- **LangChain**: Framework for LLM applications
- **Vector Search**: Databricks vector database
- **Foundation Models**: Pre-trained LLMs and embedding models
- **VS Code**: Development environment

## Core Components

### 1. Document Loader and Chunking

#### PDF Document Loader
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_documents(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Load PDF documents and split into chunks for processing
    
    Args:
        file_path (str): Path to the PDF file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between consecutive chunks
    
    Returns:
        List[Document]: List of document chunks with metadata
    """
    # Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    # Load PDF document
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Configure text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    
    # Split documents into chunks
    documents = text_splitter.split_documents(pages)
    
    # Add source metadata
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "source": os.path.basename(file_path),
            "chunk_id": i,
            "total_chunks": len(documents)
        })
    
    return documents
```

#### Batch Document Processing
```python
def process_multiple_pdfs(pdf_directory, chunk_size=1000, chunk_overlap=200):
    """
    Process multiple PDF files in a directory
    
    Args:
        pdf_directory (str): Directory containing PDF files
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between consecutive chunks
    
    Returns:
        List[Document]: Combined list of all document chunks
    """
    all_documents = []
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        print(f"Processing: {pdf_file}")
        
        try:
            documents = load_and_split_documents(
                file_path, chunk_size, chunk_overlap
            )
            all_documents.extend(documents)
            print(f"  ✅ Processed {len(documents)} chunks")
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file}: {e}")
            continue
    
    print(f"Total documents processed: {len(all_documents)}")
    return all_documents
```

### 2. Embedding Generation

#### Databricks Embedding Model
```python
from langchain_community.embeddings import DatabricksEmbeddings
import numpy as np

class RAGEmbeddingGenerator:
    """Handles embedding generation for RAG system"""
    
    def __init__(self, endpoint_name="databricks-bge-large-en"):
        """
        Initialize embedding generator
        
        Args:
            endpoint_name (str): Name of the Databricks embedding endpoint
        """
        self.endpoint_name = endpoint_name
        self.embeddings = DatabricksEmbeddings(endpoint=endpoint_name)
        self.embedding_dimension = 1024  # BGE Large dimension
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text strings
        
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Validate embedding dimensions
            for embedding in embeddings:
                if len(embedding) != self.embedding_dimension:
                    raise ValueError(f"Unexpected embedding dimension: {len(embedding)}")
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query):
        """
        Generate embedding for a single query
        
        Args:
            query (str): Query text
        
        Returns:
            List[float]: Query embedding vector
        """
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise
```

#### Batch Embedding Processing
```python
def process_documents_with_embeddings(documents, batch_size=32):
    """
    Process documents and generate embeddings in batches
    
    Args:
        documents (List[Document]): List of document chunks
        batch_size (int): Number of documents to process per batch
    
    Returns:
        List[dict]: List of documents with embeddings and metadata
    """
    embedding_generator = RAGEmbeddingGenerator()
    processed_documents = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch]
        
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} documents")
        
        try:
            # Generate embeddings for batch
            embeddings = embedding_generator.generate_embeddings(batch_texts)
            
            # Create processed document objects
            for doc, embedding in zip(batch, embeddings):
                processed_doc = {
                    "id": f"{doc.metadata['source']}_{doc.metadata['chunk_id']}",
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embedding
                }
                processed_documents.append(processed_doc)
        
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    return processed_documents
```

### 3. Vector Search Implementation

#### Vector Search Client Setup
```python
from databricks.vector_search.client import VectorSearchClient
import time

class RAGVectorSearch:
    """Manages vector search operations for RAG system"""
    
    def __init__(self, endpoint_name="rag_endpoint"):
        """
        Initialize vector search client
        
        Args:
            endpoint_name (str): Name of the vector search endpoint
        """
        self.client = VectorSearchClient(disable_notice=True)
        self.endpoint_name = endpoint_name
        self.index_name = None
    
    def create_endpoint(self):
        """Create vector search endpoint if it doesn't exist"""
        try:
            # Check if endpoint already exists
            existing_endpoints = self.client.list_endpoints()
            endpoint_names = [ep.name for ep in existing_endpoints]
            
            if self.endpoint_name in endpoint_names:
                print(f"Endpoint '{self.endpoint_name}' already exists")
                return
            
            # Create new endpoint
            self.client.create_endpoint(
                name=self.endpoint_name,
                endpoint_type="STANDARD"
            )
            
            # Wait for endpoint to be ready
            self._wait_for_endpoint_ready()
            print(f"✅ Endpoint '{self.endpoint_name}' created successfully")
            
        except Exception as e:
            print(f"Error creating endpoint: {e}")
            raise
    
    def create_vector_index(self, index_name, source_table_name=None):
        """
        Create vector index for document storage
        
        Args:
            index_name (str): Name of the vector index
            source_table_name (str): Optional source table name
        """
        self.index_name = index_name
        
        try:
            # Create index configuration
            index_spec = {
                "endpoint_name": self.endpoint_name,
                "index_name": index_name,
                "primary_key": "id",
                "embedding_dimension": 1024,
                "embedding_vector_column": "embedding",
                "schema": {
                    "id": "string",
                    "content": "string",
                    "metadata": "string",
                    "embedding": "array<float>"
                }
            }
            
            # Add source table if provided
            if source_table_name:
                index_spec["source_table_name"] = source_table_name
            
            # Create the index
            self.client.create_vector_index(**index_spec)
            
            # Wait for index to be ready
            self._wait_for_index_ready(index_name)
            print(f"✅ Vector index '{index_name}' created successfully")
            
        except Exception as e:
            print(f"Error creating vector index: {e}")
            raise
    
    def _wait_for_endpoint_ready(self, timeout=300):
        """Wait for endpoint to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                endpoint = self.client.get_endpoint(self.endpoint_name)
                if endpoint.state == "ONLINE":
                    return
                time.sleep(10)
            except:
                time.sleep(10)
                continue
        
        raise TimeoutError(f"Endpoint {self.endpoint_name} not ready after {timeout}s")
    
    def _wait_for_index_ready(self, index_name, timeout=300):
        """Wait for index to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                index = self.client.get_index(index_name)
                if index.status.ready:
                    return
                time.sleep(10)
            except:
                time.sleep(10)
                continue
        
        raise TimeoutError(f"Index {index_name} not ready after {timeout}s")
```

#### Document Indexing
```python
def index_documents(vector_search, documents_with_embeddings, index_name):
    """
    Index documents into vector search
    
    Args:
        vector_search (RAGVectorSearch): Vector search instance
        documents_with_embeddings (List[dict]): Processed documents with embeddings
        index_name (str): Name of the vector index
    """
    import json
    
    # Prepare documents for indexing
    indexed_docs = []
    for doc in documents_with_embeddings:
        indexed_doc = {
            "id": doc["id"],
            "content": doc["content"],
            "metadata": json.dumps(doc["metadata"]),  # Convert dict to JSON string
            "embedding": doc["embedding"]
        }
        indexed_docs.append(indexed_doc)
    
    try:
        # Index documents in batches
        batch_size = 100
        for i in range(0, len(indexed_docs), batch_size):
            batch = indexed_docs[i:i + batch_size]
            
            print(f"Indexing batch {i//batch_size + 1}: {len(batch)} documents")
            
            # Note: The actual indexing method depends on your vector search implementation
            # This is a placeholder for the indexing process
            vector_search.client.upsert(
                index_name=index_name,
                documents=batch
            )
        
        print(f"✅ Successfully indexed {len(indexed_docs)} documents")
        
    except Exception as e:
        print(f"Error indexing documents: {e}")
        raise
```

### 4. Similarity Search Implementation

```python
class RAGRetriever:
    """Handles document retrieval for RAG system"""
    
    def __init__(self, vector_search, embedding_generator, index_name):
        """
        Initialize retriever
        
        Args:
            vector_search (RAGVectorSearch): Vector search instance
            embedding_generator (RAGEmbeddingGenerator): Embedding generator
            index_name (str): Name of the vector index
        """
        self.vector_search = vector_search
        self.embedding_generator = embedding_generator
        self.index_name = index_name
    
    def similarity_search(self, query, top_k=5, score_threshold=0.7):
        """
        Perform similarity search for relevant documents
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            score_threshold (float): Minimum similarity score threshold
        
        Returns:
            List[dict]: List of relevant documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Perform similarity search
            results = self.vector_search.client.similarity_search(
                index_name=self.index_name,
                query_vector=query_embedding,
                columns=["id", "content", "metadata"],
                num_results=top_k
            )
            
            # Filter by score threshold
            filtered_results = []
            for result in results.get("result", {}).get("data_array", []):
                score = result.get("score", 0)
                if score >= score_threshold:
                    filtered_results.append({
                        "id": result["id"],
                        "content": result["content"],
                        "metadata": json.loads(result["metadata"]),
                        "score": score
                    })
            
            return filtered_results
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            raise
    
    def get_relevant_context(self, query, top_k=5, max_context_length=3000):
        """
        Get relevant context for a query with length constraints
        
        Args:
            query (str): Search query
            top_k (int): Number of documents to retrieve
            max_context_length (int): Maximum context length in characters
        
        Returns:
            str: Combined context from relevant documents
        """
        # Get similar documents
        similar_docs = self.similarity_search(query, top_k)
        
        # Combine context while respecting length limits
        context_parts = []
        total_length = 0
        
        for doc in similar_docs:
            content = doc["content"]
            if total_length + len(content) <= max_context_length:
                context_parts.append(content)
                total_length += len(content)
            else:
                # Add partial content if it fits
                remaining_length = max_context_length - total_length
                if remaining_length > 100:  # Only add if meaningful content fits
                    context_parts.append(content[:remaining_length] + "...")
                break
        
        return "\n\n".join(context_parts)
```

### 5. Chat Completion Implementation

```python
from langchain_community.chat_models import ChatDatabricks
from langchain.schema import HumanMessage, SystemMessage

class RAGChatbot:
    """RAG-powered chatbot implementation"""
    
    def __init__(self, retriever, model_name="databricks-dbrx-instruct"):
        """
        Initialize chatbot
        
        Args:
            retriever (RAGRetriever): Document retriever instance
            model_name (str): Name of the chat model to use
        """
        self.retriever = retriever
        self.chat_model = ChatDatabricks(endpoint=model_name)
        self.model_name = model_name
    
    def generate_answer(self, question, context, max_tokens=500, temperature=0.1):
        """
        Generate answer using retrieved context
        
        Args:
            question (str): User's question
            context (str): Retrieved relevant context
            max_tokens (int): Maximum tokens in response
            temperature (float): Model temperature for randomness
        
        Returns:
            str: Generated answer
        """
        # Create prompt template
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use only the information from the context to answer the question. 
        If the context doesn't contain enough information to answer the question, say so clearly.
        Be concise but comprehensive in your response."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Generate response
            response = self.chat_model(
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def chat(self, question, top_k=5, max_context_length=3000):
        """
        Complete RAG chat pipeline
        
        Args:
            question (str): User's question
            top_k (int): Number of documents to retrieve
            max_context_length (int): Maximum context length
        
        Returns:
            dict: Response with answer and metadata
        """
        try:
            # Retrieve relevant context
            context = self.retriever.get_relevant_context(
                question, top_k, max_context_length
            )
            
            if not context:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "context_found": False,
                    "sources": []
                }
            
            # Generate answer
            answer = self.generate_answer(question, context)
            
            # Get source information
            similar_docs = self.retriever.similarity_search(question, top_k)
            sources = [
                {
                    "source": doc["metadata"]["source"],
                    "score": doc["score"]
                }
                for doc in similar_docs
            ]
            
            return {
                "answer": answer,
                "context_found": True,
                "sources": sources,
                "context_length": len(context)
            }
            
        except Exception as e:
            print(f"Error in chat pipeline: {e}")
            return {
                "answer": "I encountered an error while processing your question.",
                "context_found": False,
                "sources": [],
                "error": str(e)
            }
```

## Step-by-Step Implementation

### Phase 1: Environment Setup and Data Preparation

```python
# Step 1: Initialize components
def setup_rag_system():
    """Initialize all RAG system components"""
    
    # 1. Create embedding generator
    embedding_generator = RAGEmbeddingGenerator()
    print("✅ Embedding generator initialized")
    
    # 2. Create vector search instance
    vector_search = RAGVectorSearch()
    vector_search.create_endpoint()
    print("✅ Vector search endpoint ready")
    
    # 3. Process documents
    pdf_directory = "data/pdf"
    documents = process_multiple_pdfs(pdf_directory)
    print(f"✅ Processed {len(documents)} document chunks")
    
    # 4. Generate embeddings
    documents_with_embeddings = process_documents_with_embeddings(documents)
    print(f"✅ Generated embeddings for {len(documents_with_embeddings)} documents")
    
    return embedding_generator, vector_search, documents_with_embeddings
```

### Phase 2: Vector Index Creation and Population

```python
def create_and_populate_index(vector_search, documents_with_embeddings, index_name="rag_index"):
    """Create vector index and populate with documents"""
    
    # 1. Create vector index
    vector_search.create_vector_index(index_name)
    print(f"✅ Vector index '{index_name}' created")
    
    # 2. Index documents
    index_documents(vector_search, documents_with_embeddings, index_name)
    print(f"✅ Documents indexed successfully")
    
    return index_name
```

### Phase 3: RAG Pipeline Setup

```python
def setup_rag_pipeline(embedding_generator, vector_search, index_name):
    """Setup complete RAG pipeline"""
    
    # 1. Create retriever
    retriever = RAGRetriever(vector_search, embedding_generator, index_name)
    print("✅ Retriever initialized")
    
    # 2. Create chatbot
    chatbot = RAGChatbot(retriever)
    print("✅ Chatbot initialized")
    
    return retriever, chatbot
```

### Complete Implementation Example

```python
def main():
    """Complete RAG system implementation"""
    
    print("🚀 Starting RAG system implementation...")
    
    try:
        # Phase 1: Setup and data preparation
        print("\n📁 Phase 1: Environment Setup")
        embedding_generator, vector_search, documents_with_embeddings = setup_rag_system()
        
        # Phase 2: Vector index creation
        print("\n🔍 Phase 2: Vector Index Creation")
        index_name = create_and_populate_index(vector_search, documents_with_embeddings)
        
        # Phase 3: RAG pipeline setup
        print("\n🤖 Phase 3: RAG Pipeline Setup")
        retriever, chatbot = setup_rag_pipeline(embedding_generator, vector_search, index_name)
        
        # Test the system
        print("\n✅ Phase 4: System Testing")
        test_question = "What are the main benefits of using RAG systems?"
        response = chatbot.chat(test_question)
        
        print(f"Test Question: {test_question}")
        print(f"Answer: {response['answer']}")
        print(f"Sources: {len(response['sources'])} documents")
        
        print("\n🎉 RAG system implementation completed successfully!")
        return chatbot
        
    except Exception as e:
        print(f"\n❌ Implementation failed: {e}")
        raise

# Run the implementation
if __name__ == "__main__":
    rag_chatbot = main()
```

## Advanced Features

### 1. Multi-turn Conversation Support

```python
class ConversationalRAG:
    """RAG system with conversation memory"""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.conversation_history = []
    
    def chat_with_history(self, question):
        """Chat with conversation context"""
        
        # Build context from history
        context_prompt = ""
        if self.conversation_history:
            context_prompt = "Previous conversation:\n"
            for turn in self.conversation_history[-3:]:  # Last 3 turns
                context_prompt += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
        
        # Modify question to include context
        enhanced_question = f"{context_prompt}Current question: {question}"
        
        # Get response
        response = self.chatbot.chat(enhanced_question)
        
        # Store in history
        self.conversation_history.append({
            "question": question,
            "answer": response["answer"]
        })
        
        return response
```

### 2. Source Citation and Verification

```python
def generate_answer_with_citations(chatbot, question):
    """Generate answer with detailed source citations"""
    
    # Get similar documents with detailed metadata
    similar_docs = chatbot.retriever.similarity_search(question, top_k=5)
    
    # Build context with source markers
    context_with_sources = ""
    source_map = {}
    
    for i, doc in enumerate(similar_docs):
        source_id = f"[{i+1}]"
        context_with_sources += f"{source_id} {doc['content']}\n\n"
        source_map[source_id] = {
            "source": doc["metadata"]["source"],
            "page": doc["metadata"].get("page", "Unknown"),
            "score": doc["score"]
        }
    
    # Generate answer with source instructions
    system_prompt = """Answer the question using the provided sources. 
    Include source citations in your answer using the format [1], [2], etc. 
    Provide specific information about which sources support each claim."""
    
    # Continue with normal generation...
    return answer, source_map
```

### 3. Query Expansion and Refinement

```python
def expand_query(original_query, chatbot):
    """Expand query for better retrieval"""
    
    expansion_prompt = f"""
    Given the user query: "{original_query}"
    
    Generate 3 alternative phrasings or related questions that would help find relevant information:
    1.
    2. 
    3.
    """
    
    # Generate expansions
    expansions = chatbot.chat_model([HumanMessage(content=expansion_prompt)])
    
    # Extract expansions and perform multiple searches
    # Combine results...
    
    return expanded_results
```

## Testing and Validation

### Unit Tests

```python
import unittest

class TestRAGComponents(unittest.TestCase):
    
    def setUp(self):
        self.embedding_generator = RAGEmbeddingGenerator()
        self.sample_texts = ["This is a test document.", "Another test text."]
    
    def test_embedding_generation(self):
        """Test embedding generation"""
        embeddings = self.embedding_generator.generate_embeddings(self.sample_texts)
        
        self.assertEqual(len(embeddings), len(self.sample_texts))
        self.assertEqual(len(embeddings[0]), 1024)  # BGE Large dimension
    
    def test_document_loading(self):
        """Test document loading and chunking"""
        # Create a test PDF file
        # Test loading and chunking...
        pass
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        # Setup test index with known documents
        # Perform search and validate results...
        pass

if __name__ == "__main__":
    unittest.main()
```

### Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete RAG pipeline"""
    
    # 1. Setup test environment
    test_docs = ["Test document content for RAG system validation."]
    
    # 2. Process documents
    # 3. Create embeddings
    # 4. Index documents
    # 5. Perform test queries
    # 6. Validate responses
    
    assert response["answer"] is not None
    assert response["context_found"] is True
    assert len(response["sources"]) > 0
```

### Performance Benchmarks

```python
import time

def benchmark_rag_performance():
    """Benchmark RAG system performance"""
    
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?"
    ]
    
    results = []
    
    for query in test_queries:
        start_time = time.time()
        response = chatbot.chat(query)
        end_time = time.time()
        
        results.append({
            "query": query,
            "response_time": end_time - start_time,
            "answer_length": len(response["answer"]),
            "sources_found": len(response["sources"])
        })
    
    # Analyze results
    avg_response_time = sum(r["response_time"] for r in results) / len(results)
    print(f"Average response time: {avg_response_time:.2f}s")
```

## Deployment Considerations

### Production Configuration

```python
PRODUCTION_CONFIG = {
    "embedding": {
        "model": "databricks-bge-large-en",
        "batch_size": 64,
        "timeout": 30
    },
    "vector_search": {
        "endpoint_type": "STANDARD",
        "index_type": "DELTA_SYNC"
    },
    "chat": {
        "model": "databricks-dbrx-instruct",
        "max_tokens": 512,
        "temperature": 0.1,
        "timeout": 45
    },
    "retrieval": {
        "top_k": 5,
        "score_threshold": 0.7,
        "max_context_length": 3000
    }
}
```

### Monitoring and Logging

```python
import logging

def setup_logging():
    """Setup comprehensive logging for RAG system"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create specific loggers
    embedding_logger = logging.getLogger('embedding')
    retrieval_logger = logging.getLogger('retrieval')
    generation_logger = logging.getLogger('generation')
    
    return embedding_logger, retrieval_logger, generation_logger
```

### Error Handling and Resilience

```python
from functools import wraps
import time

def retry_on_failure(max_retries=3, delay=1):
    """Decorator for retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    
            raise last_exception
        return wrapper
    return decorator

# Apply to critical functions
@retry_on_failure(max_retries=3)
def robust_similarity_search(retriever, query, **kwargs):
    return retriever.similarity_search(query, **kwargs)
```

This implementation guide provides a comprehensive foundation for building a production-ready RAG system using Databricks and modern development tools. Follow the step-by-step implementation and customize the components based on your specific requirements.
