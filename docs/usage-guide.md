# Usage Guide

This guide provides comprehensive instructions on how to use the RAG (Retrieval-Augmented Generation) application effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Workflow Examples](#workflow-examples)
5. [Best Practices](#best-practices)
6. [Performance Optimization](#performance-optimization)

## Getting Started

### Prerequisites
Before using the RAG application, ensure you have:
- Databricks workspace access
- Valid authentication tokens
- Python environment with required packages
- PDF documents for indexing

### Initial Setup
1. Complete the setup process as described in [Setup Guide](setup-guide.md)
2. Verify your environment configuration
3. Test basic connectivity to Databricks services

## Basic Usage

### 1. Document Ingestion

#### Loading PDF Documents
```python
# Load and process PDF documents
from rag_utils import load_and_split_documents

# Load a single PDF
documents = load_and_split_documents("data/pdf/your_document.pdf")
print(f"Loaded {len(documents)} chunks")

# Load multiple PDFs
import os
all_documents = []
pdf_directory = "data/pdf/"
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        docs = load_and_split_documents(file_path)
        all_documents.extend(docs)
```

#### Customizing Document Chunking
```python
# Custom chunk parameters
documents = load_and_split_documents(
    file_path="data/pdf/document.pdf",
    chunk_size=1500,      # Larger chunks for more context
    chunk_overlap=300     # More overlap for better continuity
)
```

### 2. Creating Embeddings

#### Generate Embeddings for Documents
```python
from databricks.vector_search.client import VectorSearchClient

# Initialize vector search client
vs_client = VectorSearchClient(
    workspace_url=os.environ["DATABRICKS_HOST"],
    personal_access_token=os.environ["DATABRICKS_TOKEN"]
)

# Generate embeddings
document_texts = [doc.page_content for doc in documents]
embeddings = generate_embeddings(
    texts=document_texts,
    model_name="databricks-bge-large-en"
)
```

### 3. Setting Up Vector Index

#### Create Vector Search Endpoint
```python
# Create endpoint (one-time setup)
endpoint_name = "rag_endpoint"
try:
    vs_client.create_endpoint(
        name=endpoint_name,
        endpoint_type="STANDARD"
    )
    print(f"Created endpoint: {endpoint_name}")
except Exception as e:
    print(f"Endpoint may already exist: {e}")
```

#### Create Vector Index
```python
# Create vector index
index_name = "document_index"
vs_client.create_vector_index(
    endpoint_name=endpoint_name,
    index_name=index_name,
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="embedding",
    schema={
        "id": "string",
        "content": "string", 
        "metadata": "string",
        "embedding": "array<float>"
    }
)
```

### 4. Performing RAG Queries

#### Basic Query
```python
from rag_pipeline import perform_rag_query

# Simple question answering
question = "What are the main benefits of using RAG systems?"
answer = perform_rag_query(
    question=question,
    index_name="document_index",
    top_k=3
)
print(f"Answer: {answer}")
```

#### Query with Custom Parameters
```python
# Advanced query with custom settings
answer = perform_rag_query(
    question="Explain the architecture of transformer models",
    index_name="document_index",
    top_k=5,
    model="databricks-dbrx-instruct",
    max_tokens=512,
    temperature=0.1
)
```

## Advanced Features

### 1. Batch Processing

#### Process Multiple Documents
```python
def batch_process_documents(pdf_directory, batch_size=10):
    """Process multiple PDFs in batches"""
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
        
        batch_documents = []
        for pdf_file in batch:
            file_path = os.path.join(pdf_directory, pdf_file)
            docs = load_and_split_documents(file_path)
            batch_documents.extend(docs)
        
        # Process batch
        yield batch_documents

# Usage
for batch_docs in batch_process_documents("data/pdf/", batch_size=5):
    # Process each batch
    embeddings = generate_embeddings([doc.page_content for doc in batch_docs])
    # Index the batch...
```

### 2. Custom Prompt Templates

#### Define Custom Prompts
```python
CUSTOM_PROMPT_TEMPLATE = """
You are an expert assistant specializing in {domain}.
Based on the following context information, provide a detailed answer to the question.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the context
- Include specific examples when available
- If the context doesn't contain enough information, state this clearly
- Structure your response with clear headings and bullet points

Answer:
"""

def custom_rag_query(question, context_docs, domain="general knowledge"):
    """RAG query with custom prompt template"""
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    formatted_prompt = CUSTOM_PROMPT_TEMPLATE.format(
        domain=domain,
        context=context,
        question=question
    )
    
    # Send to chat completion model
    return generate_response(formatted_prompt)
```

### 3. Metadata Filtering

#### Filter by Document Source
```python
def filtered_similarity_search(query, source_filter=None, date_filter=None):
    """Perform similarity search with metadata filters"""
    filters = []
    
    if source_filter:
        filters.append(f"metadata.source = '{source_filter}'")
    
    if date_filter:
        filters.append(f"metadata.date >= '{date_filter}'")
    
    filter_expr = " AND ".join(filters) if filters else None
    
    results = vs_client.similarity_search(
        query_text=query,
        columns=["content", "metadata"],
        filters=filter_expr,
        num_results=5
    )
    return results

# Usage examples
results = filtered_similarity_search(
    query="machine learning algorithms",
    source_filter="research_papers",
    date_filter="2023-01-01"
)
```

### 4. Multi-Step Reasoning

#### Chain Multiple Queries
```python
def multi_step_rag(complex_question, steps):
    """Handle complex questions requiring multiple reasoning steps"""
    results = []
    context = ""
    
    for i, step in enumerate(steps):
        print(f"Step {i+1}: {step}")
        
        # Use accumulated context from previous steps
        step_context = context + f"\n\nCurrent question: {step}"
        
        # Perform RAG for this step
        step_answer = perform_rag_query(
            question=step_context,
            index_name="document_index",
            top_k=3
        )
        
        results.append({
            "step": i+1,
            "question": step,
            "answer": step_answer
        })
        
        # Accumulate context for next step
        context += f"\n\nStep {i+1} Answer: {step_answer}"
    
    # Final synthesis
    final_question = f"Based on the following analysis, answer: {complex_question}\n\nAnalysis: {context}"
    final_answer = perform_rag_query(final_question, "document_index", top_k=5)
    
    return {
        "steps": results,
        "final_answer": final_answer
    }

# Example usage
complex_question = "How do transformer architectures improve upon RNN limitations in NLP tasks?"
steps = [
    "What are the main limitations of RNN architectures?",
    "What are the key components of transformer architecture?", 
    "How do transformers address RNN limitations?"
]

result = multi_step_rag(complex_question, steps)
```

## Workflow Examples

### Example 1: Research Assistant Workflow

```python
def research_assistant_workflow(research_topic, pdf_sources):
    """Complete workflow for research assistance"""
    
    print(f"🔍 Researching: {research_topic}")
    
    # Step 1: Process documents
    print("📄 Processing documents...")
    all_documents = []
    for pdf_path in pdf_sources:
        docs = load_and_split_documents(pdf_path)
        all_documents.extend(docs)
    
    # Step 2: Create embeddings and index
    print("🔗 Creating vector index...")
    index_name = f"research_{research_topic.replace(' ', '_')}"
    setup_vector_index(all_documents, index_name)
    
    # Step 3: Generate research questions
    research_questions = [
        f"What is {research_topic}?",
        f"What are the current challenges in {research_topic}?",
        f"What are recent developments in {research_topic}?",
        f"What are practical applications of {research_topic}?"
    ]
    
    # Step 4: Answer research questions
    research_results = {}
    for question in research_questions:
        print(f"❓ Answering: {question}")
        answer = perform_rag_query(question, index_name, top_k=5)
        research_results[question] = answer
    
    # Step 5: Generate summary report
    summary_prompt = f"""
    Create a comprehensive research summary on {research_topic} based on the following findings:
    
    {chr(10).join([f"Q: {q}\nA: {a}\n" for q, a in research_results.items()])}
    
    Structure the summary with:
    1. Overview
    2. Key Concepts
    3. Current Challenges
    4. Recent Developments
    5. Applications
    6. Future Directions
    """
    
    summary = generate_response(summary_prompt)
    
    return {
        "topic": research_topic,
        "detailed_answers": research_results,
        "summary": summary
    }

# Usage
result = research_assistant_workflow(
    research_topic="Large Language Models",
    pdf_sources=["data/pdf/llm_paper1.pdf", "data/pdf/llm_paper2.pdf"]
)
```

### Example 2: Document Q&A System

```python
class DocumentQASystem:
    def __init__(self, document_path, index_name=None):
        self.document_path = document_path
        self.index_name = index_name or f"qa_{os.path.basename(document_path)}"
        self.setup_complete = False
    
    def setup(self):
        """One-time setup for the document"""
        print("Setting up document Q&A system...")
        
        # Process document
        self.documents = load_and_split_documents(self.document_path)
        
        # Create index
        setup_vector_index(self.documents, self.index_name)
        self.setup_complete = True
        
        print(f"✅ Setup complete. Indexed {len(self.documents)} chunks.")
    
    def ask(self, question, context_size=3):
        """Ask a question about the document"""
        if not self.setup_complete:
            self.setup()
        
        answer = perform_rag_query(
            question=question,
            index_name=self.index_name,
            top_k=context_size
        )
        
        return answer
    
    def interactive_session(self):
        """Start an interactive Q&A session"""
        if not self.setup_complete:
            self.setup()
        
        print(f"📚 Interactive Q&A for: {os.path.basename(self.document_path)}")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\n❓ Your question: ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question.strip():
                answer = self.ask(question)
                print(f"\n💡 Answer: {answer}")

# Usage
qa_system = DocumentQASystem("data/pdf/technical_manual.pdf")
qa_system.interactive_session()
```

## Best Practices

### 1. Document Preparation
- **Clean PDFs**: Ensure PDFs are text-searchable, not scanned images
- **Consistent Formatting**: Use documents with similar formatting for better chunking
- **Metadata**: Include relevant metadata (author, date, source) for filtering

### 2. Chunking Strategy
```python
# Recommended chunking parameters for different document types
CHUNK_CONFIGS = {
    "research_papers": {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "separators": ["\n\n", "\n", ". ", " "]
    },
    "technical_docs": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", ". ", " "]
    },
    "books": {
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "separators": ["\n\n", "\n", ". ", " "]
    }
}

# Usage
config = CHUNK_CONFIGS["research_papers"]
documents = load_and_split_documents(
    file_path="paper.pdf",
    **config
)
```

### 3. Query Optimization
- **Specific Questions**: Ask specific, focused questions for better results
- **Context Size**: Adjust `top_k` based on question complexity (3-5 for simple, 5-10 for complex)
- **Iterative Refinement**: Refine questions based on initial results

### 4. Model Selection
```python
# Model recommendations by use case
MODEL_CONFIGS = {
    "factual_qa": {
        "model": "databricks-dbrx-instruct",
        "temperature": 0.1,
        "max_tokens": 300
    },
    "creative_writing": {
        "model": "databricks-dbrx-instruct", 
        "temperature": 0.7,
        "max_tokens": 500
    },
    "technical_analysis": {
        "model": "databricks-dbrx-instruct",
        "temperature": 0.2,
        "max_tokens": 600
    }
}
```

## Performance Optimization

### 1. Caching Strategies
```python
import functools
from typing import List, Tuple

@functools.lru_cache(maxsize=100)
def cached_similarity_search(query: str, index_name: str, top_k: int) -> Tuple:
    """Cache similarity search results"""
    results = similarity_search(query, index_name, top_k)
    return tuple(results)  # Convert to tuple for caching

@functools.lru_cache(maxsize=50)
def cached_embeddings(text: str, model: str) -> Tuple:
    """Cache embedding generation"""
    embedding = generate_embeddings([text], model)[0]
    return tuple(embedding)
```

### 2. Batch Processing
```python
def batch_embed_documents(documents: List[str], batch_size: int = 32):
    """Process documents in batches for better performance"""
    embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_embeddings = generate_embeddings(batch)
        embeddings.extend(batch_embeddings)
        
        # Optional: Add delay to respect rate limits
        time.sleep(0.1)
    
    return embeddings
```

### 3. Monitoring and Logging
```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_rag_query(question: str, **kwargs):
    """RAG query with performance monitoring"""
    start_time = time.time()
    
    try:
        # Perform RAG query
        result = perform_rag_query(question, **kwargs)
        
        # Log success
        duration = time.time() - start_time
        logger.info(f"RAG query completed in {duration:.2f}s: {question[:50]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise
```

## Common Usage Patterns

### Pattern 1: Document Comparison
```python
def compare_documents(question: str, doc_sources: List[str]):
    """Compare answers across different document sources"""
    results = {}
    
    for source in doc_sources:
        index_name = f"index_{source.replace('.', '_')}"
        answer = perform_rag_query(question, index_name)
        results[source] = answer
    
    return results
```

### Pattern 2: Progressive Context Building
```python
def progressive_context_query(questions: List[str], index_name: str):
    """Build context progressively through multiple questions"""
    context = ""
    results = []
    
    for question in questions:
        # Include previous context
        full_question = f"Context: {context}\n\nQuestion: {question}"
        answer = perform_rag_query(full_question, index_name)
        
        results.append({"question": question, "answer": answer})
        context += f" {answer}"
    
    return results
```

This usage guide provides comprehensive examples and best practices for effectively using the RAG application. For specific technical details, refer to the [API Reference](api-reference.md) and [Implementation Guide](implementation-guide.md).
