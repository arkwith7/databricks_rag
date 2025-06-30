# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when working with the RAG (Retrieval-Augmented Generation) application.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Authentication Problems](#authentication-problems)
3. [Vector Search Issues](#vector-search-issues)
4. [Model and Inference Problems](#model-and-inference-problems)
5. [Performance Issues](#performance-issues)
6. [Data Processing Problems](#data-processing-problems)
7. [Environment and Setup Issues](#environment-and-setup-issues)
8. [Debugging Tools](#debugging-tools)

## Common Issues

### Issue: "Module not found" errors
**Symptoms:**
```
ModuleNotFoundError: No module named 'databricks'
```

**Solutions:**
1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify Python environment:
   ```python
   import sys
   print(sys.path)
   ```

3. Check virtual environment activation:
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

### Issue: Notebook kernel connection problems
**Symptoms:**
- Kernel fails to start
- "Dead kernel" messages
- Code cells don't execute

**Solutions:**
1. Restart the kernel:
   - In Jupyter: Kernel → Restart
   - In VS Code: Command Palette → "Restart Kernel"

2. Check kernel availability:
   ```bash
   jupyter kernelspec list
   ```

3. Install jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=rag_env
   ```

## Authentication Problems

### Issue: Databricks authentication failure
**Symptoms:**
```
Error: Authentication failed
HTTPError: 401 Client Error: Unauthorized
```

**Solutions:**
1. **Check environment variables:**
   ```python
   import os
   print(f"DATABRICKS_HOST: {os.environ.get('DATABRICKS_HOST')}")
   print(f"DATABRICKS_TOKEN: {'***' if os.environ.get('DATABRICKS_TOKEN') else 'NOT SET'}")
   ```

2. **Verify token validity:**
   ```bash
   curl -H "Authorization: Bearer $DATABRICKS_TOKEN" \
        $DATABRICKS_HOST/api/2.0/clusters/list
   ```

3. **Generate new token:**
   - Go to Databricks workspace
   - User Settings → Access Tokens
   - Generate new token
   - Update environment variables

4. **Check token permissions:**
   - Ensure token has necessary workspace permissions
   - Verify cluster access permissions

### Issue: Token expiration
**Symptoms:**
- Authentication worked before but now fails
- "Token expired" error messages

**Solutions:**
1. **Check token expiration:**
   ```python
   import requests
   import os
   
   headers = {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}
   response = requests.get(f"{os.environ['DATABRICKS_HOST']}/api/2.0/token/list", headers=headers)
   print(response.json())
   ```

2. **Generate new token with longer expiration**

3. **Implement token refresh logic:**
   ```python
   def refresh_token_if_needed():
       try:
           # Test current token
           test_auth()
           return True
       except AuthenticationError:
           # Refresh token logic here
           return False
   ```

## Vector Search Issues

### Issue: Vector index not found
**Symptoms:**
```
Error: Vector index 'my_index' not found
404 Not Found: Index does not exist
```

**Solutions:**
1. **List available indexes:**
   ```python
   from databricks.vector_search.client import VectorSearchClient
   
   vs_client = VectorSearchClient()
   indexes = vs_client.list_indexes()
   print("Available indexes:", indexes)
   ```

2. **Check index status:**
   ```python
   try:
       index_info = vs_client.get_index("my_index")
       print(f"Index status: {index_info.status}")
   except Exception as e:
       print(f"Index error: {e}")
   ```

3. **Create missing index:**
   ```python
   vs_client.create_vector_index(
       endpoint_name="my_endpoint",
       index_name="my_index",
       # ... other parameters
   )
   ```

### Issue: Endpoint connection problems
**Symptoms:**
```
ConnectionError: Failed to connect to vector search endpoint
TimeoutError: Request timed out
```

**Solutions:**
1. **Check endpoint status:**
   ```python
   endpoints = vs_client.list_endpoints()
   for endpoint in endpoints:
       print(f"Endpoint: {endpoint.name}, Status: {endpoint.state}")
   ```

2. **Verify endpoint configuration:**
   ```python
   endpoint_info = vs_client.get_endpoint("my_endpoint")
   print(f"Endpoint URL: {endpoint_info.endpoint_url}")
   print(f"Endpoint type: {endpoint_info.endpoint_type}")
   ```

3. **Test connectivity:**
   ```python
   import requests
   
   def test_endpoint_connectivity(endpoint_url):
       try:
           response = requests.get(f"{endpoint_url}/health", timeout=10)
           return response.status_code == 200
       except requests.exceptions.RequestException as e:
           print(f"Connectivity test failed: {e}")
           return False
   ```

### Issue: Embedding dimension mismatch
**Symptoms:**
```
ValueError: Embedding dimension mismatch
Expected 1024, got 768
```

**Solutions:**
1. **Check model embedding dimensions:**
   ```python
   EMBEDDING_DIMENSIONS = {
       "databricks-bge-large-en": 1024,
       "databricks-bge-small-en": 384,
       "sentence-transformers/all-MiniLM-L6-v2": 384
   }
   
   model_name = "databricks-bge-large-en"
   expected_dim = EMBEDDING_DIMENSIONS.get(model_name)
   print(f"Expected dimension for {model_name}: {expected_dim}")
   ```

2. **Verify index schema:**
   ```python
   index_info = vs_client.get_index("my_index")
   print(f"Index embedding dimension: {index_info.index_spec.embedding_dimension}")
   ```

3. **Recreate index with correct dimensions:**
   ```python
   # Delete old index
   vs_client.delete_index("my_index")
   
   # Create new index with correct dimensions
   vs_client.create_vector_index(
       endpoint_name="my_endpoint",
       index_name="my_index",
       embedding_dimension=1024,  # Correct dimension
       # ... other parameters
   )
   ```

## Model and Inference Problems

### Issue: Model inference failures
**Symptoms:**
```
HTTPError: 500 Internal Server Error
Model inference failed
TimeoutError: Model request timed out
```

**Solutions:**
1. **Check model availability:**
   ```python
   from databricks.sdk import WorkspaceClient
   
   w = WorkspaceClient()
   serving_endpoints = w.serving_endpoints.list()
   
   for endpoint in serving_endpoints:
       print(f"Model: {endpoint.name}, State: {endpoint.state}")
   ```

2. **Test model with simple request:**
   ```python
   def test_model_inference(model_name):
       try:
           response = chat_client.complete(
               messages=[{"role": "user", "content": "Hello"}],
               model=model_name,
               max_tokens=10
           )
           return True
       except Exception as e:
           print(f"Model test failed: {e}")
           return False
   ```

3. **Implement retry logic:**
   ```python
   import time
   from functools import wraps
   
   def retry_on_failure(max_retries=3, delay=1):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if attempt == max_retries - 1:
                           raise e
                       time.sleep(delay * (2 ** attempt))  # Exponential backoff
               return wrapper
           return decorator
   
   @retry_on_failure(max_retries=3)
   def robust_model_call(prompt):
       return chat_client.complete(messages=[{"role": "user", "content": prompt}])
   ```

### Issue: Slow model responses
**Symptoms:**
- Long response times (>30 seconds)
- Intermittent timeouts
- High latency

**Solutions:**
1. **Optimize prompt length:**
   ```python
   def trim_context(context, max_tokens=2000):
       """Trim context to fit within token limits"""
       words = context.split()
       if len(words) > max_tokens:
           return " ".join(words[:max_tokens]) + "..."
       return context
   ```

2. **Use streaming responses:**
   ```python
   def stream_response(prompt):
       for chunk in chat_client.stream(
           messages=[{"role": "user", "content": prompt}]
       ):
           yield chunk.choices[0].delta.content
   ```

3. **Implement caching:**
   ```python
   import hashlib
   import json
   
   response_cache = {}
   
   def cached_model_call(prompt, model_params):
       # Create cache key
       cache_key = hashlib.md5(
           (prompt + json.dumps(model_params, sort_keys=True)).encode()
       ).hexdigest()
       
       if cache_key in response_cache:
           return response_cache[cache_key]
       
       response = chat_client.complete(prompt, **model_params)
       response_cache[cache_key] = response
       return response
   ```

## Performance Issues

### Issue: Slow similarity search
**Symptoms:**
- Search queries take >10 seconds
- High memory usage during search
- Timeout errors

**Solutions:**
1. **Optimize query parameters:**
   ```python
   # Reduce number of results
   results = similarity_search(
       query="your question",
       top_k=3,  # Reduced from 10
       include_metadata=False  # Skip metadata if not needed
   )
   ```

2. **Check index size and status:**
   ```python
   index_info = vs_client.get_index("my_index")
   print(f"Index size: {index_info.index_spec.num_vectors}")
   print(f"Index status: {index_info.status}")
   ```

3. **Monitor query performance:**
   ```python
   import time
   
   def timed_similarity_search(query, **kwargs):
       start_time = time.time()
       results = similarity_search(query, **kwargs)
       duration = time.time() - start_time
       print(f"Search completed in {duration:.2f}s")
       return results
   ```

### Issue: Memory issues during document processing
**Symptoms:**
```
MemoryError: Unable to allocate array
Out of memory errors
Process killed due to memory limits
```

**Solutions:**
1. **Process documents in batches:**
   ```python
   def process_large_pdf_in_batches(file_path, batch_size=100):
       documents = load_and_split_documents(file_path)
       
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i + batch_size]
           
           # Process batch
           embeddings = generate_embeddings([doc.page_content for doc in batch])
           
           # Index batch
           index_documents(batch, embeddings)
           
           # Clear memory
           del embeddings
           import gc
           gc.collect()
   ```

2. **Monitor memory usage:**
   ```python
   import psutil
   import os
   
   def log_memory_usage(stage):
       process = psutil.Process(os.getpid())
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"{stage}: Memory usage: {memory_mb:.1f} MB")
   
   # Usage
   log_memory_usage("Before document loading")
   documents = load_and_split_documents("large_file.pdf")
   log_memory_usage("After document loading")
   ```

3. **Use streaming for large files:**
   ```python
   def stream_process_pdf(file_path, chunk_size=1000):
       """Stream process large PDFs without loading everything into memory"""
       with open(file_path, 'rb') as file:
           # Process file in chunks
           while True:
               chunk = file.read(chunk_size)
               if not chunk:
                   break
               yield process_chunk(chunk)
   ```

## Data Processing Problems

### Issue: PDF extraction errors
**Symptoms:**
```
PDFParserError: Unable to parse PDF
UnicodeDecodeError: Invalid character encoding
Empty text extraction from PDF
```

**Solutions:**
1. **Check PDF format:**
   ```python
   import PyPDF2
   
   def validate_pdf(file_path):
       try:
           with open(file_path, 'rb') as file:
               reader = PyPDF2.PdfReader(file)
               print(f"PDF pages: {len(reader.pages)}")
               print(f"PDF encrypted: {reader.is_encrypted}")
               
               # Test text extraction from first page
               if reader.pages:
                   first_page_text = reader.pages[0].extract_text()
                   print(f"First page text length: {len(first_page_text)}")
               
               return True
       except Exception as e:
           print(f"PDF validation failed: {e}")
           return False
   ```

2. **Try alternative PDF libraries:**
   ```python
   # Method 1: PyPDF2
   def extract_with_pypdf2(file_path):
       import PyPDF2
       with open(file_path, 'rb') as file:
           reader = PyPDF2.PdfReader(file)
           text = ""
           for page in reader.pages:
               text += page.extract_text()
       return text
   
   # Method 2: pdfplumber
   def extract_with_pdfplumber(file_path):
       import pdfplumber
       text = ""
       with pdfplumber.open(file_path) as pdf:
           for page in pdf.pages:
               text += page.extract_text() or ""
       return text
   
   # Method 3: pymupdf
   def extract_with_pymupdf(file_path):
       import fitz  # PyMuPDF
       doc = fitz.open(file_path)
       text = ""
       for page_num in range(doc.page_count):
           page = doc[page_num]
           text += page.get_text()
       return text
   ```

3. **Handle OCR for scanned PDFs:**
   ```python
   def extract_with_ocr(file_path):
       """Extract text from scanned PDFs using OCR"""
       import pytesseract
       from pdf2image import convert_from_path
       
       pages = convert_from_path(file_path)
       text = ""
       
       for page in pages:
           text += pytesseract.image_to_string(page)
       
       return text
   ```

### Issue: Text chunking problems
**Symptoms:**
- Chunks are too small or too large
- Important context is split across chunks
- Poor retrieval quality

**Solutions:**
1. **Analyze chunk distribution:**
   ```python
   def analyze_chunks(documents):
       chunk_lengths = [len(doc.page_content) for doc in documents]
       
       print(f"Total chunks: {len(chunk_lengths)}")
       print(f"Average length: {sum(chunk_lengths) / len(chunk_lengths):.1f}")
       print(f"Min length: {min(chunk_lengths)}")
       print(f"Max length: {max(chunk_lengths)}")
       
       # Plot distribution
       import matplotlib.pyplot as plt
       plt.hist(chunk_lengths, bins=20)
       plt.xlabel("Chunk Length")
       plt.ylabel("Frequency")
       plt.title("Chunk Length Distribution")
       plt.show()
   ```

2. **Custom chunking strategy:**
   ```python
   def smart_chunk_by_sections(text, max_chunk_size=1000):
       """Chunk text by logical sections"""
       import re
       
       # Split by headers (assuming markdown-style headers)
       sections = re.split(r'\n#{1,3}\s', text)
       
       chunks = []
       current_chunk = ""
       
       for section in sections:
           if len(current_chunk) + len(section) <= max_chunk_size:
               current_chunk += section
           else:
               if current_chunk:
                   chunks.append(current_chunk.strip())
               current_chunk = section
       
       if current_chunk:
           chunks.append(current_chunk.strip())
       
       return chunks
   ```

## Environment and Setup Issues

### Issue: Package version conflicts
**Symptoms:**
```
ImportError: cannot import name 'X' from 'Y'
AttributeError: module 'X' has no attribute 'Y'
Version conflicts between packages
```

**Solutions:**
1. **Check package versions:**
   ```python
   def check_package_versions():
       import pkg_resources
       
       required_packages = [
           'databricks-sdk',
           'langchain',
           'pypdf2',
           'numpy',
           'pandas'
       ]
       
       for package in required_packages:
           try:
               version = pkg_resources.get_distribution(package).version
               print(f"{package}: {version}")
           except pkg_resources.DistributionNotFound:
               print(f"{package}: NOT INSTALLED")
   ```

2. **Create clean environment:**
   ```bash
   # Create new virtual environment
   python -m venv fresh_rag_env
   source fresh_rag_env/bin/activate
   
   # Install packages from requirements.txt
   pip install -r requirements.txt
   ```

3. **Pin package versions in requirements.txt:**
   ```
   databricks-sdk==0.20.0
   langchain==0.1.0
   pypdf2==3.0.1
   numpy==1.24.3
   pandas==2.0.3
   ```

## Debugging Tools

### Debug Script Template
```python
#!/usr/bin/env python3
"""
RAG Application Debug Script
Run this to diagnose common issues
"""

import os
import sys
import traceback
from datetime import datetime

def debug_environment():
    """Debug environment setup"""
    print("=== Environment Debug ===")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    required_vars = ['DATABRICKS_HOST', 'DATABRICKS_TOKEN']
    for var in required_vars:
        value = os.environ.get(var)
        print(f"{var}: {'SET' if value else 'NOT SET'}")

def debug_packages():
    """Debug package installations"""
    print("\n=== Package Debug ===")
    try:
        import databricks
        print("✅ databricks package imported")
    except ImportError as e:
        print(f"❌ databricks import failed: {e}")
    
    try:
        from langchain.document_loaders import PyPDFLoader
        print("✅ langchain package imported")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")

def debug_connectivity():
    """Debug Databricks connectivity"""
    print("\n=== Connectivity Debug ===")
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        clusters = list(w.clusters.list())
        print(f"✅ Connected to Databricks, found {len(clusters)} clusters")
    except Exception as e:
        print(f"❌ Databricks connection failed: {e}")

def debug_vector_search():
    """Debug vector search setup"""
    print("\n=== Vector Search Debug ===")
    try:
        from databricks.vector_search.client import VectorSearchClient
        vs_client = VectorSearchClient()
        endpoints = vs_client.list_endpoints()
        print(f"✅ Vector search connected, found {len(endpoints)} endpoints")
        
        for endpoint in endpoints:
            print(f"  - {endpoint.name}: {endpoint.state}")
            
    except Exception as e:
        print(f"❌ Vector search failed: {e}")

def main():
    """Run all debug checks"""
    print(f"RAG Application Debug Report - {datetime.now()}")
    print("=" * 50)
    
    try:
        debug_environment()
        debug_packages()
        debug_connectivity()
        debug_vector_search()
        
        print("\n✅ Debug completed successfully")
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### Performance Monitoring
```python
import time
import functools
from typing import Any, Callable

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise e
        finally:
            end_time = time.time()
            end_memory = get_memory_usage()
            
            # Log performance metrics
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            print(f"Function: {func.__name__}")
            print(f"Duration: {duration:.3f}s")
            print(f"Memory change: {memory_delta:.1f}MB")
            print(f"Success: {success}")
            print("-" * 40)
        
        return result
    return wrapper

def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Usage
@monitor_performance
def slow_function():
    time.sleep(1)
    return "done"
```

This troubleshooting guide covers the most common issues you'll encounter when working with the RAG application. For additional help, refer to the [Setup Guide](setup-guide.md) and [API Reference](api-reference.md).
