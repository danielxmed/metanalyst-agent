"""
Processor tools for the Metanalyst Agent system.

This module provides tools for the processor agent that combines extraction and vectorization
into a single efficient process, handling URL processing from raw URLs to vector store.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIClient
import faiss
import numpy as np

from ..utils.config import get_config


def get_processor_client():
    """Get OpenAI client for processor operations."""
    config = get_config()
    api_key = config.api.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key is required for processor operations")
    # Use type ignore for the OpenAI client
    return OpenAIClient(api_key=api_key)  # type: ignore


def get_fast_llm():
    """Get fast LLM for content processing."""
    config = get_config()
    api_key = config.api.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key is required for fast LLM operations")
    return ChatOpenAI(
        model=config.llm.fast_processing_model,
        api_key=api_key,  # type: ignore
        temperature=0.1
    )


def extract_urls_content(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from URLs using Firecrawl API with Tavily fallback."""
    extracted_contents = []
    
    # Get Firecrawl API key from environment
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        print("⚠️ FIRECRAWL_API_KEY not found in environment variables")
        print("📥 Fallback: Using Tavily for content extraction...")
        return extract_urls_content_with_tavily(urls)
    
    print("📥 Using Firecrawl API for content extraction...")
    
    # Try to use MCP Firecrawl integration first
    try:
        print("🔥 Attempting to use MCP Firecrawl integration...")
        # For now, let's try the direct API approach
        extracted_contents = extract_urls_content_with_direct_firecrawl(urls, firecrawl_api_key)
        
        if extracted_contents:
            print(f"✅ Successfully extracted {len(extracted_contents)} papers using Firecrawl")
            return extracted_contents
        else:
            print("⚠️ Firecrawl returned no content, falling back to Tavily...")
            return extract_urls_content_with_tavily(urls)
            
    except Exception as e:
        print(f"⚠️ Error with Firecrawl: {str(e)}")
        print("📥 Fallback: Using Tavily for content extraction...")
        return extract_urls_content_with_tavily(urls)


def extract_urls_content_with_direct_firecrawl(urls: List[str], api_key: str) -> List[Dict[str, Any]]:
    """Direct Firecrawl API extraction."""
    extracted_contents = []
    
    # Process URLs individually for better reliability
    for url in urls:
        try:
            import requests
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "includeTags": ["article", "main", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "td", "th", "tr"],
                "excludeTags": ["nav", "header", "footer", "aside", "script", "style", "iframe", "noscript"]
            }
            
            response = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success") and result.get("data", {}).get("markdown"):
                    extracted_contents.append({
                        "url": url,
                        "content": result["data"]["markdown"],
                        "title": result.get("data", {}).get("metadata", {}).get("title", ""),
                        "extraction_timestamp": datetime.now().isoformat()
                    })
                    print(f"✅ Successfully extracted: {url}")
                else:
                    print(f"⚠️ No content from Firecrawl for: {url}")
            else:
                print(f"⚠️ Firecrawl API error for {url}: {response.status_code}")
                        
        except Exception as e:
            print(f"⚠️ Error extracting {url} with Firecrawl: {str(e)}")
            continue
    
    return extracted_contents


def extract_urls_content_with_tavily(urls: List[str]) -> List[Dict[str, Any]]:
    """Fallback function to extract content from URLs using Tavily API."""
    extracted_contents = []
    
    # Import tavily_tools here to avoid circular imports
    from ..tools.tavily_tools import extract_paper_content
    
    # Process URLs in batches to avoid overwhelming the API
    batch_size = 5
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        
        try:
            # Use tavily extract_paper_content tool - pass as dict with proper params
            extraction_result_str = extract_paper_content.invoke({
                "urls": batch,
                "extract_depth": "advanced",
                "format_type": "markdown"
            })
            extraction_result = json.loads(extraction_result_str)
            
            if extraction_result.get("success"):
                for paper in extraction_result.get("extracted_papers", []):
                    if paper.get("raw_content"):
                        extracted_contents.append({
                            "url": paper["url"],
                            "content": paper["raw_content"],
                            "title": paper.get("title", ""),
                            "extraction_timestamp": datetime.now().isoformat()
                        })
                        
        except Exception as e:
            print(f"⚠️ Error extracting batch {i//batch_size + 1} with Tavily: {str(e)}")
            continue
    
    return extracted_contents


def process_content_to_json(extracted_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process extracted content to structured JSON using GPT-4.1-mini."""
    processed_papers = []
    fast_llm = get_fast_llm()
    
    for content_data in extracted_contents:
        try:
            # Create prompt for GPT-4.1-mini
            prompt = f"""
            Analise o seguinte conteúdo de uma publicação científica e extraia as informações em formato JSON estruturado.

            IMPORTANTE: Se o conteúdo for principalmente HTML de navegação, menus, headers ou elementos não-científicos, 
            extraia apenas o que for relevante para pesquisa médica. Se não houver conteúdo científico suficiente, 
            indique isso claramente.

            Conteúdo:
            {content_data['content'][:8000]}

            Retorne APENAS um JSON válido com a seguinte estrutura:
            {{
                "reference": "referência completa em formato Vancouver (autor, título, revista, ano, etc) - se possível extrair do conteúdo",
                "content": "Resumo do conteúdo científico encontrado: metodologia, resultados, estatísticas (n, RR, OR, IC, p-valor), conclusões. Se não houver conteúdo científico, indique 'Conteúdo científico não disponível - apenas elementos de navegação/HTML'",
                "url": "{content_data['url']}",
                "content_quality": "alta|média|baixa - baseado na quantidade de informação científica extraída"
            }}

            CRITÉRIOS DE QUALIDADE:
            - Alta: Abstract, metodologia, resultados numéricos, estatísticas
            - Média: Título, algum conteúdo científico parcial
            - Baixa: Apenas elementos de navegação/HTML, sem conteúdo científico

            Seja preciso e objetivo. Extraia apenas informações factuais e quantitativas.
            """
            
            # Use GPT-4.1-mini for fast processing
            response = fast_llm.invoke(prompt)
            
            # Try to parse JSON response
            try:
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # Ensure response_content is a string
                if isinstance(response_content, list):
                    # Join text parts if content is a list
                    response_content = "".join([
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in response_content
                    ])
                elif not isinstance(response_content, str):
                    response_content = str(response_content)
                
                # Try to extract JSON from the response, even if there's extra text
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    processed_data = json.loads(json_str)
                else:
                    # If no JSON found, try parsing the entire content
                    processed_data = json.loads(response_content)
                
                processed_data["processing_timestamp"] = datetime.now().isoformat()
                processed_papers.append(processed_data)
                
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse JSON for {content_data['url']}")
                # Create fallback structure with better content analysis
                title = content_data.get('title', 'Unknown')
                content_sample = content_data['content'][:2000]
                
                # Simple heuristic to detect if content might be scientific
                scientific_keywords = ['abstract', 'methods', 'results', 'conclusion', 'patients', 'study', 'trial', 'analysis', 'statistical', 'p-value', 'confidence interval']
                content_lower = content_sample.lower()
                keyword_count = sum(1 for keyword in scientific_keywords if keyword in content_lower)
                
                if keyword_count >= 2:
                    quality = "média"
                    content_note = f"Conteúdo parcialmente científico detectado - {keyword_count} palavras-chave encontradas"
                else:
                    quality = "baixa"
                    content_note = "Conteúdo científico não disponível - principalmente elementos de navegação/HTML"
                
                processed_papers.append({
                    "reference": f"Artigo não processado - {title}",
                    "content": content_note,
                    "url": content_data['url'],
                    "content_quality": quality,
                    "processing_timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            print(f"⚠️ Error processing content for {content_data['url']}: {str(e)}")
            continue
    
    return processed_papers


def save_json_files(processed_papers: List[Dict[str, Any]], url_json_dir: Path) -> None:
    """Save processed papers as JSON files."""
    for i, paper in enumerate(processed_papers):
        try:
            file_path = url_json_dir / f"paper_{i:04d}_{uuid4().hex[:8]}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(paper, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving JSON file for paper {i}: {str(e)}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into smaller pieces with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Avoid infinite loops
        if start >= len(text):
            break
    
    return chunks


def chunk_json_files(url_json_dir: Path, chunks_dir: Path) -> List[Dict[str, Any]]:
    """Chunk all JSON files into smaller pieces."""
    chunks = []
    config = get_config()
    
    for json_file in url_json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Chunk the content
            content = paper_data.get("content", "")
            paper_chunks = chunk_text(
                content, 
                config.vector.chunk_size, 
                config.vector.chunk_overlap
            )
            
            # Create chunk objects
            for i, chunk_content in enumerate(paper_chunks):
                chunk = {
                    "id": str(uuid4()),
                    "reference": paper_data.get("reference", ""),
                    "content": chunk_content,
                    "url": paper_data.get("url", ""),
                    "chunk_index": i,
                    "total_chunks": len(paper_chunks),
                    "source_file": json_file.name
                }
                chunks.append(chunk)
                
                # Save chunk to file
                chunk_file = chunks_dir / f"chunk_{chunk['id']}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"⚠️ Error chunking {json_file}: {str(e)}")
            continue
    
    return chunks


def vectorize_and_store_chunks(chunks: List[Dict[str, Any]], vector_store_dir: Path) -> bool:
    """Generate embeddings and store in vector store."""
    try:
        if not chunks:
            return False
        
        config = get_config()
        openai_client = get_processor_client()
        
        # Generate embeddings for all chunks
        embeddings = []
        chunk_metadata = []
        
        print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        
        # Process chunks in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk['content'] for chunk in batch]
            
            # Generate embeddings
            response = openai_client.embeddings.create(
                model=config.vector.embedding_model,
                input=batch_texts
            )
            
            # Extract embeddings
            for j, embedding_data in enumerate(response.data):
                embeddings.append(embedding_data.embedding)
                chunk_metadata.append(batch[j])
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / norms
        
        # Add embeddings to index
        index.add(embeddings_array)
        
        # Save index
        index_path = vector_store_dir / "faiss_index.index"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = vector_store_dir / "chunk_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Vector store created with {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return False


def cleanup_directory(directory: Path, description: str) -> None:
    """Clean up temporary files in a directory."""
    try:
        for file in directory.glob("*.json"):
            file.unlink()
        print(f"🧹 {description} cleaned up")
    except Exception as e:
        print(f"⚠️ Error cleaning up {description.lower()}: {str(e)}")


def deduplicate_urls(urls: List[str]) -> List[str]:
    """
    Deduplicate URLs while preserving order.
    
    Args:
        urls: List of URLs that may contain duplicates
        
    Returns:
        List of unique URLs in original order
    """
    seen = set()
    unique_urls = []
    
    for url in urls:
        # Normalize URL for comparison (remove trailing slashes, convert to lowercase)
        normalized_url = url.strip().lower().rstrip('/')
        
        if normalized_url not in seen:
            seen.add(normalized_url)
            unique_urls.append(url)  # Keep original URL format
    
    return unique_urls


@tool
def process_urls(url_list: List[str]) -> str:
    """
    Process URLs through the complete extraction and vectorization pipeline.
    
    This tool combines extraction and vectorization into a single efficient process:
    1. Deduplicates URLs to avoid processing duplicates
    2. Extracts content from URLs using Firecrawl API (focuses on main content)
    3. Processes markdown to structured JSON using GPT-4.1-mini
    4. Chunks content intelligently (1000 chars, 100 overlap)
    5. Generates embeddings with text-embedding-3-small
    6. Stores in local vector store
    7. Manages temporary files and cleanup
    
    Args:
        url_list: List of URLs to process
        
    Returns:
        JSON string with processing results and state updates
    """
    try:
        # Step 0: Deduplicate URLs
        original_count = len(url_list)
        deduplicated_urls = deduplicate_urls(url_list)
        duplicates_removed = original_count - len(deduplicated_urls)
        
        if duplicates_removed > 0:
            print(f"🔄 Removed {duplicates_removed} duplicate URLs. Processing {len(deduplicated_urls)} unique URLs...")
        else:
            print(f"🔄 Processing {len(deduplicated_urls)} URLs...")
        
        # Setup directories
        data_dir = Path("data")
        url_json_dir = data_dir / "url_json"
        chunks_dir = data_dir / "chunks"
        vector_store_dir = data_dir / "vector_store"
        
        for directory in [url_json_dir, chunks_dir, vector_store_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract content from URLs using Firecrawl
        print("📥 Step 1: Extracting content from URLs using Firecrawl...")
        extracted_contents = extract_urls_content(deduplicated_urls)
        
        if not extracted_contents:
            result = {
                "success": False,
                "error": "No content could be extracted from URLs",
                "url_processed": [],
                "url_not_processed": deduplicated_urls,
                "duplicates_removed": duplicates_removed,
                "original_url_count": original_count
            }
            return json.dumps(result, ensure_ascii=False)
        
        # Step 2: Process markdown to structured JSON
        print("🧠 Step 2: Processing content with GPT-4.1-mini...")
        processed_papers = process_content_to_json(extracted_contents)
        
        # Report content quality
        if processed_papers:
            quality_counts = {"alta": 0, "média": 0, "baixa": 0}
            for paper in processed_papers:
                quality = paper.get("content_quality", "baixa")
                if quality in quality_counts:
                    quality_counts[quality] += 1
            
            print(f"📊 Qualidade do conteúdo: {quality_counts['alta']} alta, {quality_counts['média']} média, {quality_counts['baixa']} baixa")
            
            if quality_counts["alta"] == 0 and quality_counts["média"] == 0:
                print("⚠️ ATENÇÃO: Nenhum conteúdo científico de qualidade foi extraído!")
        else:
            print("⚠️ Nenhum paper foi processado com sucesso")
        
        # Step 3: Save JSONs temporarily
        print("💾 Step 3: Saving structured JSONs...")
        save_json_files(processed_papers, url_json_dir)
        
        # Step 4: Chunk all JSON files
        print("🔪 Step 4: Chunking content...")
        chunks = chunk_json_files(url_json_dir, chunks_dir)
        
        # Step 5: Clean up JSON files
        print("🧹 Step 5: Cleaning up temporary JSON files...")
        cleanup_directory(url_json_dir, "Temporary JSON files")
        
        # Step 6: Generate embeddings and store in vector store
        print("🧠 Step 6: Generating embeddings and storing in vector store...")
        vector_store_ready = vectorize_and_store_chunks(chunks, vector_store_dir)
        
        # Step 7: Clean up chunk files
        print("🧹 Step 7: Cleaning up temporary chunk files...")
        cleanup_directory(chunks_dir, "Temporary chunk files")
        
        # Prepare results
        processed_urls = [content['url'] for content in extracted_contents]
        failed_urls = [url for url in deduplicated_urls if url not in processed_urls]
        
        result = {
            "success": True,
            "message": f"Successfully processed {len(processed_urls)} URLs",
            "url_processed": processed_urls,
            "url_not_processed": failed_urls,
            "vector_store_ready": vector_store_ready,
            "vector_store_path": str(vector_store_dir),
            "chunks_created": len(chunks),
            "processed_papers": len(processed_papers),
            "processing_timestamp": datetime.now().isoformat(),
            "duplicates_removed": duplicates_removed,
            "original_url_count": original_count
        }
        
        print(f"✅ Processing complete! {len(processed_urls)} URLs processed successfully.")
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Error processing URLs: {str(e)}")
        error_result = {
            "success": False,
            "error": str(e),
            "url_processed": [],
            "url_not_processed": deduplicated_urls if 'deduplicated_urls' in locals() else url_list,
            "duplicates_removed": duplicates_removed if 'duplicates_removed' in locals() else 0,
            "original_url_count": original_count if 'original_count' in locals() else len(url_list)
        }
        return json.dumps(error_result, ensure_ascii=False)


# List of processor tools
PROCESSOR_TOOLS = [
    process_urls
]
