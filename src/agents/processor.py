"""
Processor Agent for the Metanalyst Agent system.

This agent combines the functionality of extraction and vectorization into a single
efficient process, handling URL processing from raw URLs to vector store.

The processor agent:
1. Extracts content from URLs using tavily_extract
2. Processes markdown to structured JSON using GPT-4.1-nano
3. Chunks content intelligently
4. Generates embeddings and stores in vector store
5. Manages temporary files and cleanup
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.tools import tool
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI as OpenAIClient
from pydantic import SecretStr
import faiss
import numpy as np

from ..models.state import MetanalysisState
from ..models.schemas import ExtractedPaper, create_paper_template
from ..utils.config import get_config
from ..tools.tavily_tools import extract_paper_content


class ProcessorAgent:
    """
    Processor Agent that combines extraction and vectorization.
    
    This agent handles the complete pipeline from URLs to vector store,
    including content extraction, processing, chunking, and embedding.
    """
    
    def __init__(self):
        """Initialize the processor agent."""
        self.config = get_config()
        self.openai_client = OpenAIClient(api_key=str(self.config.api.openai_api_key or ""))
        api_key = self.config.api.openai_api_key or ""
        self.fast_llm = OpenAI(
            model=self.config.llm.fast_processing_model,
            api_key=SecretStr(str(api_key)) if api_key else None,
            temperature=0.1
        )
        
        # Ensure directories exist
        self.data_dir = Path("data")
        self.url_json_dir = self.data_dir / "url_json"
        self.chunks_dir = self.data_dir / "chunks"
        self.vector_store_dir = self.data_dir / "vector_store"
        
        for directory in [self.url_json_dir, self.chunks_dir, self.vector_store_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process a list of URLs through the complete pipeline.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dict containing processing results and updated state information
        """
        # Step 0: Deduplicate URLs
        original_count = len(urls)
        deduplicated_urls = self._deduplicate_urls(urls)
        duplicates_removed = original_count - len(deduplicated_urls)
        
        if duplicates_removed > 0:
            print(f"🔄 Removed {duplicates_removed} duplicate URLs. Processing {len(deduplicated_urls)} unique URLs...")
        else:
            print(f"🔄 Processing {len(deduplicated_urls)} URLs...")
        
        try:
            # Step 1: Extract content from URLs
            print("📥 Step 1: Extracting content from URLs...")
            extracted_contents = self._extract_urls_content(deduplicated_urls)
            
            if not extracted_contents:
                return {
                    "success": False,
                    "error": "No content could be extracted from URLs",
                    "url_processed": [],
                    "url_not_processed": deduplicated_urls,
                    "duplicates_removed": duplicates_removed,
                    "original_url_count": original_count
                }
            
            # Step 2: Process markdown to structured JSON
            print("🧠 Step 2: Processing content with GPT-4.1-nano...")
            processed_papers = self._process_content_to_json(extracted_contents)
            
            # Step 3: Save JSONs temporarily
            print("💾 Step 3: Saving structured JSONs...")
            self._save_json_files(processed_papers)
            
            # Step 4: Chunk all JSON files
            print("🔪 Step 4: Chunking content...")
            chunks = self._chunk_json_files()
            
            # Step 5: Clean up JSON files
            print("🧹 Step 5: Cleaning up temporary JSON files...")
            self._cleanup_json_files()
            
            # Step 6: Generate embeddings and store in vector store
            print("🧠 Step 6: Generating embeddings and storing in vector store...")
            vector_store_ready = self._vectorize_and_store_chunks(chunks)
            
            # Step 7: Clean up chunk files
            print("🧹 Step 7: Cleaning up temporary chunk files...")
            self._cleanup_chunk_files()
            
            # Prepare results
            processed_urls = [content['url'] for content in extracted_contents]
            failed_urls = [url for url in deduplicated_urls if url not in processed_urls]
            
            result = {
                "success": True,
                "message": f"Successfully processed {len(processed_urls)} URLs",
                "url_processed": processed_urls,
                "url_not_processed": failed_urls,
                "vector_store_ready": vector_store_ready,
                "vector_store_path": str(self.vector_store_dir),
                "chunks_created": len(chunks),
                "processed_papers": len(processed_papers),
                "processing_timestamp": datetime.now().isoformat(),
                "duplicates_removed": duplicates_removed,
                "original_url_count": original_count
            }
            
            print(f"✅ Processing complete! {len(processed_urls)} URLs processed successfully.")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing URLs: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url_processed": [],
                "url_not_processed": deduplicated_urls if 'deduplicated_urls' in locals() else urls,
                "duplicates_removed": duplicates_removed if 'duplicates_removed' in locals() else 0,
                "original_url_count": original_count if 'original_count' in locals() else len(urls)
            }
    
    def _extract_urls_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract content from URLs using tavily_extract."""
        extracted_contents = []
        
        # Process URLs in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            
            try:
                # Use tavily_extract to get content
                extraction_result_str = extract_paper_content.invoke({"urls": batch})
                extraction_result = json.loads(extraction_result_str)
                
                if extraction_result.get("success"):
                    for url_result in extraction_result.get("extracted_papers", []):
                        if url_result.get("raw_content"):
                            extracted_contents.append({
                                "url": url_result["url"],
                                "content": url_result["raw_content"],
                                "title": url_result.get("title", ""),
                                "extraction_timestamp": datetime.now().isoformat()
                            })
                            
            except Exception as e:
                print(f"⚠️ Error extracting batch {i//batch_size + 1}: {str(e)}")
                continue
        
        return extracted_contents
    
    def _process_content_to_json(self, extracted_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process extracted content to structured JSON using GPT-4.1-nano."""
        processed_papers = []
        
        for content_data in extracted_contents:
            try:
                # Create prompt for GPT-4.1-nano
                prompt = f"""
                Analise o seguinte conteúdo de uma publicação científica em markdown e extraia as informações em formato JSON estruturado.

                Conteúdo:
                {content_data['content'][:8000]}

                IMPORTANTE: Retorne APENAS um JSON válido, sem explicações ou texto adicional.

                Estrutura esperada:
                {{
                    "reference": "referência completa em formato Vancouver da publicação",
                    "content": "Relatório dos pontos mais importantes da publicação, principalmente dados objetivos como tamanho de amostra, dados estatísticos, resultados, RR, OR, etc",
                    "url": "{content_data['url']}"
                }}

                Seja preciso e objetivo. Extraia apenas informações factuais e quantitativas.
                """
                
                # Use GPT-4.1-nano for fast processing
                response = self.fast_llm.invoke(prompt)
                
                # Clean response to extract only JSON
                response_clean = response.strip()
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:]
                if response_clean.endswith('```'):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()
                
                # Try to parse JSON response
                try:
                    processed_data = json.loads(response_clean)
                    processed_data["processing_timestamp"] = datetime.now().isoformat()
                    processed_papers.append(processed_data)
                    
                except json.JSONDecodeError:
                    print(f"⚠️ Could not parse JSON for {content_data['url']}")
                    # Create fallback structure
                    processed_papers.append({
                        "reference": f"Reference for {content_data.get('title', 'Unknown')}",
                        "content": content_data['content'][:2000],  # Truncate if needed
                        "url": content_data['url'],
                        "processing_timestamp": datetime.now().isoformat()
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing content for {content_data['url']}: {str(e)}")
                continue
        
        return processed_papers
    
    def _save_json_files(self, processed_papers: List[Dict[str, Any]]) -> None:
        """Save processed papers as JSON files."""
        for i, paper in enumerate(processed_papers):
            try:
                file_path = self.url_json_dir / f"paper_{i:04d}_{uuid4().hex[:8]}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(paper, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"⚠️ Error saving JSON file for paper {i}: {str(e)}")
    
    def _chunk_json_files(self) -> List[Dict[str, Any]]:
        """Chunk all JSON files into smaller pieces."""
        chunks = []
        
        for json_file in self.url_json_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Chunk the content
                content = paper_data.get("content", "")
                paper_chunks = self._chunk_text(
                    content, 
                    self.config.vector.chunk_size, 
                    self.config.vector.chunk_overlap
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
                    chunk_file = self.chunks_dir / f"chunk_{chunk['id']}.json"
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"⚠️ Error chunking {json_file}: {str(e)}")
                continue
        
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
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
    
    def _vectorize_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Generate embeddings and store in vector store."""
        try:
            if not chunks:
                return False
            
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
                response = self.openai_client.embeddings.create(
                    model=self.config.vector.embedding_model,
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
            embeddings_array = np.ascontiguousarray(embeddings_array, dtype=np.float32)
            # Manual L2 normalization for compatibility
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            # Add embeddings to index
            if embeddings_array.size > 0 and embeddings_array.shape[0] > 0:
                index.add(embeddings_array)
            
            # Save index
            try:
                index_path = self.vector_store_dir / "faiss_index.index"
                faiss.write_index(index, str(index_path))
                
                # Save metadata
                metadata_path = self.vector_store_dir / "chunk_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
                
                print(f"✅ Vector store created with {len(embeddings)} embeddings")
                
                return True
            except Exception as faiss_error:
                print(f"❌ Error saving FAISS index: {str(faiss_error)}")
                return False
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            return False
    
    def _cleanup_json_files(self) -> None:
        """Clean up temporary JSON files."""
        try:
            for json_file in self.url_json_dir.glob("*.json"):
                json_file.unlink()
            print("🧹 Temporary JSON files cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up JSON files: {str(e)}")
    
    def _cleanup_chunk_files(self) -> None:
        """Clean up temporary chunk files."""
        try:
            for chunk_file in self.chunks_dir.glob("*.json"):
                chunk_file.unlink()
            print("🧹 Temporary chunk files cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up chunk files: {str(e)}")
    
    def _deduplicate_urls(self, urls: List[str]) -> List[str]:
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


# Create processor agent tool
@tool
def process_urls(url_list: List[str]) -> Dict[str, Any]:
    """
    Process URLs through the complete extraction and vectorization pipeline.
    
    This tool combines extraction and vectorization into a single efficient process:
    1. Extracts content from URLs using tavily_extract
    2. Processes markdown to structured JSON using GPT-4.1-nano
    3. Chunks content intelligently (1000 chars, 100 overlap)
    4. Generates embeddings with text-embedding-3-small
    5. Stores in local vector store
    6. Manages temporary files and cleanup
    
    Args:
        url_list: List of URLs to process
        
    Returns:
        Dict containing processing results and state updates
    """
    processor = ProcessorAgent()
    return processor.process_urls(url_list)


# For backwards compatibility - create the processor agent instance
processor_agent = ProcessorAgent()
