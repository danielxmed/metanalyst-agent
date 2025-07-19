"""
Processor Tools for Metanalyst-Agent

Specialized tools for the Processor Agent to extract content from articles,
process statistical data, generate citations, and create vector embeddings.
Uses AI-first approach with LLMs for intelligent data extraction.
"""

import os
import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tavily import TavilyClient
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup


# Initialize clients lazily
def get_tavily_client():
    """Get or create Tavily client"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    return TavilyClient(api_key=api_key)

def get_llm():
    """Get or create OpenAI LLM client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=api_key,
        temperature=0.1
    )

def get_embeddings():
    """Get or create OpenAI embeddings client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY")
)


@tool
def extract_article_content(url: str) -> Dict[str, Any]:
    """
    Extract full content from article URL using Tavily Extract API
    
    Args:
        url: URL of the article to extract
        
    Returns:
        Dictionary with extracted content and metadata
    """
    
    try:
        # Use Tavily Extract for comprehensive content extraction
        extract_result = get_tavily_client().extract(url)
        
        # Clean and structure the extracted content
        content = {
            "url": url,
            "raw_content": extract_result.get("raw_content", ""),
            "title": extract_result.get("title", ""),
            "author": extract_result.get("author", ""),
            "published_date": extract_result.get("published_date", ""),
            "content": extract_result.get("content", ""),
            "extracted_at": datetime.now().isoformat(),
            "extraction_method": "tavily_extract",
            "success": True
        }
        
        # Additional metadata extraction if available
        if "metadata" in extract_result:
            content["metadata"] = extract_result["metadata"]
        
        return content
    
    except Exception as e:
        # Fallback to direct HTTP extraction
        try:
            fallback_content = _fallback_extract(url)
            fallback_content["extraction_error"] = str(e)
            fallback_content["extraction_method"] = "fallback_http"
            return fallback_content
        
        except Exception as fallback_error:
            return {
                "url": url,
                "success": False,
                "error": str(e),
                "fallback_error": str(fallback_error),
                "extracted_at": datetime.now().isoformat()
            }


def _fallback_extract(url: str) -> Dict[str, Any]:
    """Fallback extraction method using requests and BeautifulSoup"""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract title
    title = ""
    if soup.title:
        title = soup.title.string.strip()
    elif soup.find('h1'):
        title = soup.find('h1').get_text().strip()
    
    # Extract main content
    content = ""
    # Try common content selectors
    content_selectors = [
        'article', '.article-content', '.content', 
        '.abstract', '.article-text', 'main'
    ]
    
    for selector in content_selectors:
        content_elem = soup.select_one(selector)
        if content_elem:
            content = content_elem.get_text(separator='\n', strip=True)
            break
    
    if not content:
        # Fallback to body text
        content = soup.get_text(separator='\n', strip=True)
    
    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": str(soup),
        "success": True
    }


@tool
def extract_statistical_data(
    content: str, 
    pico: Dict[str, str],
    study_type: str = "unknown"
) -> Dict[str, Any]:
    """
    Extract statistical data relevant to PICO from article content using LLM
    
    Args:
        content: Full article content
        pico: PICO framework for context
        study_type: Type of study (RCT, systematic_review, etc.)
        
    Returns:
        Structured statistical data
    """
    
    prompt = f"""
    You are a medical statistician expert. Extract statistical data from this scientific article 
    that is relevant to the PICO framework provided.

    PICO CONTEXT:
    Population (P): {pico.get('P', 'Not specified')}
    Intervention (I): {pico.get('I', 'Not specified')}
    Comparison (C): {pico.get('C', 'Not specified')}
    Outcome (O): {pico.get('O', 'Not specified')}

    STUDY TYPE: {study_type}

    ARTICLE CONTENT:
    {content[:4000]}...

    Extract the following statistical information if available:
    1. Sample sizes (total, intervention group, control group)
    2. Effect sizes (mean differences, odds ratios, risk ratios, etc.)
    3. Confidence intervals
    4. P-values and significance levels
    5. Baseline characteristics
    6. Primary and secondary outcomes with statistics
    7. Study duration and follow-up
    8. Dropout rates and missing data

    Return your response in this exact JSON format:
    {{
        "sample_size": {{
            "total": number or null,
            "intervention": number or null,
            "control": number or null,
            "analyzed": number or null
        }},
        "primary_outcomes": [
            {{
                "outcome": "outcome name",
                "intervention_value": number or null,
                "control_value": number or null,
                "effect_size": number or null,
                "effect_measure": "OR/RR/MD/SMD/etc",
                "confidence_interval": [lower, upper] or null,
                "p_value": number or null,
                "significance": "significant/not_significant/unclear"
            }}
        ],
        "secondary_outcomes": [
            {{
                "outcome": "outcome name",
                "intervention_value": number or null,
                "control_value": number or null,
                "effect_size": number or null,
                "effect_measure": "OR/RR/MD/SMD/etc",
                "confidence_interval": [lower, upper] or null,
                "p_value": number or null
            }}
        ],
        "baseline_characteristics": {{
            "mean_age": number or null,
            "gender_distribution": {{
                "male_percent": number or null,
                "female_percent": number or null
            }},
            "comorbidities": ["condition1", "condition2"] or null
        }},
        "study_design": {{
            "type": "RCT/cohort/case-control/cross-sectional/etc",
            "blinding": "double/single/open/unclear",
            "randomization": "yes/no/unclear",
            "duration_weeks": number or null,
            "followup_weeks": number or null
        }},
        "quality_indicators": {{
            "dropout_rate": number or null,
            "missing_data_percent": number or null,
            "intention_to_treat": "yes/no/unclear",
            "allocation_concealment": "adequate/inadequate/unclear"
        }},
        "statistical_methods": ["method1", "method2"] or null,
        "confidence_level": 95,
        "data_extraction_confidence": 0.8
    }}

    If information is not available, use null. Be precise with numbers and conservative with confidence.
    """
    
    try:
        response = get_llm().invoke(prompt)
        
        # Parse JSON response
        statistical_data = json.loads(response.content)
        
        # Add metadata
        statistical_data.update({
            "extracted_at": datetime.now().isoformat(),
            "extraction_method": "llm_structured",
            "pico_context": pico,
            "content_length": len(content),
            "success": True
        })
        
        return statistical_data
    
    except json.JSONDecodeError as e:
        # Try to extract partial information
        return _fallback_statistical_extraction(content, pico, str(e))
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "extracted_at": datetime.now().isoformat(),
            "pico_context": pico
        }


def _fallback_statistical_extraction(content: str, pico: Dict[str, str], error: str) -> Dict[str, Any]:
    """Fallback statistical extraction with simpler approach"""
    
    # Basic pattern matching for common statistical terms
    import re
    
    # Look for sample sizes
    sample_patterns = [
        r'n\s*=\s*(\d+)',
        r'N\s*=\s*(\d+)',
        r'(\d+)\s*patients?',
        r'(\d+)\s*participants?',
        r'(\d+)\s*subjects?'
    ]
    
    sample_sizes = []
    for pattern in sample_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        sample_sizes.extend([int(m) for m in matches])
    
    # Look for p-values
    p_value_patterns = [
        r'p\s*[<>=]\s*([0-9.]+)',
        r'P\s*[<>=]\s*([0-9.]+)',
        r'p-value\s*[<>=]\s*([0-9.]+)'
    ]
    
    p_values = []
    for pattern in p_value_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        p_values.extend([float(m) for m in matches if float(m) <= 1.0])
    
    return {
        "sample_size": {
            "total": max(sample_sizes) if sample_sizes else None,
            "intervention": None,
            "control": None,
            "analyzed": None
        },
        "primary_outcomes": [{
            "outcome": pico.get('O', 'Primary outcome'),
            "p_value": min(p_values) if p_values else None,
            "significance": "significant" if p_values and min(p_values) < 0.05 else "unclear"
        }] if p_values else [],
        "secondary_outcomes": [],
        "baseline_characteristics": {},
        "study_design": {"type": "unclear"},
        "quality_indicators": {},
        "statistical_methods": [],
        "confidence_level": 95,
        "data_extraction_confidence": 0.3,
        "extraction_method": "fallback_regex",
        "extraction_error": error,
        "success": False,
        "extracted_at": datetime.now().isoformat()
    }


@tool
def generate_vancouver_citation(article_data: Dict[str, Any]) -> str:
    """
    Generate Vancouver style citation for article using LLM
    
    Args:
        article_data: Article metadata and content
        
    Returns:
        Vancouver formatted citation
    """
    
    prompt = f"""
    Generate a Vancouver style citation for this scientific article.

    ARTICLE DATA:
    Title: {article_data.get('title', 'Unknown title')}
    Authors: {article_data.get('author', 'Unknown authors')}
    URL: {article_data.get('url', '')}
    Published Date: {article_data.get('published_date', 'Unknown date')}
    Content: {article_data.get('content', '')[:500]}...

    Generate a properly formatted Vancouver citation following these rules:
    1. Authors (up to 6, then "et al.")
    2. Article title
    3. Journal name (abbreviated if possible)
    4. Publication year
    5. Volume and issue
    6. Page numbers
    7. DOI or URL if no DOI

    If information is missing, extract what you can from the content and URL.
    Return ONLY the citation, no other text.
    """
    
    try:
        response = get_llm().invoke(prompt)
        citation = response.content.strip()
        
        # Add citation number placeholder
        if not citation.startswith('['):
            citation = f"[CITATION_NUMBER] {citation}"
        
        return citation
    
    except Exception as e:
        # Fallback citation
        title = article_data.get('title', 'Unknown title')
        url = article_data.get('url', '')
        date = article_data.get('published_date', datetime.now().year)
        
        return f"[CITATION_NUMBER] Unknown authors. {title}. Available from: {url}. Accessed {date}."


@tool
def process_article_metadata(article_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and standardize article metadata using LLM
    
    Args:
        article_data: Raw article data
        
    Returns:
        Processed and standardized metadata
    """
    
    prompt = f"""
    Extract and standardize metadata from this scientific article data.

    RAW ARTICLE DATA:
    {json.dumps(article_data, indent=2)[:2000]}...

    Extract and standardize the following information:
    1. Article type (RCT, systematic review, meta-analysis, cohort, case-control, etc.)
    2. Journal name and impact factor (if identifiable)
    3. Publication year
    4. Study population details
    5. Intervention details
    6. Primary endpoints
    7. Study quality indicators
    8. Funding source
    9. Conflicts of interest

    Return JSON format:
    {{
        "article_type": "string",
        "journal": {{
            "name": "string",
            "impact_factor": number or null,
            "issn": "string or null"
        }},
        "publication": {{
            "year": number,
            "volume": "string or null",
            "issue": "string or null",
            "pages": "string or null",
            "doi": "string or null"
        }},
        "study_details": {{
            "population_size": number or null,
            "population_description": "string",
            "intervention_description": "string",
            "primary_endpoint": "string",
            "study_duration": "string or null"
        }},
        "quality_assessment": {{
            "jadad_score": number or null,
            "cochrane_risk_bias": "low/high/unclear",
            "funding_source": "string or null",
            "conflicts_declared": "yes/no/unclear"
        }},
        "processing_confidence": 0.8
    }}
    """
    
    try:
        response = get_llm().invoke(prompt)
        metadata = json.loads(response.content)
        
        # Add processing information
        metadata.update({
            "processed_at": datetime.now().isoformat(),
            "original_url": article_data.get("url", ""),
            "processing_method": "llm_structured"
        })
        
        return metadata
    
    except Exception as e:
        return {
            "article_type": "unknown",
            "journal": {"name": "unknown"},
            "publication": {"year": None},
            "study_details": {"population_description": "unknown"},
            "quality_assessment": {"cochrane_risk_bias": "unclear"},
            "processing_confidence": 0.1,
            "processing_error": str(e),
            "processed_at": datetime.now().isoformat(),
            "original_url": article_data.get("url", "")
        }


@tool
def chunk_and_vectorize(
    content: str,
    article_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Chunk article content and generate vector embeddings
    
    Args:
        content: Article content to chunk
        article_id: Unique identifier for the article
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Chunks with embeddings and metadata
    """
    
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create chunks
        chunks = text_splitter.split_text(content)
        
        # Generate embeddings for each chunk
        chunk_embeddings = get_embeddings().embed_documents(chunks)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_obj = {
                "chunk_id": f"{article_id}_chunk_{i}",
                "article_id": article_id,
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding,
                "metadata": {
                    "chunk_size": len(chunk),
                    "chunk_overlap": chunk_overlap,
                    "created_at": datetime.now().isoformat()
                }
            }
            chunk_objects.append(chunk_obj)
        
        return {
            "success": True,
            "article_id": article_id,
            "total_chunks": len(chunk_objects),
            "chunks": chunk_objects,
            "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "processed_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "article_id": article_id,
            "processed_at": datetime.now().isoformat()
        }


@tool
def create_vector_store(chunks_data: List[Dict[str, Any]], store_path: str = None) -> Dict[str, Any]:
    """
    Create FAISS vector store from chunks
    
    Args:
        chunks_data: List of chunk objects with embeddings
        store_path: Path to save the vector store
        
    Returns:
        Vector store information
    """
    
    try:
        if not chunks_data:
            return {"success": False, "error": "No chunks provided"}
        
        # Extract embeddings and metadata
        embeddings_list = []
        metadata_list = []
        
        for chunk_data in chunks_data:
            for chunk in chunk_data.get("chunks", []):
                embeddings_list.append(chunk["embedding"])
                metadata_list.append({
                    "chunk_id": chunk["chunk_id"],
                    "article_id": chunk["article_id"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"]
                })
        
        if not embeddings_list:
            return {"success": False, "error": "No embeddings found"}
        
        # Create FAISS index
        embeddings_array = np.array(embeddings_list).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for cosine similarity
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Save vector store if path provided
        if store_path:
            store_dir = Path(store_path)
            store_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = store_dir / "index.faiss"
            faiss.write_index(index, str(index_path))
            
            # Save metadata
            metadata_path = store_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata_list, f, indent=2)
        
        store_id = str(uuid.uuid4())
        
        return {
            "success": True,
            "store_id": store_id,
            "dimension": dimension,
            "total_vectors": len(embeddings_list),
            "index_path": str(index_path) if store_path else None,
            "metadata_path": str(metadata_path) if store_path else None,
            "created_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "created_at": datetime.now().isoformat()
        }


@tool
def batch_process_articles(
    articles: List[Dict[str, Any]],
    pico: Dict[str, str],
    max_concurrent: int = 3
) -> Dict[str, Any]:
    """
    Process multiple articles in batch with concurrent processing
    
    Args:
        articles: List of articles to process
        pico: PICO framework for context
        max_concurrent: Maximum concurrent processing
        
    Returns:
        Batch processing results
    """
    
    processed_articles = []
    failed_articles = []
    all_chunks = []
    
    for article in articles:
        try:
            # Extract content
            content_result = extract_article_content(article["url"])
            
            if not content_result.get("success"):
                failed_articles.append({
                    "url": article["url"],
                    "error": content_result.get("error", "Unknown error")
                })
                continue
            
            # Extract statistical data
            statistical_data = extract_statistical_data(
                content_result["content"], 
                pico,
                article.get("type", "unknown")
            )
            
            # Generate citation
            citation = generate_vancouver_citation(content_result)
            
            # Process metadata
            metadata = process_article_metadata(content_result)
            
            # Create chunks and embeddings
            article_id = str(uuid.uuid4())
            chunks_result = chunk_and_vectorize(
                content_result["content"],
                article_id
            )
            
            if chunks_result.get("success"):
                all_chunks.append(chunks_result)
            
            # Combine all processed data
            processed_article = {
                "article_id": article_id,
                "original_data": article,
                "content": content_result,
                "statistical_data": statistical_data,
                "citation": citation,
                "metadata": metadata,
                "chunks_info": {
                    "total_chunks": chunks_result.get("total_chunks", 0),
                    "success": chunks_result.get("success", False)
                },
                "processed_at": datetime.now().isoformat()
            }
            
            processed_articles.append(processed_article)
        
        except Exception as e:
            failed_articles.append({
                "url": article.get("url", "unknown"),
                "error": str(e)
            })
    
    # Create vector store from all chunks
    vector_store_result = create_vector_store(all_chunks) if all_chunks else None
    
    return {
        "batch_summary": {
            "total_articles": len(articles),
            "processed_successfully": len(processed_articles),
            "failed": len(failed_articles),
            "success_rate": len(processed_articles) / len(articles) if articles else 0,
            "total_chunks_created": sum(c.get("total_chunks", 0) for c in all_chunks),
            "processed_at": datetime.now().isoformat()
        },
        "processed_articles": processed_articles,
        "failed_articles": failed_articles,
        "vector_store": vector_store_result,
        "pico_context": pico
    }