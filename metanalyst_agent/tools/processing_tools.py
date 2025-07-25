"""Processing tools for article content extraction and vectorization"""

import json
import uuid
import hashlib
import time
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Circuit breaker state for tracking URL failures
_url_failure_counts = {}
_url_circuit_breaker = {}  # 'open', 'closed', 'half_open'
_url_last_attempt = {}

def _should_skip_url(url: str, failure_threshold: int = 3) -> bool:
    """Check if URL should be skipped due to circuit breaker pattern"""
    import time
    
    current_time = time.time()
    failure_count = _url_failure_counts.get(url, 0)
    circuit_state = _url_circuit_breaker.get(url, 'closed')
    last_attempt = _url_last_attempt.get(url, 0)
    
    # If circuit is open, check if we should try half-open
    if circuit_state == 'open':
        # Try half-open after 300 seconds (5 minutes)
        if current_time - last_attempt > 300:
            _url_circuit_breaker[url] = 'half_open'
            logger.info(f"Circuit breaker for {url} switching to half-open")
            return False
        else:
            logger.info(f"Circuit breaker for {url} is open, skipping")
            return True
    
    # If too many failures, open the circuit
    if failure_count >= failure_threshold:
        _url_circuit_breaker[url] = 'open'
        _url_last_attempt[url] = current_time
        logger.warning(f"Circuit breaker for {url} opened after {failure_count} failures")
        return True
    
    return False

def _record_url_failure(url: str):
    """Record a failure for URL circuit breaker"""
    _url_failure_counts[url] = _url_failure_counts.get(url, 0) + 1
    _url_last_attempt[url] = time.time()

def _record_url_success(url: str):
    """Record a success for URL circuit breaker"""
    _url_failure_counts[url] = 0
    _url_circuit_breaker[url] = 'closed'


@tool
def extract_article_content(url: str, max_retries: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Extract full article content from URL using Tavily Extract API with retry logic and circuit breaker
    
    Args:
        url: URL of the article to extract
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (exponential backoff)
        
    Returns:
        Dictionary with extracted content, metadata, and processing status
    """
    import time
    import random
    from datetime import datetime
    import os
    
    # Check circuit breaker first
    if _should_skip_url(url):
        logger.warning(f"Skipping {url} due to circuit breaker")
        return {
            "success": False,
            "error": "URL skipped due to circuit breaker (too many previous failures)",
            "url": url,
            "attempts": 0,
            "extracted_at": datetime.now().isoformat(),
            "circuit_breaker_active": True
        }
    
    last_error = None
    
    # Check if Tavily API key is configured
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key or tavily_api_key == "your_tavily_api_key_here":
        logger.error("Tavily API key not configured properly")
        _record_url_failure(url)
        return {
            "success": False,
            "error": "Tavily API key not configured - please set TAVILY_API_KEY environment variable",
            "url": url,
            "attempts": 0,
            "extracted_at": datetime.now().isoformat(),
            "configuration_error": True
        }
    
    for attempt in range(max_retries + 1):
        try:
            # Initialize Tavily client
            client = TavilyClient(api_key=tavily_api_key)
            
            # Use Tavily Extract to get full content (requires list of URLs)
            logger.info(f"Attempting to extract content from {url} (attempt {attempt + 1})")
            
            try:
                # Use proper Tavily Extract API call with parameters
                extract_result = client.extract(
                    urls=[url],
                    extract_depth="advanced",  # Better extraction quality
                    format="markdown",         # Structured format
                    include_images=False,      # We don't need images
                    include_favicon=False      # We don't need favicon
                )
                logger.debug(f"Tavily response type: {type(extract_result)}")
                logger.debug(f"Tavily raw response: {extract_result}")
            except Exception as api_error:
                # Check if error message is just "0"
                if str(api_error).strip() == "0":
                    logger.error(f"Tavily returned '0' error for {url} - skipping retries")
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": "Tavily returned '0' - URL may be unsupported or rate limited",
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat(),
                        "use_fallback": True
                    }
                logger.error(f"Tavily API error: {api_error}")
                raise
            
            # Check if we got proper response structure
            if not isinstance(extract_result, dict):
                error_msg = f"Invalid Tavily response format: expected dict, got {type(extract_result)}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat(),
                        "use_fallback": True
                    }
            
            # Check for failed results first
            failed_results = extract_result.get("failed_results", [])
            if failed_results:
                error_msg = f"Tavily failed to extract content: {failed_results}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat(),
                        "use_fallback": True
                    }
            
            # Get successful results
            results = extract_result.get("results", [])
            if not results or len(results) == 0:
                error_msg = "No content extracted - empty results from Tavily"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat(),
                        "use_fallback": True
                    }
            
            # Get the first result (we only passed one URL)
            result_data = results[0]
            
            # Validate result structure
            if not isinstance(result_data, dict):
                error_msg = f"Invalid result format: expected dict, got {type(result_data)}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat(),
                        "use_fallback": True
                    }
            
            # Check for Tavily error response
            if isinstance(result_data, dict) and result_data.get("error"):
                error_msg = f"Tavily API error: {result_data.get('error')}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat()
                    }
            
            raw_content = result_data.get("raw_content", "") if isinstance(result_data, dict) else str(result_data)
            title = result_data.get("title", "") if isinstance(result_data, dict) else ""
            
            # Handle case where Tavily returns invalid response
            if not isinstance(result_data, dict):
                error_msg = f"Invalid Tavily response format: {type(result_data)}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": "Tavily returned invalid response (possibly rate limited or unsupported URL)",
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat()
                    }
            
            # Validate extracted content
            if not raw_content or len(str(raw_content).strip()) < 100:
                error_msg = f"Insufficient content extracted ({len(raw_content)} chars)"
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {error_msg}")
                
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    _record_url_failure(url)
                    return {
                        "success": False,
                        "error": error_msg,
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat()
                    }
            
            # Generate content hash for deduplication
            content_hash = hashlib.md5(raw_content.encode()).hexdigest()
            
            result = {
                "success": True,
                "url": url,
                "title": title,
                "raw_content": raw_content,
                "content_length": len(raw_content),
                "content_hash": content_hash,
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "tavily_extract",
                "attempts": attempt + 1
            }
            
            # Record success for circuit breaker
            _record_url_success(url)
            
            logger.info(f"Successfully extracted {len(raw_content)} characters from {url} (attempt {attempt + 1})")
            return result
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {last_error}")
            
            # Special handling for common errors
            if "unauthorized" in last_error.lower() or "401" in last_error:
                logger.error("Tavily API authentication failed - check API key")
                _record_url_failure(url)
                return {
                    "success": False,
                    "error": "Tavily API authentication failed - check TAVILY_API_KEY",
                    "url": url,
                    "attempts": attempt + 1,
                    "extracted_at": datetime.now().isoformat(),
                    "authentication_error": True
                }
            
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                _record_url_failure(url)
                return {
                    "success": False,
                    "error": last_error,
                    "url": url,
                    "attempts": attempt + 1,
                    "extracted_at": datetime.now().isoformat()
                }
    
    # This should never be reached, but included for safety
    _record_url_failure(url)
    return {
        "success": False,
        "error": last_error or "Unknown error",
        "url": url,
        "attempts": max_retries + 1,
        "extracted_at": datetime.now().isoformat()
    }


@tool
def extract_statistical_data(
    content: str,
    pico: Dict[str, str],
    article_title: str = ""
) -> Dict[str, Any]:
    """
    Extract statistical data relevant to PICO from article content using LLM
    
    Args:
        content: Full article content
        pico: PICO framework for the meta-analysis
        article_title: Title of the article for context
        
    Returns:
        Dictionary with extracted statistical data and study characteristics
    """
    try:
        llm = ChatOpenAI(model="o3-mini")
        
        # Truncate content if too long (keep first 8000 chars for context)
        if len(content) > 8000:
            content = content[:8000] + "... [content truncated]"
        
        prompt = f"""
        You are an expert biostatistician extracting data for a meta-analysis.
        
        RESEARCH FRAMEWORK:
        Population (P): {pico.get('P', 'Not specified')}
        Intervention (I): {pico.get('I', 'Not specified')}
        Comparison (C): {pico.get('C', 'Not specified')}
        Outcome (O): {pico.get('O', 'Not specified')}
        
        ARTICLE TITLE: {article_title}
        
        ARTICLE CONTENT:
        {content}
        
        Extract the following statistical data and study characteristics:
        
        1. Study design and methodology
        2. Sample sizes (intervention and control groups)
        3. Primary and secondary outcomes with statistical measures
        4. Effect sizes, confidence intervals, p-values
        5. Baseline characteristics of participants
        6. Follow-up duration
        7. Risk of bias indicators
        8. Funding source and conflicts of interest
        
        Focus on data relevant to the PICO framework. If specific values are not available, mark as null.
        
        Respond in JSON format:
        {{
            "study_design": "randomized_controlled_trial",
            "sample_size": {{
                "total": 240,
                "intervention": 120,
                "control": 120,
                "analyzed": 235
            }},
            "population_characteristics": {{
                "mean_age": 45.2,
                "age_range": "18-65",
                "gender_distribution": {{"male": 0.52, "female": 0.48}},
                "inclusion_criteria": ["adults with anxiety", "GAD-7 score > 10"],
                "exclusion_criteria": ["severe depression", "substance abuse"]
            }},
            "intervention_details": {{
                "intervention_name": "Mindfulness-based therapy",
                "duration_weeks": 8,
                "frequency": "weekly sessions",
                "control_name": "Cognitive behavioral therapy"
            }},
            "primary_outcomes": [{{
                "outcome_name": "GAD-7 score reduction",
                "intervention_mean": 12.3,
                "intervention_sd": 3.2,
                "control_mean": 14.1,
                "control_sd": 3.8,
                "mean_difference": -1.8,
                "confidence_interval": [-2.9, -0.7],
                "p_value": 0.002,
                "effect_size_cohens_d": -0.52
            }}],
            "secondary_outcomes": [{{
                "outcome_name": "Quality of life score",
                "intervention_mean": 78.5,
                "intervention_sd": 12.1,
                "control_mean": 72.3,
                "control_sd": 11.8,
                "p_value": 0.031
            }}],
            "follow_up": {{
                "duration_months": 6,
                "retention_rate": 0.89
            }},
            "risk_of_bias": {{
                "randomization": "low",
                "allocation_concealment": "low", 
                "blinding_participants": "high",
                "blinding_assessors": "low",
                "incomplete_data": "low",
                "selective_reporting": "unclear"
            }},
            "funding_source": "National Institute of Mental Health",
            "conflicts_of_interest": "None declared",
            "publication_year": 2023,
            "journal": "Journal of Anxiety Disorders",
            "study_location": "United States",
            "extraction_confidence": 0.85,
            "missing_data_indicators": ["confidence intervals not reported for secondary outcomes"]
        }}
        """
        
        response = llm.invoke(prompt)
        
        try:
            statistical_data = json.loads(response.content)
            
            # Add metadata
            statistical_data.update({
                "extracted_from": article_title,
                "extraction_method": "llm_gpt4o",
                "pico_context": pico,
                "extracted_at": "2024-01-01T00:00:00Z"
            })
            
            logger.info(f"Successfully extracted statistical data from: {article_title[:50]}...")
            return statistical_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse statistical data extraction response")
            return {
                "extraction_failed": True,
                "error": "Failed to parse LLM response",
                "raw_response": response.content[:500],
                "article_title": article_title
            }
    
    except Exception as e:
        logger.error(f"Error extracting statistical data: {str(e)}")
        return {
            "extraction_failed": True,
            "error": str(e),
            "article_title": article_title
        }


@tool
def generate_vancouver_citation(article_data: Dict[str, Any]) -> str:
    """
    Generate Vancouver style citation for an article using LLM
    
    Args:
        article_data: Dictionary with article metadata (title, authors, journal, etc.)
        
    Returns:
        Vancouver formatted citation string
    """
    try:
        llm = ChatOpenAI(model="o3-mini")
        
        prompt = f"""
        Generate a Vancouver style citation for this article.
        
        ARTICLE DATA:
        Title: {article_data.get('title', 'Unknown title')}
        URL: {article_data.get('url', 'No URL')}
        Journal: {article_data.get('journal', 'Unknown journal')}
        Year: {article_data.get('publication_year', 'Unknown year')}
        Authors: {article_data.get('authors', 'Unknown authors')}
        Volume: {article_data.get('volume', 'Unknown')}
        Issue: {article_data.get('issue', 'Unknown')}
        Pages: {article_data.get('pages', 'Unknown')}
        DOI: {article_data.get('doi', 'Unknown')}
        
        Vancouver citation format requirements:
        1. Authors (last name, initials). Maximum 6 authors, then "et al."
        2. Article title (sentence case, no quotes)
        3. Journal name (abbreviated if possible)
        4. Year;Volume(Issue):Pages
        5. DOI if available
        
        If information is missing, extract what you can from the title and URL.
        If authors are not available, use "Anonymous" or extract from content if provided.
        
        Example format:
        Smith J, Johnson AB, Brown CD. Effectiveness of mindfulness therapy for anxiety disorders. J Anxiety Disord. 2023;45(2):123-130. doi:10.1016/j.janxdis.2023.01.001
        
        Return only the citation, no other text.
        """
        
        response = llm.invoke(prompt)
        citation = response.content.strip()
        
        # Basic validation - should contain year and title
        if not citation or len(citation) < 20:
            # Fallback citation
            title = article_data.get('title', 'Unknown title')
            year = article_data.get('publication_year', 'Unknown year')
            url = article_data.get('url', '')
            citation = f"Anonymous. {title}. {year}. Available from: {url}"
        
        logger.info(f"Generated Vancouver citation for: {article_data.get('title', 'Unknown')[:50]}...")
        return citation
        
    except Exception as e:
        logger.error(f"Error generating Vancouver citation: {str(e)}")
        # Minimal fallback citation
        title = article_data.get('title', 'Unknown title')
        url = article_data.get('url', '')
        return f"Citation generation failed. Title: {title}. URL: {url}"


@tool
def chunk_and_vectorize(
    content: str,
    article_metadata: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Chunk article content and generate vector embeddings
    
    Args:
        content: Full article content to chunk
        article_metadata: Metadata about the article (title, url, etc.)
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary with chunks and their embeddings
    """
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split content into chunks
        chunks = text_splitter.split_text(content)
        
        if not chunks:
            return {
                "success": False,
                "error": "No chunks generated",
                "article_url": article_metadata.get("url", "")
            }
        
        # Initialize embeddings model
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        # Generate embeddings for all chunks
        chunk_embeddings = embeddings_model.embed_documents(chunks)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_objects.append({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "text": chunk_text,
                "embedding": embedding,
                "metadata": {
                    "article_title": article_metadata.get("title", ""),
                    "article_url": article_metadata.get("url", ""),
                    "article_hash": article_metadata.get("content_hash", ""),
                    "chunk_size": len(chunk_text),
                    "total_chunks": len(chunks),
                    "created_at": "2024-01-01T00:00:00Z"
                }
            })
        
        result = {
            "success": True,
            "total_chunks": len(chunks),
            "chunk_objects": chunk_objects,
            "embedding_dimension": 1536,
            "article_metadata": article_metadata,
            "chunking_parameters": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        }
        
        logger.info(f"Generated {len(chunks)} chunks and embeddings for: {article_metadata.get('title', 'Unknown')[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error chunking and vectorizing content: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "article_url": article_metadata.get("url", ""),
            "article_title": article_metadata.get("title", "")
        }


@tool  
def extract_article_content(url: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Fallback extraction method when Tavily fails.
    Uses simple web scraping as backup.
    
    Args:
        url: URL to extract content from
        max_retries: Maximum retry attempts
        
    Returns:
        Dictionary with extracted content or error information
    """
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime
    import time
    import random
    
    for attempt in range(max_retries + 1):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Metanalyst-Agent) Scientific Literature Research Bot'
            }
            
            logger.info(f"Fallback extraction attempt {attempt + 1} for {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else "Unknown Title"
            
            # Try to find main content area
            content_selectors = [
                'article', 'main', '.article-content', '.content', 
                '.main-content', '#content', '.post-content'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem and len(content_elem.get_text().strip()) > 100:
                    content = content_elem.get_text().strip()
                    break
            
            # If no main content found, use body
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text().strip()
            
            # Clean up content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '\n'.join(lines)
            
            if len(content) < 100:
                if attempt < max_retries:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Insufficient content extracted ({len(content)} chars)",
                        "url": url,
                        "attempts": attempt + 1,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_method": "fallback_scraping"
                    }
            
            # Generate content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            result = {
                "success": True,
                "url": url,
                "title": title,
                "raw_content": content,
                "content_length": len(content),
                "content_hash": content_hash,
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "fallback_scraping",
                "attempts": attempt + 1
            }
            
            logger.info(f"Fallback extraction successful: {len(content)} characters from {url}")
            return result
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"Fallback attempt {attempt + 1} failed for {url}: {last_error}")
            
            if attempt < max_retries:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                return {
                    "success": False,
                    "error": last_error,
                    "url": url,
                    "attempts": attempt + 1,
                    "extracted_at": datetime.now().isoformat(),
                    "extraction_method": "fallback_scraping"
                }
    
    return {
        "success": False,
        "error": "All fallback attempts failed",
        "url": url,
        "attempts": max_retries + 1,
        "extracted_at": datetime.now().isoformat(),
        "extraction_method": "fallback_scraping"
    }


@tool
def process_article_pipeline(
    url: str,
    pico: Dict[str, str],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Complete article processing pipeline: extract -> analyze -> chunk -> vectorize
    
    Args:
        url: URL of article to process
        pico: PICO framework for context
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Complete processing results including all extracted data
    """
    try:
        # Step 1: Extract content with fallback
        logger.info(f"Starting article processing pipeline for: {url}")
        extraction_result = extract_article_content.invoke({"url": url})
        
        # If Tavily extraction fails, try fallback method
        if not extraction_result.get("success"):
            logger.warning(f"Tavily extraction failed for {url}, trying fallback method")
            extraction_result = extract_article_content.invoke({"url": url})
            
            if not extraction_result.get("success"):
                logger.error(f"Both Tavily and fallback extraction failed for {url}")
                return {
                    "success": False,
                    "error": f"All extraction methods failed: {extraction_result.get('error')}",
                    "url": url,
                    "stage_failed": "extraction",
                    "tavily_error": "Primary extraction failed",
                    "fallback_error": extraction_result.get('error')
                }
        
        # Step 2: Extract statistical data
        statistical_data = extract_statistical_data.invoke({
            "content": extraction_result["raw_content"],
            "pico": pico,
            "article_title": extraction_result["title"]
        })
        
        # Step 3: Generate citation
        citation = generate_vancouver_citation.invoke({
            "article_data": {
                "title": extraction_result["title"],
                "url": url,
                **statistical_data
            }
        })
        
        # Step 4: Chunk and vectorize
        vectorization_result = chunk_and_vectorize.invoke({
            "content": extraction_result["raw_content"],
            "article_metadata": {
                "title": extraction_result["title"],
                "url": url,
                "content_hash": extraction_result["content_hash"]
            },
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        })
        
        if not vectorization_result.get("success"):
            return {
                "success": False,
                "error": f"Vectorization failed: {vectorization_result.get('error')}",
                "url": url,
                "stage_failed": "vectorization",
                "partial_data": {
                    "extraction": extraction_result,
                    "statistical_data": statistical_data,
                    "citation": citation
                }
            }
        
        # Combine all results
        complete_result = {
            "success": True,
            "url": url,
            "processing_completed_at": "2024-01-01T00:00:00Z",
            "extraction_data": extraction_result,
            "statistical_data": statistical_data,
            "vancouver_citation": citation,
            "vectorization_data": vectorization_result,
            "summary": {
                "title": extraction_result["title"],
                "content_length": extraction_result["content_length"],
                "total_chunks": vectorization_result["total_chunks"],
                "has_statistical_data": not statistical_data.get("extraction_failed", False),
                "extraction_confidence": statistical_data.get("extraction_confidence", 0.0)
            }
        }
        
        logger.info(f"Successfully completed processing pipeline for: {extraction_result['title'][:50]}...")
        return complete_result
        
    except Exception as e:
        logger.error(f"Error in article processing pipeline: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "stage_failed": "pipeline"
        }
