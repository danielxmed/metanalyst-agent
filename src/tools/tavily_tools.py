"""
Tavily Search and Extract tools for the Metanalyst Agent system.

This module provides integration with Tavily API for web search and content extraction,
specifically optimized for medical literature research with domain filtering.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from langchain_core.tools import tool
import json
from datetime import datetime
import logging

from ..utils.config import get_config

# Try to import tavily, handle if not available
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("Tavily client not available. Install with: pip install tavily-python")

# Medical literature domains for focused search
MEDICAL_LITERATURE_DOMAINS = [
    # Primary medical databases
    "pubmed.ncbi.nlm.nih.gov", 
    "www.ncbi.nlm.nih.gov/pmc", 
    "cochranelibrary.com",
    "lilacs.bvsalud.org", 
    "scielo.org", 
    "www.embase.com", 
    "www.webofscience.com",
    "www.scopus.com", 
    "www.epistemonikos.org", 
    "www.ebscohost.com",
    "www.tripdatabase.com", 
    "pedro.org.au", 
    "doaj.org", 
    "scholar.google.com",
    
    # Clinical trials databases
    "clinicaltrials.gov", 
    "apps.who.int/trialsearch", 
    "www.clinicaltrialsregister.eu",
    "www.isrctn.com",
    
    # High-impact medical journals
    "www.thelancet.com", 
    "www.nejm.org", 
    "jamanetwork.com",
    "www.bmj.com", 
    "www.nature.com/nm", 
    "www.acpjournals.org/journal/aim",
    "journals.plos.org/plosmedicine", 
    "www.jclinepi.com",
    "systematicreviewsjournal.biomedcentral.com", 
    "ascopubs.org/journal/jco",
    "www.ahajournals.org/journal/circ", 
    "www.gastrojournal.org",
    "academic.oup.com/eurheartj", 
    "www.archives-pmr.org", 
    "www.jacc.org",
    
    # Brazilian medical literature
    "www.scielo.br",
    "scielo.br/j/csp/",
    "cadernos.ensp.fiocruz.br",
    "scielo.br/j/rsp/",
    "scielo.org/journal/rpsp/",
    "journal.paho.org",
    "rbmt.org.br",
    "revistas.usp.br/rmrp",
    "memorias.ioc.fiocruz.br",
    
    # Additional sources
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "acpjournals.org/journal/aim",
    "cacancerjournal.com",
    "nature.com/nm",
    "cell.com/cell-metabolism/home",
    "thelancet.com/journals/langlo/home",
    "ncbi.nlm.nih.gov/pmc",
    "scopus.com",
    "webofscience.com",
    "bvsalud.org",
    "jbi.global",
    "tripdatabase.com",
    "gov.br",
    "droracle.ai",
    "wolterskluwer.com",
    "semanticscholar.org",
    "globalindexmedicus.net",
    "sciencedirect.com",
    "openevidence.com"
]


def get_tavily_client() -> Optional[TavilyClient]:
    """
    Get configured Tavily client.
    
    Returns:
        TavilyClient instance or None if not available
    """
    if not TAVILY_AVAILABLE:
        return None
    
    config = get_config()
    api_key = config.api.tavily_api_key
    
    if not api_key:
        logging.error("Tavily API key not found in configuration")
        return None
    
    return TavilyClient(api_key=api_key)


@tool
def search_medical_literature(
    query: str,
    max_results: int = 15,
    search_depth: str = "basic",
    time_range: Optional[str] = None,
    include_additional_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> str:
    """
    Search medical literature using Tavily with optimized parameters for medical research.
    
    This tool searches across reliable medical databases and journals, with parameters
    specifically tuned for meta-analysis literature research.
    
    Args:
        query: Search query for medical literature
        max_results: Maximum number of results (5-20, default 15)
        search_depth: Depth of search - 'basic' or 'advanced' (default 'basic')
        time_range: Time filter - 'day', 'week', 'month', 'year' (optional)
        include_additional_domains: Extra domains to include beyond medical defaults
        exclude_domains: Domains to exclude from search
        
    Returns:
        JSON string with search results including URLs, titles, and content snippets
    """
    try:
        client = get_tavily_client()
        if not client:
            return json.dumps({
                "success": False,
                "error": "Tavily client not available or API key missing",
                "results": []
            })
        
        # Validate parameters
        max_results = max(5, min(20, max_results))  # Clamp between 5-20
        if search_depth not in ["basic", "advanced"]:
            search_depth = "basic"
        
        # Prepare domain lists
        include_domains = MEDICAL_LITERATURE_DOMAINS.copy()
        if include_additional_domains:
            include_domains.extend(include_additional_domains)
        
        # Build search parameters
        search_params = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_domains": include_domains,
            "topic": "general"
        }
        
        # Add optional parameters
        if time_range:
            search_params["time_range"] = time_range
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains
        
        # Execute search
        response = client.search(**search_params)
        
        # Process results
        processed_results = []
        urls_found = []
        
        for result in response.get('results', []):
            processed_result = {
                "title": result.get('title', 'No title'),
                "url": result.get('url', ''),
                "content": result.get('content', ''),
                "score": result.get('score', 0),
                "published_date": result.get('published_date', ''),
                "raw_content": result.get('raw_content', '')
            }
            processed_results.append(processed_result)
            urls_found.append(result.get('url', ''))
        
        # Compile response
        search_result = {
            "success": True,
            "query_used": query,
            "total_results": len(processed_results),
            "search_depth": search_depth,
            "max_results": max_results,
            "urls_found": urls_found,
            "results": processed_results,
            "search_timestamp": datetime.now().isoformat(),
            "domains_searched": len(include_domains),
            "parameters_used": search_params
        }
        
        return json.dumps(search_result, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "query": query,
            "results": [],
            "error_timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_result, ensure_ascii=False)


@tool
def extract_paper_content(
    urls: List[str],
    extract_depth: Literal["basic", "advanced"] = "advanced",
    format_type: Literal["markdown", "text"] = "markdown"
) -> str:
    """
    Extract content from medical paper URLs using Tavily.
    
    This tool extracts full content from medical literature URLs, optimized
    for structured content like tables and embedded data.
    
    Args:
        urls: List of URLs to extract content from
        extract_depth: Extraction depth - 'basic' or 'advanced' (default 'advanced')
        format_type: Content format - 'markdown' or 'text' (default 'markdown')
        
    Returns:
        JSON string with extracted papers and structured content
    """
    try:
        client = get_tavily_client()
        if not client:
            return json.dumps({
                "success": False,
                "error": "Tavily client not available or API key missing",
                "extracted_papers": []
            })
        
        # Validate parameters
        validated_depth: Literal["basic", "advanced"] = "advanced"
        if extract_depth in ["basic", "advanced"]:
            validated_depth = extract_depth
            
        validated_format: Literal["markdown", "text"] = "markdown"
        if format_type in ["markdown", "text"]:
            validated_format = format_type
        
        # Extract content
        response = client.extract(
            urls=urls,
            extract_depth=validated_depth,
            format=validated_format
        )
        
        # Process extracted content
        extracted_papers = []
        successfully_processed = []
        failed_urls = []
        
        for result in response.get('results', []):
            url = result.get('url', '')
            
            try:
                # Process each paper
                paper = {
                    "url": url,
                    "title": result.get('title', ''),
                    "raw_content": result.get('raw_content', ''),
                    "content": result.get('content', ''),
                    "extraction_status": "success",
                    "content_length": len(result.get('raw_content', '')),
                    "format": validated_format,
                    "extract_depth": validated_depth,
                    "extracted_at": datetime.now().isoformat()
                }
                
                extracted_papers.append(paper)
                successfully_processed.append(url)
                
            except Exception as e:
                failed_urls.append({"url": url, "error": str(e)})
        
        # Compile response
        extraction_result = {
            "success": True,
            "total_urls": len(urls),
            "successfully_extracted": len(successfully_processed),
            "failed_extractions": len(failed_urls),
            "extracted_papers": extracted_papers,
            "successfully_processed_urls": successfully_processed,
            "failed_urls": failed_urls,
            "extraction_parameters": {
                "extract_depth": validated_depth,
                "format": validated_format
            },
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(extraction_result, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Content extraction failed: {str(e)}",
            "urls": urls,
            "extracted_papers": [],
            "error_timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_result, ensure_ascii=False)


@tool
def refine_medical_query(
    pico_components: Dict[str, str],
    search_type: str = "systematic_review"
) -> str:
    """
    Generate optimized medical search queries from PICO components.
    
    Creates multiple search query variations specifically designed for medical
    literature searches, incorporating best practices for systematic reviews.
    
    Args:
        pico_components: Dictionary with patient, intervention, comparison, outcome
        search_type: Type of search - 'systematic_review', 'meta_analysis', 'clinical_trial'
        
    Returns:
        JSON string with optimized queries and search strategies
    """
    try:
        patient = pico_components.get('patient', '').strip()
        intervention = pico_components.get('intervention', '').strip()
        comparison = pico_components.get('comparison', '').strip()
        outcome = pico_components.get('outcome', '').strip()
        
        # Base query variations
        queries = []
        
        # Primary structured query
        if all([patient, intervention, outcome]):
            primary_query = f'("{patient}") AND ("{intervention}") AND ("{outcome}")'
            if comparison:
                primary_query += f' AND ("{comparison}")'
            queries.append({
                "query": primary_query,
                "type": "structured_primary",
                "description": "Primary structured query with all PICO components"
            })
        
        # Alternative query formats
        if patient and intervention and outcome:
            # Natural language query
            natural_query = f"{intervention} in {patient} {outcome}"
            if comparison:
                natural_query += f" versus {comparison}"
            queries.append({
                "query": natural_query,
                "type": "natural_language",
                "description": "Natural language query"
            })
            
            # MeSH-style query
            mesh_query = f"{patient} {intervention} {outcome}"
            queries.append({
                "query": mesh_query,
                "type": "mesh_style",
                "description": "MeSH-style query for medical databases"
            })
        
        # Search type specific modifications
        if search_type == "systematic_review":
            for query in queries:
                query["query"] += ' AND ("systematic review" OR "meta-analysis")'
        elif search_type == "meta_analysis":
            for query in queries:
                query["query"] += ' AND ("meta-analysis" OR "pooled analysis")'
        elif search_type == "clinical_trial":
            for query in queries:
                query["query"] += ' AND ("clinical trial" OR "randomized controlled trial" OR "RCT")'
        
        # Additional search terms suggestions
        search_suggestions = {
            "inclusion_terms": [
                "randomized controlled trial",
                "systematic review",
                "meta-analysis",
                "clinical trial",
                "prospective study",
                "cohort study"
            ],
            "exclusion_terms": [
                "case report",
                "editorial",
                "letter",
                "comment",
                "animal study"
            ]
        }
        
        result = {
            "success": True,
            "pico_input": pico_components,
            "search_type": search_type,
            "generated_queries": queries,
            "recommended_primary": queries[0]["query"] if queries else "",
            "search_suggestions": search_suggestions,
            "query_count": len(queries),
            "generated_at": datetime.now().isoformat()
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Query refinement failed: {str(e)}",
            "pico_input": pico_components,
            "generated_queries": [],
            "error_timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_result, ensure_ascii=False)


# Tools for the researcher agent (only search)
RESEARCHER_TOOLS = [
    search_medical_literature
]

# Tools for the extractor agent (only extract)
EXTRACTOR_TOOLS = [
    extract_paper_content
]

# List of all Tavily tools for easy import
TAVILY_TOOLS = [
    search_medical_literature,
    extract_paper_content,
    refine_medical_query
]
