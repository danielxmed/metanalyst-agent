"""Research tools for literature search and relevance assessment"""

import json
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import logging

logger = logging.getLogger(__name__)


@tool
def search_literature(
    query: str,
    max_results: int = 20,
    domains: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search scientific literature using Tavily API with medical domain focus
    
    Args:
        query: Search query for scientific literature
        max_results: Maximum number of results to return
        domains: Specific domains to search (defaults to medical databases)
        
    Returns:
        List of articles with title, URL, snippet, and relevance score
    """
    try:
        # Initialize Tavily client (API key should be in environment)
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        client = TavilyClient(api_key=api_key)
        
        # Default to medical literature domains
        if not domains:
            domains = [
                "pubmed.ncbi.nlm.nih.gov",
                "cochranelibrary.com", 
                "clinicaltrials.gov",
                "bmj.com",
                "thelancet.com",
                "nejm.org",
                "jama.jamanetwork.com"
            ]
        
        # Perform search with medical focus
        results = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=domains,
            include_answer=False,  # We want raw results
            include_raw_content=False  # Just metadata for now
        )
        
        # Structure results for downstream processing
        structured_results = []
        for i, result in enumerate(results.get("results", [])):
            structured_results.append({
                "rank": i + 1,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),
                "score": result.get("score", 0.0),
                "source_domain": extract_domain(result.get("url", "")),
                "published_date": result.get("published_date"),
            })
        
        logger.info(f"Found {len(structured_results)} articles for query: {query[:50]}...")
        return structured_results
        
    except Exception as e:
        logger.error(f"Error searching literature: {str(e)}")
        return []


@tool
def generate_search_queries(pico: Dict[str, str]) -> List[str]:
    """
    Generate optimized search queries based on PICO framework using LLM
    
    Args:
        pico: Dictionary with P (Population), I (Intervention), C (Comparison), O (Outcome)
        
    Returns:
        List of optimized search queries for different databases
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        
        prompt = f"""
        You are an expert medical librarian specializing in systematic reviews and meta-analyses.
        
        Generate 5-7 optimized search queries for medical literature databases based on this PICO framework:
        
        Population (P): {pico.get('P', 'Not specified')}
        Intervention (I): {pico.get('I', 'Not specified')}
        Comparison (C): {pico.get('C', 'Not specified')}
        Outcome (O): {pico.get('O', 'Not specified')}
        
        Requirements:
        1. Use medical terminology and MeSH terms when appropriate
        2. Include both broad and specific queries
        3. Consider synonyms and alternative terms
        4. Focus on randomized controlled trials, systematic reviews, and meta-analyses
        5. Each query should be 10-20 words maximum
        6. Include at least one query combining all PICO elements
        7. Include queries focusing on specific aspects (intervention effectiveness, comparison studies)
        
        Return only a JSON list of query strings, no other text.
        
        Example format: ["query 1", "query 2", "query 3"]
        """
        
        response = llm.invoke(prompt)
        
        # Parse LLM response to extract queries
        try:
            queries = json.loads(response.content)
            if isinstance(queries, list):
                logger.info(f"Generated {len(queries)} search queries from PICO")
                return queries
        except json.JSONDecodeError:
            # Fallback: extract queries from text response
            lines = response.content.strip().split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('[') and not line.startswith(']'):
                    # Remove quotes and numbering
                    query = line.strip('"').strip("'").strip()
                    if query.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.')):
                        query = query[2:].strip()
                    if query:
                        queries.append(query)
            
            if queries:
                logger.info(f"Generated {len(queries)} search queries (fallback parsing)")
                return queries
        
        # Ultimate fallback: generate basic queries
        fallback_queries = []
        if pico.get('P') and pico.get('I'):
            fallback_queries.append(f"{pico['P']} AND {pico['I']}")
        if pico.get('I') and pico.get('O'):
            fallback_queries.append(f"{pico['I']} AND {pico['O']} systematic review")
        if pico.get('I') and pico.get('C'):
            fallback_queries.append(f"{pico['I']} vs {pico['C']} randomized controlled trial")
        
        logger.warning("Using fallback query generation")
        return fallback_queries or [f"{pico.get('I', 'intervention')} meta-analysis"]
        
    except Exception as e:
        logger.error(f"Error generating search queries: {str(e)}")
        # Return basic query as last resort
        intervention = pico.get('I', 'intervention')
        return [f"{intervention} systematic review", f"{intervention} meta-analysis"]


@tool
def assess_article_relevance(
    article: Dict[str, Any],
    pico: Dict[str, str],
    inclusion_criteria: List[str],
    exclusion_criteria: List[str]
) -> Dict[str, Any]:
    """
    Assess article relevance to research question using LLM analysis
    
    Args:
        article: Article metadata (title, snippet, url)
        pico: PICO framework for the meta-analysis
        inclusion_criteria: List of inclusion criteria
        exclusion_criteria: List of exclusion criteria
        
    Returns:
        Assessment with relevance score, reasoning, and recommendation
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
        
        prompt = f"""
        You are an expert systematic reviewer assessing article relevance for a meta-analysis.
        
        RESEARCH FRAMEWORK:
        Population (P): {pico.get('P', 'Not specified')}
        Intervention (I): {pico.get('I', 'Not specified')}
        Comparison (C): {pico.get('C', 'Not specified')}
        Outcome (O): {pico.get('O', 'Not specified')}
        
        INCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in inclusion_criteria) if inclusion_criteria else "- Not specified"}
        
        EXCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in exclusion_criteria) if exclusion_criteria else "- Not specified"}
        
        ARTICLE TO ASSESS:
        Title: {article.get('title', 'No title')}
        Abstract/Snippet: {article.get('snippet', 'No abstract available')}
        Source: {article.get('source_domain', 'Unknown source')}
        
        Please assess this article's relevance and provide:
        1. Relevance score (0-100, where 100 is perfectly relevant)
        2. Brief reasoning for the score
        3. Specific PICO elements addressed
        4. Study type identification (RCT, systematic review, etc.)
        5. Recommendation (include/exclude/needs_full_text_review)
        
        Respond in JSON format:
        {{
            "relevance_score": 85,
            "reasoning": "Brief explanation of relevance assessment",
            "pico_coverage": {{
                "population": true,
                "intervention": true, 
                "comparison": false,
                "outcome": true
            }},
            "study_type": "randomized_controlled_trial",
            "recommendation": "include",
            "quality_indicators": ["large_sample_size", "peer_reviewed"],
            "potential_issues": ["old_study", "limited_population"]
        }}
        """
        
        response = llm.invoke(prompt)
        
        try:
            assessment = json.loads(response.content)
            
            # Add metadata
            assessment.update({
                "article_url": article.get("url", ""),
                "article_title": article.get("title", ""),
                "assessed_at": "2024-01-01T00:00:00Z",  # Would use datetime.now() in real implementation
            })
            
            logger.info(f"Assessed article relevance: {assessment.get('relevance_score', 0)}/100")
            return assessment
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM assessment response")
            return {
                "relevance_score": 50,  # Neutral score
                "reasoning": "Assessment parsing failed",
                "recommendation": "needs_full_text_review",
                "error": "Failed to parse LLM response"
            }
    
    except Exception as e:
        logger.error(f"Error assessing article relevance: {str(e)}")
        return {
            "relevance_score": 0,
            "reasoning": f"Assessment failed: {str(e)}",
            "recommendation": "exclude",
            "error": str(e)
        }


def extract_domain(url: str) -> str:
    """Extract domain from URL for categorization"""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"