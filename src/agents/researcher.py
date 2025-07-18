"""
Researcher Agent for the Metanalyst Agent system.

This agent specializes in searching medical literature using Tavily API,
focusing specifically on finding relevant URLs for meta-analysis.
The researcher's role is to search and return URLs, not to extract content.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import json
from datetime import datetime

from ..models.state import MetanalysisState, update_state_step, log_error
from ..models.schemas import PICO, validate_pico
from ..utils.config import get_config
from ..tools.tavily_tools import search_medical_literature


def create_researcher_agent():
    """
    Create the researcher agent specialized in medical literature URL search.
    
    This agent uses Tavily API with medical domain filtering to search
    for relevant paper URLs based on PICO structure. It does NOT extract content.
    
    Returns:
        Configured researcher agent with search tool only
    """
    config = get_config()
    
    # Simplified system prompt focused on URL discovery with semantic search guidance
    system_prompt = f"""
    You are an expert medical literature researcher specializing in finding relevant URLs for meta-analyses.
    
    🎯 YOUR SOLE RESPONSIBILITY:
    Find and return URLs of relevant medical papers based on PICO structure.
    You do NOT extract content - that's another agent's job.
    
    🛠️ YOUR TOOL:
    - search_medical_literature: Search across medical databases for relevant URLs
    
    🔍 SEARCH APPROACH - SEMANTIC/VECTORIAL:
    IMPORTANT: You're searching with Tavily, which uses semantic/vectorial search, NOT MeSH terms like PubMed!
    
    ✅ USE natural language queries that describe concepts clearly
    ✅ USE descriptive terms and synonyms
    ✅ COMBINE concepts with natural language
    ❌ DON'T use formal MeSH terminology
    ❌ DON'T use complex Boolean operators excessively
    ❌ DON'T rely on database-specific syntax
    
    🎯 SEMANTIC QUERY STRATEGY:
    1. Analyze the PICO structure provided
    2. Create natural, descriptive search queries that capture the essence of the research question
    3. Use search_medical_literature with semantic queries
    4. Think "How would a researcher describe this study?" rather than "What are the exact medical terms?"
    
    📊 FOCUS ON QUALITY SOURCES:
    - PubMed/PMC (primary medical database)
    - Cochrane Library (systematic reviews)
    - High-impact journals (NEJM, Lancet, BMJ, JAMA)
    - Clinical trial databases
    - Embase, Web of Science, Scopus
    - Brazilian medical literature (SciELO)
    
    🎯 STUDY TYPES TO PRIORITIZE:
    - Randomized controlled trials (RCTs)
    - Systematic reviews and meta-analyses
    - Cohort studies
    - Case-control studies
    - Clinical trials
    
    🚫 AVOID:
    - Case reports and case series
    - Editorial content and opinions
    - Animal studies (unless specifically requested)
    - Conference abstracts without full papers
    
    ⚙️ SEARCH PARAMETERS:
    - Max results: {config.search.max_papers_per_search}
    - Search depth: basic (for better latency)
    - Medical domains: pre-configured list of trusted sources
    
    🎭 YOUR WORKFLOW:
    1. Read the PICO components
    2. Generate natural language queries that semantically capture the research question
    3. Execute search_medical_literature with the best semantic query
    4. Return a list of relevant URLs for the next agent
    5. Provide brief summary of search strategy used
    
    💡 SEMANTIC QUERY EXAMPLES:
    - Instead of: "Diabetes Mellitus, Type 2"[Mesh] AND "Metformin"[Mesh]
    - Use: "type 2 diabetes metformin treatment efficacy outcomes"
    - Instead of: "Myocardial Infarction"[Mesh] AND "Aspirin"[Mesh] 
    - Use: "heart attack aspirin prevention cardiovascular outcomes"
    
    Remember: You find URLs using semantic search, you don't extract content. Be descriptive and natural!
    """
    
    return create_react_agent(
        model=config.llm.primary_model,
        tools=[search_medical_literature],
        prompt=system_prompt
    )


def researcher_node(state: MetanalysisState) -> Dict[str, Any]:
    """
    Researcher node function for the workflow graph.
    
    Executes literature search based on PICO structure, focusing only on
    finding relevant URLs. Does NOT extract content.
    
    Args:
        state: Current meta-analysis state containing PICO and query info
        
    Returns:
        State updates with found URLs
    """
    try:
        # Create researcher agent
        researcher = create_researcher_agent()
        
        # Extract PICO and query from state
        pico = state.get("pico", {})
        research_query = state.get("research_query", "")
        
        # Validate inputs
        if not pico:
            error_msg = "PICO structure not found in state"
            print(f"❌ {error_msg}")
            return log_error(state, "missing_pico", error_msg, "researcher")
        
        if not research_query:
            error_msg = "Research query not found in state"
            print(f"❌ {error_msg}")
            return log_error(state, "missing_query", error_msg, "researcher")
        
        # Prepare simplified research instruction
        research_instruction = f"""
        🔍 URL SEARCH REQUEST
        
        PICO Structure:
        - Patient/Population: {pico.get('patient', 'N/A')}
        - Intervention: {pico.get('intervention', 'N/A')}
        - Comparison: {pico.get('comparison', 'N/A')}
        - Outcome: {pico.get('outcome', 'N/A')}
        
        Research Query: "{research_query}"
        
        🎯 YOUR TASK:
        Use search_medical_literature to find relevant URLs for this research question.
        
        IMPORTANT: You only need to find URLs - do NOT extract content!
        The extraction will be done by another agent later.
        
        💡 SEARCH APPROACH:
        1. Generate an optimized search query based on the PICO components
        2. Use search_medical_literature to find candidate URLs
        3. Return the URLs for the extraction agent to process later
        4. Focus on high-quality medical sources (RCTs, systematic reviews, meta-analyses)
        
        Execute the search now and return the URLs found.
        """
        
        print(f"🔍 Starting URL search for: {research_query}")
        print(f"📋 PICO: P={pico.get('patient', '')[:30]}... I={pico.get('intervention', '')[:30]}...")
        
        # Execute researcher agent
        result = researcher.invoke({
            "messages": [HumanMessage(content=research_instruction)]
        })
        
        # Extract URLs from tool calls in the result
        urls_found = []
        search_summary = ""
        
        # Parse the messages to extract tool call results
        for message in result["messages"]:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call["name"] == "search_medical_literature":
                        # The tool returns JSON string with results
                        try:
                            tool_result = json.loads(message.content) if hasattr(message, 'content') else {}
                            if isinstance(tool_result, dict) and "urls_found" in tool_result:
                                urls_found.extend(tool_result["urls_found"])
                        except:
                            pass
            
            # Also check regular message content for search results
            if hasattr(message, 'content') and message.content:
                try:
                    # Convert content to string if it's a list
                    content_str = message.content
                    if isinstance(message.content, list):
                        content_str = "".join([
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in message.content
                        ])
                    
                    # Try to parse as JSON to extract URLs
                    content_data = json.loads(content_str)
                    if isinstance(content_data, dict) and "urls_found" in content_data:
                        urls_found.extend(content_data["urls_found"])
                        search_summary = content_data.get("query_used", "")
                except:
                    # If not JSON, extract text summary
                    content_str = message.content
                    if isinstance(message.content, list):
                        content_str = "".join([
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in message.content
                        ])
                    
                    if isinstance(content_str, str) and "search" in content_str.lower():
                        search_summary = content_str[:200] + "..."
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls_found:
            if url and url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        print(f"✅ Found {len(unique_urls)} unique URLs")
        
        # Update state with URLs found
        update_data = {
            "urls_found": unique_urls,
            "search_summary": search_summary,
            "current_agent": "researcher",
            "last_researcher_action": datetime.now().isoformat(),
            "total_urls_found": len(unique_urls)
        }
        
        # Add step update
        step_update = update_state_step(state, "urls_found", "researcher")
        update_data.update(step_update)
        
        return update_data
        
    except Exception as e:
        error_msg = f"Researcher execution failed: {str(e)}"
        print(f"❌ {error_msg}")
        return log_error(state, "researcher_error", error_msg, "researcher")


def evaluate_urls_quality(urls: List[str]) -> Dict[str, Any]:
    """
    Evaluate the quality of found URLs based on domain reliability.
    
    Args:
        urls: List of URLs found
        
    Returns:
        Quality evaluation of the URLs
    """
    try:
        if not urls:
            return {
                "quality_score": 0.0,
                "recommendations": ["No URLs found"],
                "issues": ["Empty URL list"]
            }
        
        # Quality domains for medical research
        quality_domains = {
            'pubmed.ncbi.nlm.nih.gov': 10,
            'www.ncbi.nlm.nih.gov/pmc': 10,
            'cochranelibrary.com': 9,
            'www.nejm.org': 9,
            'www.thelancet.com': 9,
            'jamanetwork.com': 9,
            'www.bmj.com': 8,
            'www.nature.com': 8,
            'clinicaltrials.gov': 8,
            'scielo.org': 7,
            'scholar.google.com': 6
        }
        
        total_urls = len(urls)
        quality_score = 0.0
        domain_distribution = {}
        
        for url in urls:
            domain = extract_domain(url)
            domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            
            # Add quality score based on domain
            for quality_domain, score in quality_domains.items():
                if quality_domain in url:
                    quality_score += score
                    break
            else:
                quality_score += 3  # Default score for unknown domains
        
        # Normalize score (0-10)
        normalized_score = min(10.0, quality_score / total_urls)
        
        # Generate recommendations
        recommendations = []
        issues = []
        
        if total_urls < 10:
            issues.append(f"Low URL count: {total_urls}")
            recommendations.append("Consider broader search terms")
        
        if normalized_score < 6.0:
            issues.append("Many low-quality domains")
            recommendations.append("Focus on high-impact medical databases")
        
        high_quality_count = sum(1 for url in urls if any(qd in url for qd in quality_domains.keys()))
        if high_quality_count < total_urls * 0.7:
            issues.append("Few high-quality sources")
            recommendations.append("Refine search to prioritize peer-reviewed sources")
        
        return {
            "quality_score": normalized_score,
            "total_urls": total_urls,
            "high_quality_count": high_quality_count,
            "domain_distribution": domain_distribution,
            "recommendations": recommendations,
            "issues": issues,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "quality_score": 0.0,
            "error": f"URL quality evaluation failed: {str(e)}",
            "recommendations": ["Unable to evaluate URL quality"],
            "issues": ["Evaluation error"]
        }


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain name
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or "unknown"
    except Exception:
        return "unknown"


# Create the global researcher agent instance
researcher_agent = create_researcher_agent()
