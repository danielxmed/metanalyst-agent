"""
Processor Agent - Specialized in article extraction, data processing, and vectorization.
Autonomous agent that processes scientific articles using AI-first approach with Tavily and LLMs.
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config.settings import settings
from .base_agent import create_agent_with_state_context
from ..tools.processor_tools import (
    batch_process_articles,
    get_processed_urls_for_analysis,
    get_article_chunks_for_retrieval,
    extract_article_content,
    extract_statistical_data,
    generate_vancouver_citation,
    chunk_and_vectorize,
    extract_with_fallback,
)
from ..tools.handoff_tools import (
    create_handoff_tool,
    request_supervisor_intervention,
    signal_completion,
    request_quality_check,
)


def create_processor_agent():
    """
    Create the Processor Agent specialized in article extraction and processing.
    
    This agent autonomously:
    - Extracts full content from article URLs using Tavily
    - Processes content to extract statistical data using LLMs
    - Generates proper Vancouver citations
    - Creates chunks and vector embeddings
    - Builds and manages vector stores
    - Handles retry logic for failed extractions
    - OPTIMIZED: Uses PostgreSQL for storage and avoids raw content in state
    
    Returns:
        Configured processor agent
    """
    
    # Initialize LLM
    llm = ChatOpenAI(**settings.get_openai_config())
    
    # Processor agent tools - OPTIMIZED
    processor_tools = [
        # Main processing tools
        batch_process_articles,  # OPTIMIZED: No raw content in state
        get_processed_urls_for_analysis,  # NEW: Get processed data from DB
        get_article_chunks_for_retrieval,  # NEW: Get chunks from DB
        # Individual processing tools
        extract_article_content,
        extract_with_fallback,
        extract_statistical_data,
        generate_vancouver_citation,
        chunk_and_vectorize,
        # Handoff tools
        request_supervisor_intervention,
        signal_completion,
        request_quality_check,
    ]

    # System prompt for processor agent - UPDATED
    system_prompt = """
    You are the PROCESSOR AGENT for the Metanalyst-Agent system - an expert in article extraction and data processing.
    
    YOUR EXPERTISE:
    - Web scraping and content extraction using Tavily API
    - Statistical data extraction from medical literature
    - Vancouver citation formatting
    - Text chunking and vectorization strategies
    - Vector store creation and management
    - Quality assessment of extracted data
    - OPTIMIZED: PostgreSQL storage management for large datasets
    
    YOUR RESPONSIBILITIES:
    1. Extract full content from article URLs using Tavily Extract API
    2. Process extracted content to identify and extract statistical data
    3. Generate proper Vancouver-style citations for all articles
    4. Create intelligent text chunks optimized for semantic search
    5. Generate vector embeddings and store them in PostgreSQL
    6. Handle extraction failures with appropriate retry strategies
    7. Assess quality of extracted data and processing success
    8. CRITICAL: Use batch_process_articles with meta_analysis_id to avoid duplicates
    9. CRITICAL: Never store raw article content in the state after vectorization
    
    OPTIMIZATION RULES:
    - Always use batch_process_articles with meta_analysis_id parameter
    - Store raw content and chunks in PostgreSQL, not in state
    - Only keep essential metadata and URLs in state
    - Use get_processed_urls_for_analysis to retrieve data when needed
    - Use get_article_chunks_for_retrieval for semantic search
    
    PROCESSING WORKFLOW:
    1. Receive list of URLs to process from state
    2. Use batch_process_articles(articles, pico, meta_analysis_id) 
    3. This will automatically:
       - Skip already processed URLs
       - Extract content and process it
       - Store chunks in PostgreSQL
       - Return only essential metadata (NO raw content)
    4. Update state with processed URLs summary only
    5. Signal completion or request analysis
    
    QUALITY CONTROL:
    - Monitor processing success rates
    - Handle extraction failures gracefully
    - Validate statistical data extraction quality
    - Ensure proper citation formatting
    - Check vectorization success
    
    HANDOFF CONDITIONS:
    - Transfer to analyst when sufficient articles processed
    - Request supervisor intervention if too many failures
    - Signal completion when batch processing done
    
    Remember: NEVER store raw article content in the shared state after vectorization!
    Use PostgreSQL for persistent storage and keep state lightweight.
    """
    
    # Create the processor agent with state context
    processor_agent = create_agent_with_state_context(
        name="processor",
        system_prompt=system_prompt,
        tools=processor_tools,
        model=llm
    )
    
    return processor_agent
