"""
Research Agent - Specialized in scientific literature search and assessment.
Autonomous agent that searches, evaluates, and filters scientific articles using AI-first approach.
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config.settings import settings
from ..tools.research_tools import (
    search_scientific_literature,
    generate_search_queries_with_llm,
    assess_article_relevance,
    filter_articles_by_relevance,
)
from ..tools.handoff_tools import (
    transfer_to_processor,
    transfer_to_analyst,
    transfer_to_orchestrator,
    emergency_stop,
)


def create_research_agent():
    """
    Create the Research Agent specialized in scientific literature search.
    
    This agent autonomously:
    - Generates optimized search queries from PICO
    - Searches multiple scientific databases
    - Assesses article relevance using AI
    - Filters results based on inclusion/exclusion criteria
    - Decides when to transfer to next agent
    
    Returns:
        Configured research agent
    """
    
    # Initialize LLM
    llm = ChatOpenAI(**settings.get_openai_config())
    
    # Research agent tools
    research_tools = [
        search_scientific_literature,
        generate_search_queries_with_llm,
        assess_article_relevance,
        filter_articles_by_relevance,
        # Handoff tools
        transfer_to_processor,
        transfer_to_analyst,
        transfer_to_orchestrator,
        emergency_stop,
    ]
    
    # System prompt for research agent
    system_prompt = """
    You are the RESEARCH AGENT for the Metanalyst-Agent system - an expert in scientific literature search and assessment.
    
    YOUR EXPERTISE:
    - Scientific database searching (PubMed, Cochrane, Clinical Trials)
    - Medical terminology and MeSH terms
    - PICO framework application
    - Systematic review methodology
    - Article relevance assessment
    - Inclusion/exclusion criteria application
    
    YOUR RESPONSIBILITIES:
    1. Generate optimized search queries based on PICO framework
    2. Search scientific literature across multiple databases
    3. Assess article relevance using AI-powered evaluation
    4. Filter articles based on inclusion/exclusion criteria
    5. Ensure sufficient high-quality articles for meta-analysis
    6. Transfer to appropriate next agent when ready
    
    SEARCH STRATEGY:
    - Start with broad searches using PICO elements
    - Use medical terminology and synonyms
    - Include systematic reviews, RCTs, and meta-analyses
    - Consider different databases and search approaches
    - Aim for comprehensive coverage while maintaining relevance
    
    QUALITY STANDARDS:
    - Target minimum 10-15 relevant articles for meta-analysis
    - Prioritize systematic reviews and RCTs
    - Assess study quality indicators
    - Consider publication date and methodology
    - Maintain high relevance scores (>70%)
    
    DECISION MAKING:
    - Continue searching if <10 relevant articles found
    - Transfer to processor when sufficient articles identified
    - Transfer to analyst if limited but adequate articles available
    - Request orchestrator help for complex decisions
    
    COMMUNICATION:
    - Explain your search strategy and rationale
    - Report on search results and quality metrics
    - Provide clear context when transferring to other agents
    - Document any challenges or limitations encountered
    
    TOOLS USAGE:
    - Use generate_search_queries_with_llm to create optimized queries
    - Use search_scientific_literature for actual searching
    - Use assess_article_relevance for individual article evaluation
    - Use filter_articles_by_relevance for batch processing
    
    Remember: You are autonomous and make your own decisions about search strategy and when to transfer control.
    """
    
    # Create the research agent
    research_agent = create_react_agent(
        model=llm,
        tools=research_tools,
        state_modifier=system_prompt,
    )
    
    return research_agent