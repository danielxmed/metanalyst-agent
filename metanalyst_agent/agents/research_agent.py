"""
Research Agent - Specialized in scientific literature search and assessment.
Autonomous agent that searches, evaluates, and filters scientific articles using AI-first approach.
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config.settings import settings
from .base_agent import create_agent_with_state_context
from ..tools.research_tools import (
    search_literature,
    generate_search_queries,
    assess_article_relevance,

)
from ..tools.handoff_tools import (
    create_handoff_tool,
    transfer_to_processor,
    transfer_to_analyst,
    request_supervisor_intervention,
    signal_completion,
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
    
    # Create handoff tools
    transfer_to_processor = create_handoff_tool(
        agent_name="processor",
        description="Transfer control to processor agent for data extraction and quality assessment"
    )
    
    transfer_to_analyst = create_handoff_tool(
        agent_name="analyst", 
        description="Transfer control to analyst agent for statistical analysis"
    )
    
    transfer_to_orchestrator = create_handoff_tool(
        agent_name="orchestrator",
        description="Transfer control back to orchestrator for coordination"
    )
    
    # Research agent tools
    research_tools = [
        search_literature,
        generate_search_queries,
        assess_article_relevance,
        # Handoff tools
        transfer_to_processor,
        transfer_to_analyst,
        transfer_to_orchestrator,
        request_supervisor_intervention,
        signal_completion,

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
    - Use generate_search_queries with the PICO framework provided in your context to create optimized queries
    - Extract PICO data from the current state context and pass it to generate_search_queries
    - Use search_literature for actual searching with the generated queries
    - Use assess_article_relevance for individual article evaluation
    
    IMPORTANT: When calling generate_search_queries, you MUST provide the PICO framework from your context.
    The PICO will be provided in the system context - extract it and pass it to the tool.
    
    Remember: You are autonomous and make your own decisions about search strategy and when to transfer control.
    """
    
    # Create the research agent with state context
    research_agent = create_agent_with_state_context(
        name="researcher",
        system_prompt=system_prompt,
        tools=research_tools,
        model=llm
    )
    
    return research_agent
