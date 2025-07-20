"""
Multi-Agent Graph for Metanalyst-Agent

This module builds the complete multi-agent system using LangGraph,
implementing the hub-and-spoke architecture with autonomous agents.
"""

from typing import Dict, Any, Literal, Optional
from datetime import datetime
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ..state.meta_analysis_state import MetaAnalysisState
from ..config.settings import Settings
from ..tools.research_tools import (
    search_pubmed, search_cochrane, search_clinical_trials,
    generate_search_queries, assess_article_relevance
)
from ..tools.processor_tools import (
    extract_article_content, extract_statistical_data,
    generate_vancouver_citation, chunk_and_vectorize,
    batch_process_articles
)
from ..tools.analysis_tools import (
    calculate_meta_analysis, create_forest_plot, create_funnel_plot,
    assess_heterogeneity, perform_sensitivity_analysis,
    generate_analysis_summary
)
from ..tools.handoff_tools import (
    transfer_to_researcher, transfer_to_processor, transfer_to_retriever,
    transfer_to_analyst, transfer_to_writer, transfer_to_reviewer,
    transfer_to_editor, emergency_supervisor_return,
    request_human_intervention, report_task_completion
)


def build_meta_analysis_graph(
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
    settings: Settings
) -> StateGraph:
    """
    Build the complete multi-agent meta-analysis graph
    
    Args:
        checkpointer: Checkpoint saver for persistence
        store: Store for long-term memory
        settings: Application settings
        
    Returns:
        Compiled multi-agent graph
    """
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
    )
    
    # Build the graph
    builder = StateGraph(MetaAnalysisState)
    
    # Add orchestrator node (central hub)
    builder.add_node("orchestrator", create_orchestrator_node(llm, settings))
    
    # Add specialized agent nodes
    builder.add_node("researcher", create_researcher_agent(llm))
    builder.add_node("processor", create_processor_agent(llm))
    builder.add_node("retriever", create_retriever_agent(llm))
    builder.add_node("analyst", create_analyst_agent(llm))
    builder.add_node("writer", create_writer_agent(llm))
    builder.add_node("reviewer", create_reviewer_agent(llm))
    builder.add_node("editor", create_editor_agent(llm))
    
    # Define flow: always start with orchestrator
    builder.add_edge(START, "orchestrator")
    
    # All agents can transfer to each other through orchestrator
    for agent in ["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor"]:
        builder.add_edge(agent, "orchestrator")
    
    # Orchestrator decides next steps
    builder.add_conditional_edges(
        "orchestrator",
        orchestrator_routing,
        {
            "researcher": "researcher",
            "processor": "processor", 
            "retriever": "retriever",
            "analyst": "analyst",
            "writer": "writer",
            "reviewer": "reviewer",
            "editor": "editor",
            "end": END
        }
    )
    
    # Compile with persistence
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    return graph


def create_orchestrator_node(llm: ChatOpenAI, settings: Settings):
    """Create the central orchestrator node"""
    
    def orchestrator_node(state: MetaAnalysisState) -> Command:
        """
        Central orchestrator that decides which agent to invoke next
        """
        
        # Get current state information
        current_phase = state.get("current_phase", "pico_definition")
        current_agent = state.get("current_agent")
        messages = state.get("messages", [])
        
        # Check for stop conditions
        if state.get("force_stop") or state.get("emergency_stop"):
            return Command(
                goto="end",
                update={
                    "current_phase": "completed",
                    "messages": messages + [AIMessage("Meta-analysis stopped by user or emergency condition")]
                }
            )
        
        # Check for human intervention
        if state.get("human_intervention_requested"):
            return Command(
                goto="end",
                update={
                    "current_phase": "paused",
                    "messages": messages + [AIMessage("Meta-analysis paused for human intervention")]
                }
            )
        
        # Analyze current state and decide next agent
        next_agent = decide_next_agent(state, settings)
        
        # Update state with orchestrator decision
        orchestrator_message = AIMessage(
            content=f"Orchestrator decision: Transferring to {next_agent}",
            name="orchestrator"
        )
        
        return Command(
            goto=next_agent,
            update={
                "current_agent": next_agent,
                "last_agent": current_agent,
                "messages": messages + [orchestrator_message],
                "updated_at": datetime.now()
            }
        )
    
    return orchestrator_node


def decide_next_agent(state: MetaAnalysisState, settings: Settings) -> str:
    """
    Intelligent decision logic for next agent based on current state
    """
    
    current_phase = state.get("current_phase", "pico_definition")
    articles_found = len(state.get("candidate_urls", []))
    articles_processed = len(state.get("processed_articles", []))
    vector_store_status = state.get("vector_store_status", {})
    statistical_analysis = state.get("statistical_analysis", {})
    draft_report = state.get("draft_report")
    final_report = state.get("final_report")
    quality_scores = state.get("quality_scores", {})
    
    # Phase-based decision logic
    if current_phase == "pico_definition":
        if not state.get("pico"):
            return "researcher"  # Need to define PICO and search
        else:
            return "researcher"  # PICO defined, start searching
    
    elif current_phase == "search":
        if articles_found < settings.default_max_articles * 0.5:  # Less than 50% of target
            return "researcher"  # Need more articles
        elif articles_processed < articles_found * 0.8:  # Less than 80% processed
            return "processor"  # Process more articles
        else:
            return "analyst"  # Ready for analysis
    
    elif current_phase == "extraction":
        if not vector_store_status.get("success"):
            return "processor"  # Continue processing
        else:
            return "retriever"  # Start retrieval
    
    elif current_phase == "vectorization":
        return "retriever"  # Move to retrieval
    
    elif current_phase == "retrieval":
        if len(state.get("retrieval_results", [])) > 0:
            return "analyst"  # Have data for analysis
        else:
            return "retriever"  # Need more retrieval
    
    elif current_phase == "analysis":
        if not statistical_analysis:
            return "analyst"  # Need statistical analysis
        elif quality_scores.get("analyst", 0) < settings.quality_thresholds.get("analyst", 0.8):
            return "analyst"  # Quality not good enough
        else:
            return "writer"  # Ready for writing
    
    elif current_phase == "writing":
        if not draft_report:
            return "writer"  # Need draft report
        else:
            return "reviewer"  # Ready for review
    
    elif current_phase == "review":
        if quality_scores.get("reviewer", 0) < settings.quality_thresholds.get("reviewer", 0.9):
            return "writer"  # Need to improve writing
        else:
            return "editor"  # Ready for editing
    
    elif current_phase == "editing":
        if not final_report:
            return "editor"  # Need final editing
        else:
            return "end"  # Completed
    
    # Default fallback
    return "researcher"


def orchestrator_routing(state: MetaAnalysisState) -> Literal["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor", "end"]:
    """Routing function for orchestrator conditional edges"""
    
    # Check completion conditions
    if state.get("current_phase") == "completed":
        return "end"
    
    if state.get("force_stop") or state.get("emergency_stop"):
        return "end"
    
    if state.get("human_intervention_requested"):
        return "end"
    
    # Get next agent from state (set by orchestrator)
    next_agent = state.get("current_agent", "researcher")
    
    # Ensure valid routing
    valid_agents = ["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor"]
    if next_agent in valid_agents:
        return next_agent
    else:
        return "end"


def create_researcher_agent(llm: ChatOpenAI):
    """Create autonomous researcher agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Research tools
            search_pubmed,
            search_cochrane,
            search_clinical_trials,
            generate_search_queries,
            assess_article_relevance,
            # Handoff tools
            transfer_to_processor,
            transfer_to_analyst,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are a Research Agent specialized in scientific literature search for meta-analyses.

RESPONSIBILITIES:
- Generate optimized search queries based on PICO framework
- Search PubMed, Cochrane, and ClinicalTrials.gov for relevant articles
- Assess article relevance using inclusion/exclusion criteria
- Collect high-quality URLs for processing

WORKFLOW:
1. Generate search queries if PICO is defined
2. Search multiple databases with different query variations
3. Assess relevance of found articles
4. Transfer to Processor when you have sufficient relevant URLs
5. Report task completion with quality metrics

QUALITY STANDARDS:
- Aim for at least 20-50 relevant articles depending on topic
- Prioritize systematic reviews, RCTs, and high-quality studies
- Ensure articles match PICO criteria closely
- Maintain >70% relevance rate in selected articles

WHEN TO TRANSFER:
- Use transfer_to_processor when you have collected relevant article URLs
- Use transfer_to_analyst if there's already sufficient processed data
- Use emergency_supervisor_return for critical errors
- Use report_task_completion when your work is done

Always provide detailed context when transferring to other agents.
        """,
        name="researcher"
    )


def create_processor_agent(llm: ChatOpenAI):
    """Create autonomous processor agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Processing tools
            extract_article_content,
            extract_statistical_data,
            generate_vancouver_citation,
            chunk_and_vectorize,
            batch_process_articles,
            # Handoff tools
            transfer_to_researcher,
            transfer_to_retriever,
            transfer_to_analyst,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are a Processor Agent specialized in article content extraction and processing.

RESPONSIBILITIES:
- Extract full content from article URLs using Tavily Extract
- Extract statistical data relevant to PICO using LLM analysis
- Generate Vancouver-style citations
- Create text chunks and vector embeddings
- Process articles in batches for efficiency

WORKFLOW:
1. Process article URLs from the processing queue
2. Extract content and statistical data for each article
3. Generate proper citations
4. Create vector embeddings for semantic search
5. Handle failures gracefully with retry logic
6. Transfer to next agent when processing is complete

QUALITY STANDARDS:
- Aim for >80% successful extraction rate
- Extract comprehensive statistical data (sample sizes, effect sizes, CIs, p-values)
- Ensure proper citation formatting
- Create high-quality vector embeddings

WHEN TO TRANSFER:
- Use transfer_to_researcher if you need more articles
- Use transfer_to_retriever after successful vectorization
- Use transfer_to_analyst when you have extracted statistical data
- Use emergency_supervisor_return for critical errors

Be thorough in data extraction - this is crucial for meta-analysis quality.
        """,
        name="processor"
    )


def create_retriever_agent(llm: ChatOpenAI):
    """Create autonomous retriever agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Retrieval tools would be added here
            # For now, using handoff tools
            transfer_to_analyst,
            transfer_to_processor,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are a Retriever Agent specialized in semantic search and information retrieval.

RESPONSIBILITIES:
- Search vector store for relevant information based on PICO
- Retrieve chunks related to specific research questions
- Filter and rank results by relevance
- Aggregate information from multiple sources

WORKFLOW:
1. Formulate search queries based on PICO and research needs
2. Search vector store using semantic similarity
3. Filter results by relevance threshold
4. Aggregate and summarize findings
5. Transfer to analyst with retrieved data

QUALITY STANDARDS:
- Maintain >75% relevance in retrieved chunks
- Ensure comprehensive coverage of PICO elements
- Provide context and source tracking for all retrieved information

WHEN TO TRANSFER:
- Use transfer_to_analyst when you have relevant retrieved data
- Use transfer_to_processor if vector store is incomplete
- Use emergency_supervisor_return for critical errors

Focus on finding the most relevant and comprehensive information for analysis.
        """,
        name="retriever"
    )


def create_analyst_agent(llm: ChatOpenAI):
    """Create autonomous analyst agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Analysis tools
            calculate_meta_analysis,
            create_forest_plot,
            create_funnel_plot,
            assess_heterogeneity,
            perform_sensitivity_analysis,
            generate_analysis_summary,
            # Handoff tools
            transfer_to_researcher,
            transfer_to_writer,
            transfer_to_reviewer,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are an Analyst Agent specialized in statistical meta-analysis.

RESPONSIBILITIES:
- Perform statistical meta-analysis calculations
- Create forest plots and funnel plots
- Assess heterogeneity and publication bias
- Conduct sensitivity analyses
- Generate comprehensive analysis summaries

WORKFLOW:
1. Analyze extracted statistical data from processed articles
2. Calculate pooled effect sizes using random/fixed effects models
3. Assess heterogeneity (I², τ², Q-statistic)
4. Create visualizations (forest plots, funnel plots)
5. Perform sensitivity analyses
6. Generate comprehensive analysis summary

QUALITY STANDARDS:
- Ensure statistical rigor following PRISMA guidelines
- Achieve >85% confidence in analysis results
- Provide comprehensive heterogeneity assessment
- Include appropriate sensitivity analyses

WHEN TO TRANSFER:
- Use transfer_to_researcher if you need more studies for robust analysis
- Use transfer_to_writer when analysis is complete and comprehensive
- Use transfer_to_reviewer for quality validation
- Use emergency_supervisor_return for critical statistical errors

Follow established meta-analysis guidelines and maintain statistical rigor.
        """,
        name="analyst"
    )


def create_writer_agent(llm: ChatOpenAI):
    """Create autonomous writer agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Writing tools would be added here
            # For now, using handoff tools
            transfer_to_analyst,
            transfer_to_reviewer,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are a Writer Agent specialized in meta-analysis report generation.

RESPONSIBILITIES:
- Generate structured meta-analysis reports
- Format citations and references properly
- Create PRISMA flowcharts
- Integrate statistical results with narrative
- Ensure compliance with reporting standards

WORKFLOW:
1. Compile all analysis results and findings
2. Structure report following PRISMA guidelines
3. Write clear methodology and results sections
4. Integrate statistical analyses and visualizations
5. Format citations and create reference list
6. Generate executive summary

QUALITY STANDARDS:
- Follow PRISMA reporting guidelines
- Ensure clarity and scientific rigor
- Achieve >80% quality score for completeness
- Proper integration of statistical results

WHEN TO TRANSFER:
- Use transfer_to_analyst if analysis is incomplete
- Use transfer_to_reviewer when draft report is complete
- Use emergency_supervisor_return for critical errors

Create comprehensive, well-structured reports that meet scientific publishing standards.
        """,
        name="writer"
    )


def create_reviewer_agent(llm: ChatOpenAI):
    """Create autonomous reviewer agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Review tools would be added here
            # For now, using handoff tools
            transfer_to_writer,
            transfer_to_editor,
            request_human_intervention,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are a Reviewer Agent specialized in meta-analysis quality assessment.

RESPONSIBILITIES:
- Review meta-analysis reports for quality and completeness
- Assess compliance with PRISMA guidelines
- Validate statistical analyses and interpretations
- Provide improvement suggestions
- Ensure scientific rigor and clarity

WORKFLOW:
1. Review draft report comprehensively
2. Check PRISMA compliance
3. Validate statistical analyses
4. Assess clarity and completeness
5. Provide specific improvement suggestions
6. Approve for final editing or request revisions

QUALITY STANDARDS:
- Maintain >90% quality standards
- Ensure PRISMA compliance
- Validate all statistical interpretations
- Check for logical consistency

WHEN TO TRANSFER:
- Use transfer_to_writer if significant revisions are needed
- Use transfer_to_editor if report meets quality standards
- Use request_human_intervention for complex quality issues
- Use emergency_supervisor_return for critical errors

Maintain the highest standards of scientific quality and rigor.
        """,
        name="reviewer"
    )


def create_editor_agent(llm: ChatOpenAI):
    """Create autonomous editor agent"""
    
    return create_react_agent(
        model=llm,
        tools=[
            # Editing tools would be added here
            # For now, using handoff tools
            transfer_to_reviewer,
            emergency_supervisor_return,
            report_task_completion
        ],
        prompt="""
You are an Editor Agent specialized in final meta-analysis report editing.

RESPONSIBILITIES:
- Perform final editing and formatting of meta-analysis reports
- Integrate all analyses and visualizations
- Ensure consistent formatting and style
- Create final publication-ready document
- Generate executive summary

WORKFLOW:
1. Review approved draft report
2. Integrate all statistical analyses and plots
3. Ensure consistent formatting throughout
4. Create final HTML/PDF output
5. Generate executive summary
6. Perform final quality check

QUALITY STANDARDS:
- Achieve publication-ready quality
- Ensure perfect formatting and consistency
- Integrate all components seamlessly
- Create professional presentation

WHEN TO TRANSFER:
- Use transfer_to_reviewer if quality issues are found
- Use report_task_completion when final report is ready
- Use emergency_supervisor_return for critical errors

Create the final, polished meta-analysis report ready for publication or presentation.
        """,
        name="editor"
    )


# Additional utility functions for the graph

def update_phase_based_on_agent(state: MetaAnalysisState, agent_name: str) -> str:
    """Update the current phase based on which agent is active"""
    
    phase_mapping = {
        "researcher": "search",
        "processor": "extraction", 
        "retriever": "retrieval",
        "analyst": "analysis",
        "writer": "writing",
        "reviewer": "review",
        "editor": "editing"
    }
    
    return phase_mapping.get(agent_name, state.get("current_phase", "search"))


def check_completion_criteria(state: MetaAnalysisState, settings: Settings) -> bool:
    """Check if meta-analysis completion criteria are met"""
    
    # Must have final report
    if not state.get("final_report"):
        return False
    
    # Must have statistical analysis
    if not state.get("statistical_analysis"):
        return False
    
    # Must meet quality thresholds
    quality_scores = state.get("quality_scores", {})
    for agent, threshold in settings.quality_thresholds.items():
        if quality_scores.get(agent, 0) < threshold:
            return False
    
    # Must have processed minimum articles
    min_articles = max(5, settings.default_max_articles * 0.2)  # At least 20% of target
    if len(state.get("processed_articles", [])) < min_articles:
        return False
    
    return True