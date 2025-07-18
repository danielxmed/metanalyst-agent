"""
Central Orchestrator Agent for the Metanalyst Agent system.

This module implements the hub of the hub-and-spoke architecture, responsible for
coordinating all specialized agents and managing the meta-analysis workflow.

The orchestrator uses LLM-based decision making instead of hard-coded heuristics,
making it truly agentic and adaptable to different scenarios.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
import json
from datetime import datetime

from ..models.state import MetanalysisState, update_state_step, log_error
from ..models.schemas import PICO, validate_pico
from ..utils.config import get_config
from ..tools.orchestrator_tools import ORCHESTRATOR_TOOLS


def create_orchestrator_agent():
    """
    Create the central orchestrator agent with LLM-based decision making.
    
    This orchestrator uses a sophisticated prompt to guide the LLM in making
    intelligent decisions about which agent to invoke next, rather than
    relying on hard-coded heuristics.
    
    Returns:
        Configured orchestrator agent with all tools
    """
    config = get_config()
    
    # Enhanced system prompt for LLM-based decision making
    system_prompt = f"""
    You are the central orchestrator of an automated medical meta-analysis system.
    
    🎯 YOUR ROLE:
    You coordinate 9 specialized agents in a hub-and-spoke architecture to conduct
    complete medical meta-analyses from PICO definition to final report generation.
    
    🧠 DECISION MAKING:
    You analyze the current workflow state and intelligently decide which agent
    to invoke next. You are NOT following rigid rules - you make contextual
    decisions based on the current state and what makes most sense.
    
    🛠️ AVAILABLE TOOLS:
    1. define_pico_structure - Define research question (PICO structure)
    2. generate_research_query - Create optimized search queries
    3. call_researcher_agent - Search scientific literature
    4. call_processor_agent - Process URLs (extract + vectorize in one step)
    5. call_writer_agent - Generate structured report drafts
    6. call_reviewer_agent - Review report quality
    7. call_analyst_agent - Perform statistical analyses
    8. call_editor_agent - Create final integrated reports
    9. get_workflow_status - Check workflow information
    
    📋 SUGGESTED WORKFLOW SEQUENCE (flexible, not rigid):
    1. Define PICO → 2. Generate query → 3. Search literature → 
    4. Process URLs → 5. Write draft → 6. Review → 7. Analyze → 8. Final edit
    
    🔄 URL PROCESSING WORKFLOW:
    The processor agent combines extraction and vectorization:
    - Receives url_not_processed from state
    - Extracts content using Firecrawl API (focuses on main content)
    - Processes markdown to structured JSON using GPT-4.1-mini
    - Chunks content intelligently (1000 chars, 100 overlap)
    - Generates embeddings with text-embedding-3-small
    - Stores in local vector store
    - Moves URLs from url_not_processed to url_processed
    
    🎭 DECISION PRINCIPLES:
    - ANALYZE the current state thoroughly before deciding
    - PRIORITIZE logical workflow progression
    - HANDLE errors gracefully and retry when appropriate
    - CONSIDER alternative paths when standard flow is blocked
    - MAINTAIN context and explain your reasoning
    - ONLY invoke ONE tool per iteration
    
    💡 INTELLIGENT BEHAVIORS:
    - If a step fails, consider alternative approaches
    - If review suggests more research, go back to researcher
    - If data is insufficient, gather more before proceeding
    - If analysis is incomplete, revisit previous steps
    - Adapt to unexpected situations flexibly
    
    📊 CONTEXT AWARENESS:
    - Max papers per search: {config.search.max_papers_per_search}
    - Vector store path: {config.vector.vector_store_path}
    - Output directory: outputs/
    
    🚨 CRITICAL RULES:
    1. ALWAYS analyze the current state first
    2. EXPLAIN your reasoning for each decision
    3. NEVER follow rigid sequences - be adaptive
    4. HANDLE exceptions and edge cases intelligently
    5. MAINTAIN workflow context throughout
    
    Remember: You are an intelligent orchestrator, not a rule-following automaton.
    Make smart, contextual decisions based on the current situation.
    """
    
    return create_react_agent(
        model=config.llm.primary_model,
        tools=ORCHESTRATOR_TOOLS,
        prompt=system_prompt
    )


def orchestrator_node(state: MetanalysisState) -> Dict[str, Any]:
    """
    Main orchestrator node function for the workflow graph.
    
    This function presents the current state to the LLM orchestrator
    and lets it make intelligent decisions about next steps.
    
    Args:
        state: Current meta-analysis state
        
    Returns:
        State updates based on orchestrator decisions
    """
    try:
        # Create orchestrator agent
        orchestrator = create_orchestrator_agent()
        
        # Prepare state summary for the orchestrator
        state_summary = _create_state_summary(state)
        
        # Create input message with current state
        input_message = HumanMessage(content=f"""
        🔄 WORKFLOW STATUS UPDATE
        
        Current State Analysis:
        {state_summary}
        
        🤔 DECISION REQUIRED:
        Based on the current state above, analyze what needs to be done next
        and invoke the appropriate tool to continue the meta-analysis workflow.
        
        Consider:
        - What has been completed successfully?
        - What is missing or needs attention?
        - What would be the most logical next step?
        - Are there any errors that need handling?
        
        Make your decision and invoke the appropriate tool.
        """)
        
        # Get orchestrator decision
        result = orchestrator.invoke({
            "messages": [input_message]
        })
        
        # Extract the last message (should contain tool results)
        last_message = result["messages"][-1]
        
        # Log the orchestrator's decision
        print(f"🎯 Orchestrator analyzing state and making decision...")
        
        # The orchestrator will have invoked a tool, and the result
        # will be in the last message. We need to parse this and
        # update the state accordingly.
        
        # For now, return a basic state update
        return {
            "current_agent": "orchestrator",
            "last_orchestrator_action": datetime.now().isoformat(),
            "orchestrator_active": True
        }
        
    except Exception as e:
        error_msg = f"Orchestrator execution failed: {str(e)}"
        print(f"❌ {error_msg}")
        return log_error(state, "orchestrator_error", error_msg, "orchestrator")


def _create_state_summary(state: MetanalysisState) -> str:
    """
    Create a comprehensive summary of the current workflow state.
    
    Args:
        state: Current meta-analysis state
        
    Returns:
        Formatted state summary string
    """
    summary_parts = []
    
    # PICO Status
    pico = state.get("pico")
    if pico:
        summary_parts.append(f"✅ PICO Defined: {pico}")
    else:
        summary_parts.append("❌ PICO: Not defined")
    
    # Query Status
    query = state.get("research_query")
    if query:
        summary_parts.append(f"✅ Research Query: {query}")
    else:
        summary_parts.append("❌ Research Query: Not generated")
    
    # Literature Search Status
    urls_found = state.get("urls_found", [])
    if urls_found:
        summary_parts.append(f"✅ URLs Found: {len(urls_found)} papers")
    else:
        summary_parts.append("❌ Literature Search: No URLs found")
    
    # Extraction Status
    urls_processed = state.get("urls_processed", [])
    extracted_papers = state.get("extracted_papers", [])
    if extracted_papers:
        summary_parts.append(f"✅ Papers Extracted: {len(extracted_papers)} papers")
    elif urls_found and not urls_processed:
        summary_parts.append(f"⏳ Extraction Pending: {len(urls_found)} URLs to process")
    else:
        summary_parts.append("❌ Content Extraction: Not started")
    
    # Vectorization Status
    vector_ready = state.get("vector_store_ready", False)
    if vector_ready:
        vector_path = state.get("vector_store_path", "Unknown")
        summary_parts.append(f"✅ Vector Store: Ready at {vector_path}")
    else:
        summary_parts.append("❌ Vector Store: Not created")
    
    # Report Status
    report_draft = state.get("report_draft")
    if report_draft:
        summary_parts.append("✅ Report Draft: Generated")
    else:
        summary_parts.append("❌ Report Draft: Not created")
    
    # Review Status
    review_feedback = state.get("review_feedback")
    report_approved = state.get("report_approved", False)
    if report_approved:
        summary_parts.append("✅ Report Review: Approved")
    elif review_feedback:
        needs_more = review_feedback.get("needs_more_research", False)
        if needs_more:
            summary_parts.append("⚠️ Report Review: Requires more research")
        else:
            summary_parts.append("⏳ Report Review: Feedback received, pending approval")
    else:
        summary_parts.append("❌ Report Review: Not performed")
    
    # Analysis Status
    statistical_analysis = state.get("statistical_analysis")
    if statistical_analysis:
        summary_parts.append("✅ Statistical Analysis: Completed")
    else:
        summary_parts.append("❌ Statistical Analysis: Not performed")
    
    # Final Report Status
    final_report = state.get("final_report")
    if final_report:
        summary_parts.append("✅ Final Report: Completed")
        final_path = state.get("final_report_path", "Unknown")
        summary_parts.append(f"📄 Final Report Path: {final_path}")
    else:
        summary_parts.append("❌ Final Report: Not created")
    
    # Error Status
    error_log = state.get("error_log", [])
    if error_log:
        recent_errors = error_log[-3:]  # Show last 3 errors
        summary_parts.append(f"⚠️ Recent Errors: {len(recent_errors)} errors")
        for error in recent_errors:
            summary_parts.append(f"   - {error.get('type', 'unknown')}: {error.get('message', 'No details')}")
    
    # Workflow Progress
    current_step = state.get("current_step", "unknown")
    current_agent = state.get("current_agent", "unknown")
    workflow_id = state.get("workflow_id", "unknown")
    
    summary_parts.extend([
        f"📍 Current Step: {current_step}",
        f"🤖 Last Agent: {current_agent}",
        f"🆔 Workflow ID: {workflow_id}"
    ])
    
    return "\n".join(summary_parts)


# Create the global orchestrator agent instance
orchestrator_agent = create_orchestrator_agent()


class OrchestratorDecisionEngine:
    """
    Legacy decision engine - kept for backward compatibility.
    
    The new orchestrator uses LLM-based decision making instead
    of hard-coded heuristics, but this class is preserved for
    any existing code that might reference it.
    """
    
    @staticmethod
    def determine_next_agent(state: MetanalysisState) -> str:
        """
        Legacy method - now delegates to LLM-based orchestrator.
        
        Args:
            state: Current meta-analysis state
            
        Returns:
            Name of the next agent (always "orchestrator" now)
        """
        # The new approach always returns "orchestrator" since
        # the LLM handles tool selection internally
        return "orchestrator"
    
    @staticmethod
    def get_decision_rationale(state: MetanalysisState, next_agent: str) -> str:
        """
        Legacy method - provides generic rationale.
        
        Args:
            state: Current state
            next_agent: Chosen next agent
            
        Returns:
            Generic explanation
        """
        return "Using LLM-based intelligent decision making"
