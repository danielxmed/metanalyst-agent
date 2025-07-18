"""
State definitions for the Metanalyst Agent system.

This module defines the core state structure used throughout the multi-agent
meta-analysis workflow, including the main MetanalysisState and related schemas.
"""

from typing import TypedDict, Annotated, List, Dict, Optional, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator
from datetime import datetime


class MetanalysisState(TypedDict):
    """
    Main state schema for the metanalyst-agent workflow.
    
    This state is shared across all agents and maintains the complete
    context of the meta-analysis process from start to finish.
    """
    
    # Message history for agent communication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Core meta-analysis components
    pico: Optional[Dict[str, str]]  # PICO structure
    user_request: Optional[str]     # Original natural language request
    research_query: Optional[str]   # Generated search query
    
    # Literature search results
    search_results: Annotated[List[Dict], operator.add]
    url_not_processed: Annotated[List[str], operator.add]  # URLs waiting to be processed
    url_processed: Annotated[List[str], operator.add]      # URLs already processed
    
    # Extracted and processed papers (legacy - will be removed)
    extracted_papers: Annotated[List[Dict], operator.add]
    processed_papers: Annotated[List[Dict], operator.add]
    
    # Vector store and retrieval
    vector_store_ready: bool
    vector_store_path: Optional[str]
    chunks_created: Optional[int]
    
    # Report generation
    relevant_chunks: Annotated[List[Dict], operator.add]
    report_draft: Optional[str]
    report_approved: bool
    review_feedback: Optional[Dict[str, Any]]
    
    # Statistical analysis
    statistical_analysis: Optional[Dict[str, Any]]
    forest_plot_path: Optional[str]
    analysis_tables: Annotated[List[Dict], operator.add]
    
    # Final output
    final_report: Optional[str]
    final_report_path: Optional[str]
    
    # Workflow control
    current_step: str
    current_agent: Optional[str]
    step_history: Annotated[List[str], operator.add]
    error_log: Annotated[List[Dict], operator.add]
    
    # Orchestrator control
    max_iterations: Optional[int]
    orchestrator_iterations: Optional[int]
    total_iterations: Optional[int]
    last_decision: Optional[str]
    
    # Metadata
    workflow_id: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    workflow_complete: bool
    total_papers_found: int
    total_papers_processed: int


class WorkflowMetadata(TypedDict):
    """Metadata for tracking workflow execution."""
    workflow_id: str
    started_at: str
    current_step: str
    steps_completed: List[str]
    estimated_completion: Optional[str]
    quality_score: Optional[float]


def create_initial_state(
    pico: Optional[Dict[str, str]] = None,
    workflow_id: Optional[str] = None
) -> MetanalysisState:
    """
    Create an initial state for a new meta-analysis workflow.
    
    Args:
        pico: Optional PICO structure to start with
        workflow_id: Optional workflow identifier
        
    Returns:
        Initial MetanalysisState with default values
    """
    from uuid import uuid4
    
    return MetanalysisState(
        messages=[],
        pico=pico,
        user_request=None,
        research_query=None,
        search_results=[],
        url_not_processed=[],
        url_processed=[],
        extracted_papers=[],
        processed_papers=[],
        vector_store_ready=False,
        vector_store_path=None,
        chunks_created=None,
        relevant_chunks=[],
        report_draft=None,
        report_approved=False,
        review_feedback=None,
        statistical_analysis=None,
        forest_plot_path=None,
        analysis_tables=[],
        final_report=None,
        final_report_path=None,
        current_step="initialize",
        current_agent=None,
        step_history=[],
        error_log=[],
        max_iterations=None,
        orchestrator_iterations=None,
        total_iterations=None,
        last_decision=None,
        workflow_id=workflow_id or str(uuid4()),
        started_at=datetime.now().isoformat(),
        completed_at=None,
        workflow_complete=False,
        total_papers_found=0,
        total_papers_processed=0
    )


def update_state_step(
    state: MetanalysisState, 
    new_step: str, 
    agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Helper function to update workflow step tracking.
    
    Args:
        state: Current state
        new_step: New step identifier
        agent: Optional agent name
        
    Returns:
        State update dictionary
    """
    return {
        "current_step": new_step,
        "current_agent": agent,
        "step_history": [new_step]
    }


def log_error(
    state: MetanalysisState,
    error_type: str,
    error_message: str,
    agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Helper function to log errors in the workflow.
    
    Args:
        state: Current state
        error_type: Type of error
        error_message: Error description
        agent: Optional agent that encountered the error
        
    Returns:
        State update dictionary with error logged
    """
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": error_type,
        "message": error_message,
        "agent": agent,
        "step": state.get("current_step", "unknown")
    }
    
    return {
        "error_log": [error_entry]
    }
