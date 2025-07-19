"""
Meta-Analysis State Definition for Metanalyst-Agent

This module defines the shared state structure used across all agents
in the meta-analysis system, following LangGraph best practices.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from pydantic import BaseModel


class MetaAnalysisState(TypedDict):
    """
    Complete shared state for the meta-analysis system.
    
    This state is shared between all agents and contains all information
    needed to coordinate the meta-analysis process.
    """
    
    # Core identification and control
    meta_analysis_id: str
    thread_id: str
    current_phase: Literal[
        "pico_definition", "search", "extraction", 
        "vectorization", "retrieval", "analysis", 
        "writing", "review", "editing", "completed"
    ]
    current_agent: Optional[str]
    last_agent: Optional[str]
    
    # PICO framework and research definition
    pico: Dict[str, str]  # {P: Population, I: Intervention, C: Comparison, O: Outcome}
    research_question: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    search_strategy: Dict[str, Any]
    
    # Literature search and URLs
    search_queries: List[str]
    search_domains: List[str]
    candidate_urls: List[Dict[str, Any]]  # [{url, title, relevance_score, source}]
    processing_queue: List[str]
    processed_articles: List[Dict[str, Any]]
    failed_urls: List[Dict[str, str]]  # [{url, error, attempts}]
    
    # Content extraction and processing
    extracted_data: List[Dict[str, Any]]  # Structured data from articles
    vancouver_citations: List[str]
    article_summaries: List[Dict[str, Any]]
    
    # Vector store and retrieval
    vector_store_id: Optional[str]
    vector_store_status: Dict[str, Any]
    chunk_count: int
    embedding_model_used: str
    
    # Information retrieval
    retrieval_queries: List[str]
    retrieval_results: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    
    # Statistical analysis and results
    statistical_analysis: Dict[str, Any]
    forest_plots: List[Dict[str, Any]]
    funnel_plots: List[Dict[str, Any]]
    quality_assessments: Dict[str, float]
    heterogeneity_analysis: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    
    # Report generation
    draft_report: Optional[str]
    report_sections: Dict[str, str]  # {section_name: content}
    review_feedback: List[Dict[str, str]]
    final_report: Optional[str]
    report_metadata: Dict[str, Any]
    
    # Quality control and iteration
    quality_scores: Dict[str, float]  # Quality scores by component
    quality_thresholds: Dict[str, float]
    improvement_rates: Dict[str, float]
    
    # Agent iteration control
    agent_iterations: Dict[str, int]
    agent_limits: Dict[str, int]
    global_iterations: int
    max_global_iterations: int
    remaining_steps: RemainingSteps
    
    # Circuit breaker and retry
    circuit_status: Dict[str, Literal["closed", "open", "half_open"]]
    failure_counts: Dict[str, int]
    retry_counts: Dict[str, int]
    last_failure_time: Dict[str, float]
    
    # Messages and communication
    messages: Annotated[List[BaseMessage], add_messages]
    agent_logs: List[Dict[str, Any]]  # Detailed logs per agent
    handoff_history: List[Dict[str, Any]]  # Agent transition history
    
    # Checkpoints and recovery
    checkpoints: List[Dict[str, Any]]
    last_checkpoint_time: float
    checkpoint_frequency: int
    
    # Control flags
    force_stop: bool
    quality_satisfied: bool
    deadline_reached: bool
    max_retries_reached: bool
    emergency_stop: bool
    
    # Metadata and metrics
    created_at: datetime
    updated_at: datetime
    total_articles_found: int
    total_articles_processed: int
    total_articles_included: int
    processing_success_rate: float
    execution_time: Dict[str, float]  # Time per phase
    
    # Configuration
    config: Dict[str, Any]  # Runtime configuration parameters


class MetaAnalysisResult(BaseModel):
    """
    Final result structure for a completed meta-analysis
    """
    
    # Identification
    meta_analysis_id: str
    research_question: str
    pico: Dict[str, str]
    
    # Process summary
    total_articles_screened: int
    total_articles_included: int
    processing_time_minutes: float
    quality_score: float
    
    # Results
    final_report: str
    statistical_analysis: Dict[str, Any]
    forest_plots: List[str]  # File paths or base64 encoded images
    citations: List[str]
    
    # Quality metrics
    heterogeneity_i2: float
    overall_effect_size: float
    confidence_interval: List[float]
    p_value: float
    
    # Metadata
    generated_at: datetime
    agent_performance: Dict[str, Dict[str, Any]]
    execution_log: List[Dict[str, Any]]
    
    class Config:
        arbitrary_types_allowed = True


def create_initial_state(
    research_question: str,
    meta_analysis_id: str,
    thread_id: str,
    config: Dict[str, Any] = None
) -> MetaAnalysisState:
    """
    Create initial state for a new meta-analysis
    
    Args:
        research_question: The research question to investigate
        meta_analysis_id: Unique identifier for this meta-analysis
        thread_id: Thread identifier for conversation persistence
        config: Optional configuration overrides
        
    Returns:
        Initialized MetaAnalysisState
    """
    
    now = datetime.now()
    default_config = config or {}
    
    return MetaAnalysisState(
        # Core identification
        meta_analysis_id=meta_analysis_id,
        thread_id=thread_id,
        current_phase="pico_definition",
        current_agent=None,
        last_agent=None,
        
        # Research definition
        pico={},
        research_question=research_question,
        inclusion_criteria=[],
        exclusion_criteria=[],
        search_strategy={},
        
        # Search and processing
        search_queries=[],
        search_domains=[],
        candidate_urls=[],
        processing_queue=[],
        processed_articles=[],
        failed_urls=[],
        
        # Extraction
        extracted_data=[],
        vancouver_citations=[],
        article_summaries=[],
        
        # Vector store
        vector_store_id=None,
        vector_store_status={},
        chunk_count=0,
        embedding_model_used="text-embedding-3-small",
        
        # Retrieval
        retrieval_queries=[],
        retrieval_results=[],
        relevant_chunks=[],
        
        # Analysis
        statistical_analysis={},
        forest_plots=[],
        funnel_plots=[],
        quality_assessments={},
        heterogeneity_analysis={},
        sensitivity_analysis={},
        
        # Reports
        draft_report=None,
        report_sections={},
        review_feedback=[],
        final_report=None,
        report_metadata={},
        
        # Quality control
        quality_scores={},
        quality_thresholds=default_config.get("quality_thresholds", {}),
        improvement_rates={},
        
        # Iteration control
        agent_iterations={},
        agent_limits=default_config.get("agent_limits", {}),
        global_iterations=0,
        max_global_iterations=default_config.get("max_global_iterations", 10),
        
        # Circuit breaker
        circuit_status={},
        failure_counts={},
        retry_counts={},
        last_failure_time={},
        
        # Communication
        messages=[],
        agent_logs=[],
        handoff_history=[],
        
        # Checkpoints
        checkpoints=[],
        last_checkpoint_time=now.timestamp(),
        checkpoint_frequency=5,
        
        # Flags
        force_stop=False,
        quality_satisfied=False,
        deadline_reached=False,
        max_retries_reached=False,
        emergency_stop=False,
        
        # Metadata
        created_at=now,
        updated_at=now,
        total_articles_found=0,
        total_articles_processed=0,
        total_articles_included=0,
        processing_success_rate=0.0,
        execution_time={},
        
        # Configuration
        config=default_config
    )