"""
Modelos de dados para o sistema metanalyst-agent.
"""

from .state import (
    MetaAnalysisState,
    create_initial_state,
    update_state_phase,
    add_agent_log,
    get_state_summary
)

from .schemas import (
    PICO,
    StudyType,
    OutcomeType,
    StudyCharacteristics,
    OutcomeData,
    QualityAssessment,
    ExtractedStudy,
    StatisticalAnalysis,
    Citation,
    MetaAnalysisReport,
    VectorChunk,
    SearchResult
)

__all__ = [
    # State
    "MetaAnalysisState",
    "create_initial_state", 
    "update_state_phase",
    "add_agent_log",
    "get_state_summary",
    
    # Schemas
    "PICO",
    "StudyType",
    "OutcomeType", 
    "StudyCharacteristics",
    "OutcomeData",
    "QualityAssessment",
    "ExtractedStudy",
    "StatisticalAnalysis",
    "Citation",
    "MetaAnalysisReport",
    "VectorChunk",
    "SearchResult"
]