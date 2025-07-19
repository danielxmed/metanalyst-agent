"""Tools module for Metanalyst-Agent specialized agents"""

from .research_tools import (
    search_literature,
    generate_search_queries,
    assess_article_relevance
)

from .processor_tools import (
    extract_article_content,
    extract_statistical_data,
    generate_vancouver_citation,
    chunk_and_vectorize,
    process_article_metadata
)

from .retrieval_tools import (
    search_vector_store,
    retrieve_relevant_chunks,
    search_for_statistical_data
)

from .analysis_tools import (
    calculate_meta_analysis,
    create_forest_plot,
    assess_heterogeneity,
    perform_sensitivity_analysis,
    generate_funnel_plot
)

from .writing_tools import (
    generate_report_section_with_llm,
    format_citations_with_llm,
    create_complete_report
)

from .handoff_tools import (
    create_handoff_tool,
    request_supervisor_intervention,
    signal_completion,
    request_quality_check
)

__all__ = [
    # Research tools
    "search_literature",
    "generate_search_queries",
    "assess_article_relevance",
    
    # Processor tools
    "extract_article_content",
    "extract_statistical_data",
    "generate_vancouver_citation",
    "chunk_and_vectorize",
    "process_article_metadata",
    
    # Retrieval tools
    "search_vector_store",
    "retrieve_relevant_chunks",
    "search_for_statistical_data",
    
    # Analysis tools
    "calculate_meta_analysis",
    "create_forest_plot",
    "assess_heterogeneity",
    "perform_sensitivity_analysis",
    "create_funnel_plot",
    
    # Writing tools
    "generate_report_section_with_llm",
    "format_citations_with_llm",
    "create_complete_report",
    
    # Handoff tools
    "create_handoff_tool",
    "request_supervisor_intervention",
    "signal_completion",
    "request_quality_check"
]