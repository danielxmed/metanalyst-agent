"""
Data schemas for the Metanalyst Agent system.

This module defines the structured data types used throughout the workflow,
including PICO, extracted papers, analysis results, and other domain-specific schemas.
"""

from typing import TypedDict, List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum


class PICO(TypedDict):
    """
    PICO structure for defining research questions.
    
    PICO (Patient/Population, Intervention, Comparison, Outcome) is the
    standard framework for formulating clinical research questions.
    """
    patient: str        # Patient/Population description
    intervention: str   # Intervention being studied
    comparison: str     # Comparison/Control group
    outcome: str        # Outcome being measured


class PaperMetadata(TypedDict):
    """Metadata for a scientific paper."""
    authors: List[str]
    year: Optional[int]
    journal: Optional[str]
    doi: Optional[str]
    pmid: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    publication_type: Optional[str]
    impact_factor: Optional[float]


class StatisticalData(TypedDict):
    """Statistical data extracted from papers."""
    sample_size: Optional[int]
    control_group_size: Optional[int]
    intervention_group_size: Optional[int]
    relative_risk: Optional[float]
    odds_ratio: Optional[float]
    hazard_ratio: Optional[float]
    confidence_interval: Optional[List[float]]  # [lower, upper]
    p_value: Optional[float]
    effect_size: Optional[float]
    standard_error: Optional[float]
    variance: Optional[float]
    mean_difference: Optional[float]
    standardized_mean_difference: Optional[float]
    chi_square: Optional[float]
    degrees_of_freedom: Optional[int]
    heterogeneity_i2: Optional[float]
    heterogeneity_p: Optional[float]


class ExtractedPaper(TypedDict):
    """
    Structure for papers extracted from URLs.
    
    Contains the full processed information from a scientific paper,
    including content, metadata, and extracted statistical data.
    """
    paper_id: str                           # Unique identifier
    title: str                              # Paper title
    abstract: str                           # Abstract text
    content: str                            # Full processed content
    summary: str                            # Objective summary
    url: str                                # Original URL
    reference: str                          # Vancouver-style reference
    metadata: PaperMetadata                 # Paper metadata
    statistics: StatisticalData             # Extracted statistical data
    quality_score: Optional[float]          # Quality assessment score
    inclusion_criteria_met: bool            # Whether paper meets inclusion criteria
    extraction_timestamp: str               # When data was extracted
    extractor_version: str                  # Version of extraction logic used


class SearchResult(TypedDict):
    """Result from literature search."""
    query: str
    url: str
    title: str
    snippet: str
    score: Optional[float]
    domain: str
    search_timestamp: str


class VectorChunk(TypedDict):
    """Chunk of text with vector embedding and metadata."""
    chunk_id: str
    text: str
    paper_id: str
    paper_reference: str
    chunk_index: int
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]


class RetrievalResult(TypedDict):
    """Result from vector similarity search."""
    chunk: VectorChunk
    similarity_score: float
    rank: int


class ReviewFeedback(TypedDict):
    """Feedback from the reviewer agent."""
    overall_quality: float                  # 0-10 scale
    needs_more_research: bool               # Whether additional searches are needed
    missing_aspects: List[str]              # Missing aspects in the report
    suggested_improvements: List[str]       # Specific improvement suggestions
    approval_status: str                    # "approved", "needs_revision", "rejected"
    reviewer_comments: str                  # Detailed comments
    statistical_concerns: List[str]         # Concerns about statistical analysis
    methodology_concerns: List[str]         # Concerns about methodology


class MetaAnalysisResult(TypedDict):
    """Results from statistical meta-analysis."""
    pooled_effect_size: float
    pooled_confidence_interval: List[float]
    pooled_p_value: float
    heterogeneity_i2: float
    heterogeneity_p_value: float
    tau_squared: Optional[float]            # Between-study variance
    prediction_interval: Optional[List[float]]
    studies_included: int
    total_sample_size: int
    effect_measure: str                     # "RR", "OR", "MD", "SMD", etc.
    analysis_method: str                    # "fixed", "random"
    subgroup_analyses: List[Dict[str, Any]]
    sensitivity_analyses: List[Dict[str, Any]]
    publication_bias_tests: Dict[str, Any]


class Visualization(TypedDict):
    """Visualization data and metadata."""
    type: str                               # "forest_plot", "funnel_plot", "table", etc.
    file_path: str                          # Path to generated file
    title: str
    description: str
    data_source: List[str]                  # Source paper IDs
    creation_timestamp: str


class QualityAssessment(TypedDict):
    """Quality assessment for individual studies."""
    paper_id: str
    assessment_tool: str                    # "Cochrane", "Newcastle-Ottawa", etc.
    risk_of_bias: Dict[str, str]           # Domain -> "low", "high", "unclear"
    overall_quality: str                   # "high", "moderate", "low"
    quality_score: Optional[float]         # Numerical score if applicable
    assessor_notes: str


class ReportSection(TypedDict):
    """Individual section of the meta-analysis report."""
    section_id: str
    title: str
    content: str
    subsections: List['ReportSection']
    references: List[str]                   # Paper IDs referenced
    tables: List[str]                       # Table IDs
    figures: List[str]                      # Figure IDs


class FinalReport(TypedDict):
    """Complete meta-analysis report structure."""
    title: str
    abstract: str
    sections: List[ReportSection]
    references: List[ExtractedPaper]
    tables: List[Dict[str, Any]]
    figures: List[Visualization]
    appendices: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    generation_timestamp: str
    word_count: int
    page_count: Optional[int]


# Enums for standardized values
class StudyType(Enum):
    """Types of studies that can be included."""
    RCT = "randomized_controlled_trial"
    COHORT = "cohort_study"
    CASE_CONTROL = "case_control_study"
    CROSS_SECTIONAL = "cross_sectional_study"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"


class EffectMeasure(Enum):
    """Types of effect measures for meta-analysis."""
    RELATIVE_RISK = "RR"
    ODDS_RATIO = "OR"
    HAZARD_RATIO = "HR"
    MEAN_DIFFERENCE = "MD"
    STANDARDIZED_MEAN_DIFFERENCE = "SMD"
    RISK_DIFFERENCE = "RD"
    INCIDENCE_RATE_RATIO = "IRR"


class AnalysisMethod(Enum):
    """Statistical methods for meta-analysis."""
    FIXED_EFFECTS = "fixed"
    RANDOM_EFFECTS = "random"
    MIXED_EFFECTS = "mixed"


# Helper functions for schema validation and creation
def create_pico(patient: str, intervention: str, comparison: str, outcome: str) -> PICO:
    """Create a validated PICO structure."""
    return PICO(
        patient=patient.strip(),
        intervention=intervention.strip(),
        comparison=comparison.strip(),
        outcome=outcome.strip()
    )


def create_empty_statistical_data() -> StatisticalData:
    """Create an empty statistical data structure with None values."""
    return StatisticalData(
        sample_size=None,
        control_group_size=None,
        intervention_group_size=None,
        relative_risk=None,
        odds_ratio=None,
        hazard_ratio=None,
        confidence_interval=None,
        p_value=None,
        effect_size=None,
        standard_error=None,
        variance=None,
        mean_difference=None,
        standardized_mean_difference=None,
        chi_square=None,
        degrees_of_freedom=None,
        heterogeneity_i2=None,
        heterogeneity_p=None
    )


def create_paper_template(url: str, title: str = "") -> ExtractedPaper:
    """Create a template ExtractedPaper with default values."""
    from uuid import uuid4
    
    return ExtractedPaper(
        paper_id=str(uuid4()),
        title=title,
        abstract="",
        content="",
        summary="",
        url=url,
        reference="",
        metadata=PaperMetadata(
            authors=[],
            year=None,
            journal=None,
            doi=None,
            pmid=None,
            volume=None,
            issue=None,
            pages=None,
            publication_type=None,
            impact_factor=None
        ),
        statistics=create_empty_statistical_data(),
        quality_score=None,
        inclusion_criteria_met=False,
        extraction_timestamp=datetime.now().isoformat(),
        extractor_version="1.0.0"
    )


def validate_pico(pico: Dict[str, str]) -> bool:
    """Validate that a PICO structure has all required fields."""
    required_fields = ["patient", "intervention", "comparison", "outcome"]
    return all(
        field in pico and isinstance(pico[field], str) and len(pico[field].strip()) > 0
        for field in required_fields
    )
