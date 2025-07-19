"""
Schemas Pydantic para dados estruturados do sistema de meta-análise.
Define modelos para extração de dados, análises estatísticas e relatórios.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class StudyType(str, Enum):
    """Tipos de estudos científicos."""
    RCT = "randomized_controlled_trial"
    COHORT = "cohort_study"
    CASE_CONTROL = "case_control_study"
    CROSS_SECTIONAL = "cross_sectional_study"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    CASE_REPORT = "case_report"
    OTHER = "other"


class OutcomeType(str, Enum):
    """Tipos de desfechos."""
    BINARY = "binary"
    CONTINUOUS = "continuous"
    TIME_TO_EVENT = "time_to_event"
    ORDINAL = "ordinal"


class PICO(BaseModel):
    """Estrutura PICO para definição da questão de pesquisa."""
    
    population: str = Field(..., description="População estudada")
    intervention: str = Field(..., description="Intervenção")
    comparison: str = Field(..., description="Comparação ou controle")
    outcome: str = Field(..., description="Desfecho de interesse")
    
    class Config:
        json_schema_extra = {
            "example": {
                "population": "Pacientes com diabetes tipo 2",
                "intervention": "Metformina",
                "comparison": "Placebo",
                "outcome": "Controle glicêmico (HbA1c)"
            }
        }


class StudyCharacteristics(BaseModel):
    """Características extraídas de um estudo."""
    
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str] = None
    pmid: Optional[str] = None
    
    study_type: StudyType
    sample_size: int
    population_description: str
    
    intervention_group: Dict[str, Any]
    control_group: Dict[str, Any]
    
    primary_outcome: str
    secondary_outcomes: List[str] = []
    
    follow_up_duration: Optional[str] = None
    
    @validator('year')
    def validate_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year:
            raise ValueError(f'Ano deve estar entre 1900 e {current_year}')
        return v


class OutcomeData(BaseModel):
    """Dados de desfecho extraídos de um estudo."""
    
    outcome_name: str
    outcome_type: OutcomeType
    
    # Para desfechos binários
    intervention_events: Optional[int] = None
    intervention_total: Optional[int] = None
    control_events: Optional[int] = None
    control_total: Optional[int] = None
    
    # Para desfechos contínuos
    intervention_mean: Optional[float] = None
    intervention_sd: Optional[float] = None
    intervention_n: Optional[int] = None
    control_mean: Optional[float] = None
    control_sd: Optional[float] = None
    control_n: Optional[int] = None
    
    # Medidas de efeito reportadas
    effect_measure: Optional[str] = None  # OR, RR, MD, SMD, etc.
    effect_size: Optional[float] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    p_value: Optional[float] = None
    
    units: Optional[str] = None
    notes: Optional[str] = None


class QualityAssessment(BaseModel):
    """Avaliação de qualidade de um estudo."""
    
    study_id: str
    assessment_tool: str = "custom"  # ROB2, Newcastle-Ottawa, etc.
    
    # Domínios de risco de viés
    randomization: Optional[str] = None  # low, high, unclear
    allocation_concealment: Optional[str] = None
    blinding_participants: Optional[str] = None
    blinding_outcome: Optional[str] = None
    incomplete_data: Optional[str] = None
    selective_reporting: Optional[str] = None
    other_bias: Optional[str] = None
    
    overall_quality: str = Field(..., description="low, moderate, high")
    quality_score: float = Field(..., ge=0, le=10)
    
    notes: Optional[str] = None


class ExtractedStudy(BaseModel):
    """Dados completos extraídos de um estudo."""
    
    url: str
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    
    characteristics: StudyCharacteristics
    outcomes: List[OutcomeData]
    quality_assessment: QualityAssessment
    
    # Dados brutos para referência
    full_text_available: bool
    abstract: str
    key_findings: List[str]
    limitations: List[str]
    
    # Metadados de processamento
    processing_notes: List[str] = []
    confidence_score: float = Field(ge=0, le=1, description="Confiança na extração")


class StatisticalAnalysis(BaseModel):
    """Resultados de análise estatística."""
    
    outcome_name: str
    analysis_type: str  # fixed_effect, random_effect
    
    # Resultados da meta-análise
    pooled_effect: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: float
    
    # Heterogeneidade
    i_squared: float = Field(ge=0, le=100)
    tau_squared: Optional[float] = None
    q_statistic: float
    q_p_value: float
    
    # Número de estudos
    number_of_studies: int
    total_participants: int
    
    # Forest plot data
    forest_plot_data: List[Dict[str, Any]]
    
    # Análises de sensibilidade
    sensitivity_analyses: List[Dict[str, Any]] = []
    
    # Avaliação de viés de publicação
    publication_bias: Optional[Dict[str, Any]] = None


class Citation(BaseModel):
    """Citação em formato Vancouver."""
    
    study_id: str
    authors: List[str]
    title: str
    journal: str
    year: int
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    
    def to_vancouver(self) -> str:
        """Converte para formato Vancouver."""
        # Formatar autores
        if len(self.authors) <= 6:
            author_str = ", ".join(self.authors)
        else:
            author_str = ", ".join(self.authors[:6]) + ", et al"
        
        # Construir citação
        citation = f"{author_str}. {self.title}. {self.journal}. {self.year}"
        
        if self.volume:
            citation += f";{self.volume}"
            if self.issue:
                citation += f"({self.issue})"
        
        if self.pages:
            citation += f":{self.pages}"
        
        citation += "."
        
        if self.doi:
            citation += f" doi:{self.doi}"
        
        return citation


class MetaAnalysisReport(BaseModel):
    """Relatório completo de meta-análise."""
    
    title: str
    pico: PICO
    
    # Metodologia
    search_strategy: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    databases_searched: List[str]
    
    # Resultados
    studies_found: int
    studies_included: int
    studies_excluded: int
    exclusion_reasons: Dict[str, int]
    
    # Características dos estudos
    study_characteristics_summary: Dict[str, Any]
    
    # Análises estatísticas
    statistical_analyses: List[StatisticalAnalysis]
    
    # Qualidade da evidência
    overall_quality: str
    grade_assessment: Optional[Dict[str, str]] = None
    
    # Conclusões
    main_findings: List[str]
    clinical_implications: List[str]
    limitations: List[str]
    recommendations: List[str]
    
    # Metadados
    generated_at: datetime = Field(default_factory=datetime.now)
    citations: List[Citation]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VectorChunk(BaseModel):
    """Chunk de texto vetorizado."""
    
    chunk_id: str
    study_id: str
    content: str
    embedding: List[float]
    
    # Metadados do chunk
    section: str  # abstract, methods, results, discussion
    start_char: int
    end_char: int
    
    # Referência para busca
    study_title: str
    study_authors: List[str]
    study_year: int
    
    created_at: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """Resultado de busca de literatura."""
    
    url: str
    title: str
    authors: List[str] = []
    abstract: str = ""
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    
    # Scores de relevância
    relevance_score: float = Field(ge=0, le=1)
    pico_match_score: float = Field(ge=0, le=1)
    
    # Metadados da busca
    search_query: str
    search_domain: str
    found_at: datetime = Field(default_factory=datetime.now)
    
    @validator('relevance_score', 'pico_match_score')
    def validate_scores(cls, v):
        return round(v, 3)