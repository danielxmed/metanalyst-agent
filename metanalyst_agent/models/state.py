"""
Estado compartilhado para o sistema multi-agente de meta-análise.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from datetime import datetime
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Tipos auxiliares
AgentPhase = Literal[
    "initialization",
    "pico_definition", 
    "literature_search",
    "content_extraction", 
    "data_processing",
    "vectorization",
    "information_retrieval",
    "statistical_analysis",
    "report_generation",
    "quality_review",
    "final_editing",
    "completed"
]

AgentName = Literal[
    "supervisor",
    "researcher", 
    "processor",
    "vectorizer",
    "retriever",
    "analyst",
    "writer", 
    "reviewer",
    "editor"
]

class ArticleData(TypedDict):
    """Estrutura para dados de artigos processados"""
    id: str
    url: str
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: Optional[str]
    abstract: str
    full_text: Optional[str]
    statistical_data: Dict[str, Any]
    quality_score: float
    vancouver_citation: str
    extraction_timestamp: str

class StatisticalAnalysis(TypedDict):
    """Resultados de análise estatística"""
    pooled_effect_size: float
    confidence_interval: List[float]  # [lower, upper]
    p_value: float
    heterogeneity_i2: float
    heterogeneity_p: float
    tau_squared: float
    total_participants: int
    number_of_studies: int
    forest_plot_path: Optional[str]
    funnel_plot_path: Optional[str]
    sensitivity_analysis: Dict[str, Any]

class AgentTransition(TypedDict):
    """Registro de transições entre agentes"""
    from_agent: str
    to_agent: str
    reason: str
    context: str
    timestamp: str
    success: bool

class MetaAnalysisState(TypedDict):
    """Estado compartilhado completo para o sistema de meta-análise"""
    
    # === IDENTIFICAÇÃO E CONTROLE ===
    meta_analysis_id: str
    thread_id: str
    current_phase: AgentPhase
    current_agent: Optional[AgentName]
    last_agent: Optional[AgentName]
    
    # === PICO E DEFINIÇÃO DA PESQUISA ===
    pico: Dict[str, str]  # Population, Intervention, Comparison, Outcome
    research_question: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    search_strategy: Dict[str, Any]
    
    # === BUSCA E COLETA ===
    search_queries: List[str]
    search_domains: List[str]  # Domínios médicos específicos
    candidate_urls: List[Dict[str, Any]]  # URLs encontradas
    total_articles_found: int
    
    # === PROCESSAMENTO DE ARTIGOS ===
    processing_queue: List[str]  # URLs a processar
    processed_articles: List[ArticleData]
    failed_extractions: List[Dict[str, str]]  # {url, error}
    extraction_progress: Dict[str, int]  # {total, processed, failed}
    
    # === VETORIZAÇÃO E ARMAZENAMENTO ===
    vector_store_id: Optional[str]
    vector_store_status: Dict[str, Any]
    total_chunks_created: int
    embedding_model_used: str
    vectorization_complete: bool
    
    # === RECUPERAÇÃO DE INFORMAÇÕES ===
    retrieval_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    relevant_information: Dict[str, List[str]]  # Por categoria
    
    # === ANÁLISE ESTATÍSTICA ===
    extracted_statistics: List[Dict[str, Any]]
    statistical_analysis: Optional[StatisticalAnalysis]
    quality_assessment: Dict[str, float]  # Pontuações de qualidade
    bias_assessment: Dict[str, Any]
    
    # === GERAÇÃO DE RELATÓRIOS ===
    draft_report: Optional[str]
    report_sections: Dict[str, str]  # {section_name: content}
    citations: List[str]  # Citações Vancouver
    figures_generated: List[str]  # Caminhos dos gráficos
    
    # === REVISÃO E QUALIDADE ===
    review_feedback: List[Dict[str, str]]  # {section, feedback}
    quality_checks: Dict[str, bool]
    revision_history: List[Dict[str, Any]]
    
    # === DOCUMENTO FINAL ===
    final_report: Optional[str]
    final_report_path: Optional[str]
    executive_summary: Optional[str]
    
    # === COMUNICAÇÃO ENTRE AGENTES ===
    messages: Annotated[List[BaseMessage], add_messages]
    agent_transitions: List[AgentTransition]
    agent_logs: List[Dict[str, Any]]  # Logs detalhados
    
    # === METADADOS E CONTROLE ===
    created_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp
    execution_time_by_phase: Dict[str, float]  # Segundos por fase
    total_execution_time: float
    
    # === CONFIGURAÇÕES ===
    system_config: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    # === FLAGS DE ESTADO ===
    error_flag: bool
    error_details: Optional[Dict[str, Any]]
    requires_human_intervention: bool
    completion_percentage: float
    
    # === DADOS TEMPORÁRIOS ===
    temp_data: Dict[str, Any]  # Para dados temporários entre agentes
    cache: Dict[str, Any]  # Cache de operações custosas

def create_initial_state(
    research_question: str,
    thread_id: str,
    user_config: Optional[Dict[str, Any]] = None
) -> MetaAnalysisState:
    """Criar estado inicial para uma nova meta-análise"""
    
    import uuid
    from datetime import datetime
    
    meta_analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    return MetaAnalysisState(
        # Identificação
        meta_analysis_id=meta_analysis_id,
        thread_id=thread_id,
        current_phase="initialization",
        current_agent=None,
        last_agent=None,
        
        # Pesquisa
        pico={},
        research_question=research_question,
        inclusion_criteria=[],
        exclusion_criteria=[],
        search_strategy={},
        
        # Busca
        search_queries=[],
        search_domains=[
            "nejm.org",           # New England Journal of Medicine
            "jamanetwork.com",    # JAMA
            "thelancet.com",      # The Lancet
            "bmj.com",            # BMJ
            "pubmed.ncbi.nlm.nih.gov",  # PubMed
            "ncbi.nlm.nih.gov/pmc",     # PMC
            "scielo.org",         # SciELO
            "cochranelibrary.com" # Cochrane Library
        ],
        candidate_urls=[],
        total_articles_found=0,
        
        # Processamento
        processing_queue=[],
        processed_articles=[],
        failed_extractions=[],
        extraction_progress={"total": 0, "processed": 0, "failed": 0},
        
        # Vetorização
        vector_store_id=None,
        vector_store_status={},
        total_chunks_created=0,
        embedding_model_used="text-embedding-3-small",
        vectorization_complete=False,
        
        # Recuperação
        retrieval_queries=[],
        retrieved_chunks=[],
        relevant_information={},
        
        # Análise
        extracted_statistics=[],
        statistical_analysis=None,
        quality_assessment={},
        bias_assessment={},
        
        # Relatórios
        draft_report=None,
        report_sections={},
        citations=[],
        figures_generated=[],
        
        # Revisão
        review_feedback=[],
        quality_checks={},
        revision_history=[],
        
        # Final
        final_report=None,
        final_report_path=None,
        executive_summary=None,
        
        # Comunicação
        messages=[],
        agent_transitions=[],
        agent_logs=[],
        
        # Metadados
        created_at=timestamp,
        updated_at=timestamp,
        execution_time_by_phase={},
        total_execution_time=0.0,
        
        # Config
        system_config=user_config or {},
        user_preferences={},
        
        # Flags
        error_flag=False,
        error_details=None,
        requires_human_intervention=False,
        completion_percentage=0.0,
        
        # Temporários
        temp_data={},
        cache={}
    )