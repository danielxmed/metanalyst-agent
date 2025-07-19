"""
Estado compartilhado para o sistema de meta-análise multi-agente.
Implementa memória de curto e longo prazo usando LangGraph.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import uuid


class MetaAnalysisState(TypedDict):
    """
    Estado compartilhado para o sistema de meta-análise.
    
    Este estado é mantido pelo orquestrador central e compartilhado
    entre todos os agentes especializados na arquitetura hub-and-spoke.
    """
    
    # === IDENTIFICAÇÃO E CONTROLE ===
    meta_analysis_id: str
    thread_id: str
    current_phase: Literal[
        "pico_definition", 
        "search", 
        "extraction", 
        "vectorization", 
        "analysis", 
        "writing", 
        "review", 
        "editing",
        "completed"
    ]
    current_agent: Optional[str]
    
    # === PICO E PESQUISA ===
    pico: Dict[str, str]  # {P: Population, I: Intervention, C: Comparison, O: Outcome}
    search_queries: List[str]
    search_domains: List[str]
    user_request: str  # Solicitação original do usuário
    
    # === URLs E PROCESSAMENTO ===
    candidate_urls: List[Dict[str, Any]]  # [{url, title, relevance_score, domain}]
    processing_queue: List[str]  # URLs aguardando processamento
    processed_articles: List[Dict[str, Any]]  # Artigos processados com dados extraídos
    failed_urls: List[Dict[str, str]]  # [{url, error, timestamp}]
    
    # === EXTRAÇÃO E VETORIZAÇÃO ===
    extracted_data: List[Dict[str, Any]]  # Dados estruturados extraídos
    vector_store_id: Optional[str]  # ID do vector store no PostgreSQL Store
    vector_store_status: Dict[str, Any]  # Status da vetorização
    chunk_count: int  # Número total de chunks criados
    
    # === ANÁLISE E RESULTADOS ===
    retrieval_results: List[Dict[str, Any]]  # Resultados da busca semântica
    statistical_analysis: Dict[str, Any]  # Análises estatísticas realizadas
    forest_plots: List[Dict[str, Any]]  # Dados para forest plots
    quality_assessments: Dict[str, float]  # Avaliações de qualidade dos estudos
    
    # === RELATÓRIOS E REVISÕES ===
    draft_report: Optional[str]  # Rascunho do relatório em HTML
    review_feedback: List[Dict[str, str]]  # Feedback do revisor
    final_report: Optional[str]  # Relatório final em HTML
    citations: List[Dict[str, str]]  # Citações em formato Vancouver
    
    # === MENSAGENS E LOGS ===
    messages: Annotated[List[BaseMessage], add_messages]  # Histórico de mensagens
    agent_logs: List[Dict[str, Any]]  # Logs detalhados por agente
    
    # === METADADOS ===
    created_at: datetime
    updated_at: datetime
    total_articles_processed: int
    execution_time: Dict[str, float]  # Tempo de execução por fase
    
    # === CONFIGURAÇÕES ===
    config: Dict[str, Any]  # Parâmetros configuráveis do sistema


def create_initial_state(user_request: str, config: Dict[str, Any] = None) -> MetaAnalysisState:
    """
    Cria o estado inicial para uma nova meta-análise.
    
    Args:
        user_request: Solicitação do usuário em linguagem natural
        config: Configurações opcionais do sistema
        
    Returns:
        Estado inicial da meta-análise
    """
    now = datetime.now()
    analysis_id = str(uuid.uuid4())
    thread_id = f"metanalysis_{analysis_id[:8]}"
    
    return MetaAnalysisState(
        # Identificação
        meta_analysis_id=analysis_id,
        thread_id=thread_id,
        current_phase="pico_definition",
        current_agent=None,
        
        # PICO e pesquisa
        pico={},
        search_queries=[],
        search_domains=[],
        user_request=user_request,
        
        # URLs e processamento
        candidate_urls=[],
        processing_queue=[],
        processed_articles=[],
        failed_urls=[],
        
        # Extração e vetorização
        extracted_data=[],
        vector_store_id=None,
        vector_store_status={},
        chunk_count=0,
        
        # Análise e resultados
        retrieval_results=[],
        statistical_analysis={},
        forest_plots=[],
        quality_assessments={},
        
        # Relatórios e revisões
        draft_report=None,
        review_feedback=[],
        final_report=None,
        citations=[],
        
        # Mensagens e logs
        messages=[],
        agent_logs=[],
        
        # Metadados
        created_at=now,
        updated_at=now,
        total_articles_processed=0,
        execution_time={},
        
        # Configurações
        config=config or {}
    )


def update_state_phase(
    state: MetaAnalysisState, 
    new_phase: str, 
    agent_name: str = None
) -> Dict[str, Any]:
    """
    Atualiza a fase atual do estado e registra a mudança.
    
    Args:
        state: Estado atual
        new_phase: Nova fase
        agent_name: Nome do agente responsável pela mudança
        
    Returns:
        Dicionário com atualizações do estado
    """
    return {
        "current_phase": new_phase,
        "current_agent": agent_name,
        "updated_at": datetime.now(),
        "agent_logs": state.get("agent_logs", []) + [{
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name or "system",
            "action": "phase_change",
            "from_phase": state.get("current_phase"),
            "to_phase": new_phase
        }]
    }


def add_agent_log(
    state: MetaAnalysisState,
    agent_name: str,
    action: str,
    details: Dict[str, Any] = None,
    status: str = "success"
) -> Dict[str, Any]:
    """
    Adiciona log de ação de agente ao estado.
    
    Args:
        state: Estado atual
        agent_name: Nome do agente
        action: Ação realizada
        details: Detalhes adicionais
        status: Status da ação (success, error, warning)
        
    Returns:
        Dicionário com atualização dos logs
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "action": action,
        "status": status,
        "details": details or {}
    }
    
    return {
        "agent_logs": state.get("agent_logs", []) + [log_entry],
        "updated_at": datetime.now()
    }


def get_state_summary(state: MetaAnalysisState) -> Dict[str, Any]:
    """
    Gera resumo do estado atual para debug e monitoramento.
    
    Args:
        state: Estado atual
        
    Returns:
        Resumo do estado
    """
    return {
        "meta_analysis_id": state["meta_analysis_id"],
        "current_phase": state["current_phase"],
        "current_agent": state["current_agent"],
        "urls_found": len(state["candidate_urls"]),
        "urls_processed": len(state["processed_articles"]),
        "urls_failed": len(state["failed_urls"]),
        "chunks_created": state["chunk_count"],
        "has_vector_store": bool(state["vector_store_id"]),
        "has_analysis": bool(state["statistical_analysis"]),
        "has_report": bool(state["final_report"]),
        "total_messages": len(state["messages"]),
        "execution_time": state["execution_time"],
        "created_at": state["created_at"].isoformat(),
        "updated_at": state["updated_at"].isoformat()
    }