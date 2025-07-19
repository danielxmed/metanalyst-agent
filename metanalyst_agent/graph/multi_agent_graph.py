"""
Grafo multi-agente para sistema de meta-análise automatizada.
Implementa arquitetura hub-and-spoke com agentes autônomos.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from ..config.database import get_checkpointer, get_store
from ..agents import (
    supervisor_agent,
    researcher_agent,
    processor_agent,
    retriever_agent,
    analyst_agent,
    writer_agent,
    reviewer_agent,
    editor_agent
)


class MetaAnalysisState(MessagesState):
    """Estado compartilhado entre todos os agentes do sistema."""
    
    # Identificação e controle
    meta_analysis_id: Optional[str] = None
    current_phase: Optional[str] = None
    last_agent: Optional[str] = None
    
    # Critérios PICO
    pico_criteria: Dict[str, str] = {}
    
    # Dados de pesquisa
    search_queries: List[str] = []
    candidate_articles: List[Dict[str, Any]] = []
    relevant_urls: List[str] = []
    
    # Processamento
    processed_articles: List[Dict[str, Any]] = []
    statistical_data: List[Dict[str, Any]] = []
    vector_store_status: Dict[str, Any] = {}
    
    # Análise
    meta_analysis_results: Dict[str, Any] = {}
    forest_plot_path: Optional[str] = None
    funnel_plot_path: Optional[str] = None
    
    # Relatório
    report_sections: Dict[str, str] = {}
    quality_assessment: Dict[str, Any] = {}
    final_report_path: Optional[str] = None
    
    # Metadados
    handoff_reason: Optional[str] = None
    handoff_context: Optional[str] = None
    error_flag: bool = False
    meta_analysis_complete: bool = False


def build_meta_analysis_graph() -> StateGraph:
    """
    Constrói o grafo multi-agente para meta-análise.
    
    Returns:
        Grafo compilado com persistência PostgreSQL
    """
    
    # Criar builder do grafo
    builder = StateGraph(MetaAnalysisState)
    
    # Adicionar supervisor como nó central
    builder.add_node("supervisor", supervisor_agent)
    
    # Adicionar todos os agentes especializados
    builder.add_node("researcher", researcher_agent)
    builder.add_node("processor", processor_agent)
    builder.add_node("retriever", retriever_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("reviewer", reviewer_agent)
    builder.add_node("editor", editor_agent)
    
    # Configurar fluxo: sempre começar pelo supervisor
    builder.add_edge(START, "supervisor")
    
    # Todos os agentes especializados retornam ao supervisor após execução
    # Isso implementa a arquitetura hub-and-spoke
    for agent_name in ["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor"]:
        builder.add_edge(agent_name, "supervisor")
    
    # Obter checkpointer e store
    checkpointer = get_checkpointer()
    store = get_store()
    
    # Compilar grafo com persistência
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    return graph


def create_meta_analysis_config(thread_id: str) -> Dict[str, Any]:
    """
    Cria configuração para execução de meta-análise.
    
    Args:
        thread_id: ID único para a thread de execução
    
    Returns:
        Configuração para o grafo
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "metanalysis",
        }
    }