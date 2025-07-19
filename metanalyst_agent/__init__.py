"""
MetAnalyst Agent - Sistema Multi-Agente Autônomo para Meta-Análises Médicas

Este módulo implementa um sistema de agentes autônomos usando LangGraph
para automatizar o processo completo de meta-análise médica seguindo diretrizes PRISMA.

Arquitetura Hub-and-Spoke com agentes especializados:
- Supervisor: Coordenador central
- Researcher: Busca de literatura científica
- Processor: Extração e processamento de artigos
- Vectorizer: Vetorização para busca semântica
- Retriever: Recuperação inteligente de informações
- Analyst: Análise estatística e visualizações
- Writer: Geração de relatórios PRISMA
- Reviewer: Controle de qualidade
- Editor: Formatação final
"""

from .models.state import MetaAnalysisState
from .models.config import config
from .graph.multi_agent_graph import (
    run_meta_analysis,
    build_meta_analysis_graph,
    create_default_graph,
    create_memory_graph,
    visualize_graph
)

__version__ = "1.0.0"
__author__ = "Nobrega Medtech"
__description__ = "Sistema multi-agente autônomo para meta-análises médicas"

__all__ = [
    "MetaAnalysisState",
    "config",
    "run_meta_analysis",
    "build_meta_analysis_graph",
    "create_default_graph",
    "create_memory_graph",
    "visualize_graph"
]