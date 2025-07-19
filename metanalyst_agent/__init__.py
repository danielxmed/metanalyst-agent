"""
MetAnalyst Agent - Sistema Multi-Agente Autônomo para Meta-Análises Médicas

Este módulo implementa um sistema de agentes autônomos usando LangGraph
para automatizar o processo completo de meta-análise médica.
"""

from .graph.multi_agent_graph import create_meta_analysis_system
from .models.state import MetaAnalysisState

__version__ = "1.0.0"
__author__ = "Nobrega Medtech"
__description__ = "Sistema multi-agente autônomo para meta-análises médicas"

__all__ = [
    "create_meta_analysis_system",
    "MetaAnalysisState",
]