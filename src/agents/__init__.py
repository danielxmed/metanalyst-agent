"""
Agentes especializados para o sistema metanalyst-agent.
Implementa arquitetura hub-and-spoke com orquestrador central.
"""

from .orchestrator import orchestrator_node, OrchestratorAgent
from .researcher import researcher_agent, ResearcherAgent  
from .processor import processor_agent, ProcessorAgent

__all__ = [
    # Orchestrator
    "orchestrator_node",
    "OrchestratorAgent",
    
    # Researcher
    "researcher_agent", 
    "ResearcherAgent",
    
    # Processor
    "processor_agent",
    "ProcessorAgent"
]