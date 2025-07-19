"""
Metanalyst-Agent: Automated Meta-Analysis Generation using Multi-Agent Systems

The first open-source project by Nobrega Medtech for automated meta-analysis 
generation using Python and LangGraph.
"""

from .main import MetanalystAgent
from .config.settings import Settings

__version__ = "0.1.0"
__author__ = "Nobrega Medtech"
__email__ = "contact@nobregamedtech.com"

__all__ = [
    "MetanalystAgent",
    "Settings",
]