#!/usr/bin/env python3
"""
Script principal para executar o Metanalyst-Agent CLI
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório do projeto ao Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar variáveis de ambiente se necessário
if not os.getenv("PYTHONPATH"):
    os.environ["PYTHONPATH"] = str(project_root)

# Importar e executar CLI
if __name__ == "__main__":
    from metanalyst_agent.cli import cli_main
    cli_main()
