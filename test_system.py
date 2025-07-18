#!/usr/bin/env python3
"""
Script de teste rápido para o sistema Metanalyst Agent
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.table import Table

def main():
    console = Console()
    
    console.print(Panel(
        Align.center(
            "[bold blue]🧪 Teste Rápido do Sistema Metanalyst Agent[/bold blue]\n"
            "[green]Verificando componentes principais...[/green]"
        ),
        style="blue"
    ))
    
    # Test results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Componente", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Detalhes", style="yellow")
    
    # Test 1: Environment validation
    try:
        from src.utils.config import validate_environment
        if validate_environment():
            table.add_row("🔧 Ambiente", "✅ Válido", "Variáveis de ambiente carregadas")
        else:
            table.add_row("🔧 Ambiente", "❌ Inválido", "Verifique o arquivo .env")
    except Exception as e:
        table.add_row("🔧 Ambiente", "❌ Erro", str(e))
    
    # Test 2: Configuration loading
    try:
        from src.utils.config import get_config
        config = get_config()
        table.add_row("⚙️ Configuração", "✅ Carregada", f"LLM: {config.llm.primary_model}")
    except Exception as e:
        table.add_row("⚙️ Configuração", "❌ Erro", str(e))
    
    # Test 3: Orchestrator creation
    try:
        from src.agents.orchestrator import create_orchestrator_agent
        orchestrator = create_orchestrator_agent()
        table.add_row("🎯 Orquestrador", "✅ Pronto", "Agente criado com sucesso")
    except Exception as e:
        table.add_row("🎯 Orquestrador", "❌ Erro", str(e))
    
    # Test 4: Orchestrator tools
    try:
        from src.tools.orchestrator_tools import ORCHESTRATOR_TOOLS
        tool_count = len(ORCHESTRATOR_TOOLS)
        table.add_row("🛠️ Ferramentas", "✅ Carregadas", f"{tool_count} ferramentas disponíveis")
    except Exception as e:
        table.add_row("🛠️ Ferramentas", "❌ Erro", str(e))
    
    # Test 5: Tavily client
    try:
        from src.tools.tavily_tools import get_tavily_client
        client = get_tavily_client()
        if client:
            table.add_row("🔍 Tavily", "✅ Disponível", "Cliente configurado")
        else:
            table.add_row("🔍 Tavily", "❌ Indisponível", "Verifique a API key")
    except Exception as e:
        table.add_row("🔍 Tavily", "❌ Erro", str(e))
    
    # Test 6: State management
    try:
        from src.models.state import create_initial_state
        state = create_initial_state()
        workflow_id = state.get('workflow_id', 'N/A') if state else 'N/A'
        if workflow_id != 'N/A':
            workflow_id = workflow_id[:8]
        table.add_row("📊 Estado", "✅ Funcional", f"Estado inicial criado (ID: {workflow_id})")
    except Exception as e:
        table.add_row("📊 Estado", "❌ Erro", str(e))
    
    # Test 7: Processor tools
    try:
        from src.tools.processor_tools import process_urls
        table.add_row("⚙️ Processador", "✅ Disponível", "Ferramenta de processamento carregada")
    except Exception as e:
        table.add_row("⚙️ Processador", "❌ Erro", str(e))
    
    # Test 8: CLI simplificado
    try:
        from src.cli_simplified import SimplifiedMetanalystCLI
        cli = SimplifiedMetanalystCLI()
        if cli:
            table.add_row("🖥️ CLI", "✅ Pronto", "Interface simplificada carregada")
        else:
            table.add_row("🖥️ CLI", "❌ Erro", "CLI não foi criado")
    except Exception as e:
        table.add_row("🖥️ CLI", "❌ Erro", str(e))
    
    console.print(table)
    console.print()
    
    # Test summary
    console.print(Panel(
        Align.center(
            "[bold green]🎉 Teste Concluído![/bold green]\n"
            "[yellow]Para executar o sistema:[/yellow]\n"
            "[cyan]python -m src.cli_simplified run[/cyan]\n"
            "[cyan]python -m src.cli_simplified status[/cyan]\n"
            "[cyan]python -m src.cli_simplified test[/cyan]"
        ),
        style="green"
    ))

if __name__ == "__main__":
    main()
