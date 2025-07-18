"""
CLI simplificado e otimizado para Metanalyst Agent
Interface mais rápida e responsiva seguindo a arquitetura agents-as-a-tool
"""

import click
import json
import time
import sys
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.rule import Rule

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Ensure .env file is loaded before importing other modules
import dotenv
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
if env_file.exists():
    dotenv.load_dotenv(env_file)
    # Verify Firecrawl API key is loaded
    if os.getenv("FIRECRAWL_API_KEY"):
        print("✅ Firecrawl API key loaded successfully")
    else:
        print("⚠️ Firecrawl API key not found in .env")

from src.models.state import MetanalysisState, create_initial_state
from src.utils.config import get_config, validate_environment
from src.agents.orchestrator import create_orchestrator_agent
from langchain_core.runnables import RunnableConfig


class SimplifiedMetanalystCLI:
    """
    CLI simplificado com melhor performance e responsividade
    """
    
    def __init__(self):
        """Initialize the simplified CLI."""
        self.console = Console()
        self.config = get_config()
        self.orchestrator = None
        
        # Initialize orchestrator
        try:
            self.orchestrator = create_orchestrator_agent()
            self.console.print("✅ Orchestrator inicializado com sucesso")
        except Exception as e:
            self.console.print(f"[red]❌ Erro ao inicializar orchestrator: {e}[/red]")
            sys.exit(1)
    
    def show_current_state(self, state: MetanalysisState) -> None:
        """Show current workflow state in a simple format."""
        self.console.print(Rule(f"[bold blue]Estado Atual - Iteração {state.get('orchestrator_iterations', 0)}[/bold blue]"))
        
        # Create status table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Componente", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Detalhes", style="yellow")
        
        # PICO Status
        pico = state.get("pico")
        if pico:
            table.add_row("🎯 PICO", "✅ Definido", f"P: {pico.get('patient', 'N/A')[:30]}...")
        else:
            table.add_row("🎯 PICO", "❌ Não definido", "Aguardando definição")
        
        # Literature Search
        urls_not_processed = state.get("url_not_processed", [])
        urls_processed = state.get("url_processed", [])
        total_urls = len(urls_not_processed) + len(urls_processed)
        
        if total_urls > 0:
            table.add_row("📚 Literatura", "✅ Encontrada", f"{total_urls} URLs ({len(urls_processed)} processadas)")
        else:
            table.add_row("📚 Literatura", "❌ Não encontrada", "Nenhuma URL encontrada")
        
        # Vector Store
        vector_ready = state.get("vector_store_ready", False)
        if vector_ready:
            table.add_row("🧠 Vector Store", "✅ Pronto", "Pronto para retrieval")
        else:
            table.add_row("🧠 Vector Store", "❌ Não pronto", "Aguardando processamento")
        
        # Report Status
        report_draft = state.get("report_draft")
        if report_draft:
            table.add_row("📝 Relatório", "✅ Rascunho", "Rascunho gerado")
        else:
            table.add_row("📝 Relatório", "❌ Não gerado", "Aguardando geração")
        
        # Final Report
        final_report = state.get("final_report")
        if final_report:
            table.add_row("📋 Relatório Final", "✅ Completo", "Metanálise concluída")
        else:
            table.add_row("📋 Relatório Final", "❌ Não completo", "Aguardando conclusão")
        
        self.console.print(table)
        self.console.print()
    
    def run_optimized_workflow(
        self, 
        user_request: str, 
        max_iterations: int = 15
    ) -> Dict[str, Any]:
        """
        Execute the optimized workflow with simplified interface.
        """
        # Create initial state
        state = create_initial_state()
        state["user_request"] = user_request
        state["max_iterations"] = max_iterations
        
        # Show initial banner
        self.console.print(Panel(
            Align.center(
                f"[bold blue]🤖 Metanalyst Agent - Workflow Otimizado[/bold blue]\n"
                f"[green]Solicitação: {user_request}[/green]\n"
                f"[yellow]Máximo de iterações: {max_iterations}[/yellow]"
            ),
            style="blue"
        ))
        
        # Run workflow iterations
        iteration = 0
        workflow_complete = False
        
        while iteration < max_iterations and not workflow_complete:
            iteration += 1
            
            # Update iteration in state
            state["orchestrator_iterations"] = iteration
            state["current_step"] = f"iteration_{iteration}"
            
            # Show current state
            self.show_current_state(state)
            
            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task(f"Executando iteração {iteration}...", total=None)
                
                try:
                    # Create state analysis
                    state_analysis = self._create_state_analysis(state)
                    
                    # Create orchestrator input
                    orchestrator_input = {
                        "messages": [{
                            "role": "user",
                            "content": f"""
                            🎯 ITERAÇÃO {iteration} - ORQUESTRADOR
                            
                            Solicitação Original: {user_request}
                            
                            Análise do Estado Atual:
                            {state_analysis}
                            
                            🤔 DECISÃO NECESSÁRIA:
                            Analise o estado atual e decida qual ferramenta usar para continuar o workflow.
                            
                            Se o workflow estiver completo, use get_workflow_status com status="complete".
                            Caso contrário, invoque a ferramenta apropriada para continuar.
                            
                            Você tem controle total - tome a melhor decisão com base no estado atual.
                            """
                        }]
                    }
                    
                    # Invoke orchestrator
                    if not self.orchestrator:
                        raise Exception("Orchestrator não disponível")
                    
                    config = RunnableConfig(recursion_limit=100)
                    result = self.orchestrator.invoke(orchestrator_input, config)
                    
                    # Process orchestrator results
                    last_message = result["messages"][-1]
                    decision_content = last_message.content
                    
                    # Check if workflow is complete
                    if "complete" in decision_content.lower() or "concluído" in decision_content.lower():
                        workflow_complete = True
                        self.console.print("[bold green]✅ Workflow marcado como completo![/bold green]")
                        break
                    
                    # Process tool results and update state
                    state_updates = self._process_orchestrator_results(state, result["messages"])
                    
                    # Update state with results
                    for key, value in state_updates.items():
                        state[key] = value
                    
                    # Update current agent info
                    state["current_agent"] = "orchestrator"
                    state["last_decision"] = decision_content
                    
                    # Show tool results
                    self._show_tool_results(state_updates)
                    
                except Exception as e:
                    progress.update(task, description=f"Erro na iteração {iteration}: {str(e)}")
                    self.console.print(f"[red]❌ Erro na iteração {iteration}: {str(e)}[/red]")
                    
                    # Add error to state
                    state["error_log"] = state.get("error_log", [])
                    state["error_log"].append({
                        "iteration": iteration,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Continue to next iteration unless critical error
                    if "critical" in str(e).lower():
                        break
                    
                    # Brief pause before next iteration
                    time.sleep(1)
        
        # Final state update
        state["workflow_complete"] = workflow_complete
        state["total_iterations"] = iteration
        state["completed_at"] = datetime.now().isoformat()
        
        # Show final results
        self._show_final_results(state, workflow_complete, iteration)
        
        return dict(state)
    
    def _create_state_analysis(self, state: MetanalysisState) -> str:
        """Create simplified state analysis for orchestrator."""
        analysis = []
        
        # PICO Status
        pico = state.get("pico")
        if pico:
            analysis.append(f"✅ PICO: Definido - {pico}")
        else:
            analysis.append("❌ PICO: Não definido")
        
        # Literature Search
        urls_not_processed = state.get("url_not_processed", [])
        urls_processed = state.get("url_processed", [])
        total_urls = len(urls_not_processed) + len(urls_processed)
        
        if total_urls > 0:
            analysis.append(f"✅ Literatura: {total_urls} URLs encontradas ({len(urls_processed)} processadas, {len(urls_not_processed)} pendentes)")
        else:
            analysis.append("❌ Literatura: Nenhuma URL encontrada")
        
        # Vector Store
        vector_ready = state.get("vector_store_ready", False)
        if vector_ready:
            analysis.append("✅ Vector Store: Pronto para retrieval")
        else:
            analysis.append("❌ Vector Store: Não pronto")
        
        # Report Status
        report_draft = state.get("report_draft")
        if report_draft:
            analysis.append("✅ Relatório: Rascunho gerado")
        else:
            analysis.append("❌ Relatório: Não gerado")
        
        # Final Report
        final_report = state.get("final_report")
        if final_report:
            analysis.append("✅ Relatório Final: Completo")
        else:
            analysis.append("❌ Relatório Final: Não completo")
        
        # Workflow info
        analysis.append(f"🆔 Workflow ID: {state.get('workflow_id', 'unknown')}")
        analysis.append(f"🔄 Iteração: {state.get('orchestrator_iterations', 0)}")
        
        return "\n".join(analysis)
    
    def _process_orchestrator_results(self, state: MetanalysisState, messages: list) -> Dict[str, Any]:
        """Process orchestrator messages and extract state updates."""
        state_updates = {}
        
        try:
            # Find tool messages with results
            for msg in messages:
                msg_type = type(msg).__name__
                
                if msg_type == 'ToolMessage' and hasattr(msg, 'name') and hasattr(msg, 'content'):
                    tool_name = getattr(msg, 'name', 'unknown')
                    tool_content = getattr(msg, 'content', '{}')
                    
                    try:
                        if isinstance(tool_content, str):
                            tool_result = json.loads(tool_content)
                            
                            # Update state based on tool results
                            if tool_name == "define_pico_structure" and tool_result.get("success"):
                                if "pico" in tool_result:
                                    state_updates["pico"] = tool_result["pico"]
                            
                            elif tool_name == "call_researcher_agent" and tool_result.get("success"):
                                if "urls_found" in tool_result:
                                    state_updates["url_not_processed"] = tool_result["urls_found"]
                            
                            elif tool_name == "call_processor_agent" and tool_result.get("success"):
                                if "url_processed" in tool_result:
                                    state_updates["url_processed"] = tool_result["url_processed"]
                                if "url_not_processed" in tool_result:
                                    state_updates["url_not_processed"] = tool_result["url_not_processed"]
                                if "vector_store_ready" in tool_result:
                                    state_updates["vector_store_ready"] = tool_result["vector_store_ready"]
                            
                            elif tool_name == "call_writer_agent" and tool_result.get("success"):
                                if "report_draft" in tool_result:
                                    state_updates["report_draft"] = tool_result["report_draft"]
                            
                            elif tool_name == "call_reviewer_agent" and tool_result.get("success"):
                                if "review_feedback" in tool_result:
                                    state_updates["review_feedback"] = tool_result["review_feedback"]
                            
                            elif tool_name == "call_analyst_agent" and tool_result.get("success"):
                                if "statistical_analysis" in tool_result:
                                    state_updates["statistical_analysis"] = tool_result["statistical_analysis"]
                            
                            elif tool_name == "call_editor_agent" and tool_result.get("success"):
                                if "final_report" in tool_result:
                                    state_updates["final_report"] = tool_result["final_report"]
                                    state_updates["final_report_path"] = tool_result.get("final_report_path")
                            
                            # Store tool result for display
                            state_updates[f"last_tool_result_{tool_name}"] = tool_result
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            self.console.print(f"[red]⚠️ Erro ao processar resultados: {e}[/red]")
        
        return state_updates
    
    def _show_tool_results(self, state_updates: Dict[str, Any]) -> None:
        """Show tool results in a simple format."""
        tool_results = []
        
        for key, value in state_updates.items():
            if key.startswith("last_tool_result_"):
                tool_name = key.replace("last_tool_result_", "")
                if isinstance(value, dict) and value.get("success"):
                    message = value.get("message", "Concluído com sucesso")
                    tool_results.append(f"✅ {tool_name}: {message}")
                elif isinstance(value, dict):
                    error = value.get("error", "Erro desconhecido")
                    tool_results.append(f"❌ {tool_name}: {error}")
        
        if tool_results:
            self.console.print(Panel(
                "\n".join(tool_results),
                title="[bold green]Resultados das Ferramentas[/bold green]",
                border_style="green"
            ))
    
    def _show_final_results(self, state: MetanalysisState, workflow_complete: bool, iteration: int) -> None:
        """Show final workflow results."""
        if workflow_complete:
            self.console.print(Panel(
                Align.center(
                    f"[bold green]🎉 Workflow Concluído com Sucesso![/bold green]\n"
                    f"[yellow]Total de iterações: {iteration}[/yellow]\n"
                    f"[cyan]Concluído em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/cyan]"
                ),
                style="green"
            ))
            
            # Show final report path if available
            final_report_path = state.get("final_report_path")
            if final_report_path:
                self.console.print(f"[bold blue]📊 Relatório Final: {final_report_path}[/bold blue]")
        else:
            self.console.print(Panel(
                Align.center(
                    f"[bold yellow]⚠️ Workflow Incompleto[/bold yellow]\n"
                    f"[red]Total de iterações: {iteration}[/red]\n"
                    f"[cyan]Interrompido em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/cyan]"
                ),
                style="yellow"
            ))
            
            # Show errors if any
            errors = state.get("error_log", [])
            if errors:
                self.console.print(f"[bold red]❌ Erros encontrados: {len(errors)}[/bold red]")
                for error in errors[-3:]:  # Show last 3 errors
                    self.console.print(f"   • {error.get('error', 'Erro desconhecido')}")


# CLI Commands
@click.group()
def cli():
    """🤖 Metanalyst Agent - Interface Simplificada e Otimizada"""
    pass


@cli.command()
@click.option('--request', '-r', help='Solicitação em linguagem natural para meta-análise')
@click.option('--max-iterations', '-i', default=15, help='Máximo de iterações do orquestrador')
def run(request, max_iterations):
    """Execute o workflow de meta-análise com interface otimizada."""
    
    # Validate environment
    if not validate_environment():
        console = Console()
        console.print("[red]❌ Validação do ambiente falhou. Verifique seu arquivo .env.[/red]")
        return
    
    # Create CLI app
    cli_app = SimplifiedMetanalystCLI()
    
    # Get request if not provided
    if not request:
        console = Console()
        console.print(Panel(
            Align.center(
                "[bold blue]🤖 Metanalyst Agent - Interface Otimizada[/bold blue]\n\n"
                "[bold]Exemplos de solicitações:[/bold]\n"
                "• 'Meta-análise sobre eficácia da metformina em diabéticos'\n"
                "• 'Analisar impacto de estatinas na prevenção cardiovascular'\n"
                "• 'Meta-análise de aspirina vs placebo para prevenção de AVC'"
            ),
            style="blue"
        ))
        request = click.prompt("🤔 Qual meta-análise deseja realizar?")
    
    # Run optimized workflow
    result = cli_app.run_optimized_workflow(
        user_request=request,
        max_iterations=max_iterations
    )
    
    # Show summary
    console = Console()
    if result.get("workflow_complete"):
        console.print("[bold green]🎉 Meta-análise concluída com sucesso![/bold green]")
        if result.get("final_report_path"):
            console.print(f"[bold blue]📊 Relatório: {result['final_report_path']}[/bold blue]")
    else:
        console.print("[bold yellow]⚠️ Meta-análise incompleta[/bold yellow]")


@cli.command()
def status():
    """Verificar status do sistema."""
    console = Console()
    console.print("[bold blue]🔍 Status do Sistema[/bold blue]")
    console.print("=" * 30)
    
    # Environment check
    if validate_environment():
        console.print("✅ Ambiente: Válido")
    else:
        console.print("❌ Ambiente: Inválido")
    
    # Configuration check
    try:
        config = get_config()
        console.print("✅ Configuração: Carregada")
        console.print(f"   • LLM: {config.llm.primary_model}")
        console.print(f"   • Max Papers: {config.search.max_papers_per_search}")
    except Exception as e:
        console.print(f"❌ Configuração: Erro - {e}")
    
    # Orchestrator check
    try:
        orchestrator = create_orchestrator_agent()
        console.print("✅ Orquestrador: Pronto")
    except Exception as e:
        console.print(f"❌ Orquestrador: Erro - {e}")


@cli.command()
def test():
    """Executar teste rápido do sistema."""
    console = Console()
    console.print("[bold blue]🧪 Executando teste rápido...[/bold blue]")
    
    cli_app = SimplifiedMetanalystCLI()
    result = cli_app.run_optimized_workflow(
        user_request="Teste de eficácia da metformina em diabéticos tipo 2",
        max_iterations=5
    )
    
    if result.get("pico"):
        console.print("✅ Teste concluído - PICO foi definido com sucesso")
    else:
        console.print("⚠️ Teste incompleto - PICO não foi definido")


if __name__ == "__main__":
    cli()
