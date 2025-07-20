#!/usr/bin/env python3
"""
CLI Interface for Metanalyst-Agent System
Provides interactive command-line interface with detailed debug logging.
"""

import asyncio
import argparse
import json
import sys
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
from contextlib import asynccontextmanager

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import print as rprint
from rich.prompt import Prompt, Confirm

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.errors import GraphRecursionError

from .main import MetanalystAgent
from .config.settings import settings
from .utils.logging_config import setup_logging


# Rich console para output colorido
console = Console()

# Logger específico para CLI
cli_logger = logging.getLogger("metanalyst_cli")


class DebugFormatter:
    """Formatador para logs de debug detalhados"""
    
    @staticmethod
    def format_message(message: BaseMessage, indent: int = 0) -> str:
        """Formata mensagem para exibição"""
        prefix = "  " * indent
        
        if isinstance(message, HumanMessage):
            return f"{prefix}🧑 [bold blue]User:[/bold blue] {message.content}"
        elif isinstance(message, AIMessage):
            role = getattr(message, 'name', 'Assistant')
            content = message.content[:500] + "..." if len(str(message.content)) > 500 else message.content
            
            formatted = f"{prefix}🤖 [bold green]{role}:[/bold green] {content}"
            
            # Adicionar tool calls se existirem
            if hasattr(message, 'tool_calls') and message.tool_calls:
                formatted += f"\n{prefix}   🛠️ [yellow]Tool Calls:[/yellow]"
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    formatted += f"\n{prefix}      - {tool_name}"
            
            return formatted
        elif isinstance(message, ToolMessage):
            tool_name = getattr(message, 'name', 'unknown_tool')
            content = str(message.content)[:300] + "..." if len(str(message.content)) > 300 else str(message.content)
            return f"{prefix}⚙️ [bold cyan]{tool_name}:[/bold cyan] {content}"
        else:
            return f"{prefix}📄 [dim]{type(message).__name__}:[/dim] {str(message)[:200]}"
    
    @staticmethod
    def format_state_update(update: Dict[str, Any], node_name: str) -> str:
        """Formata update de estado"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        formatted = f"[dim]{timestamp}[/dim] 🔄 [bold magenta]Node '{node_name}'[/bold magenta]\n"
        
        # Informações básicas do estado
        if 'messages' in update:
            messages = update['messages']
            if messages:
                formatted += f"   📨 Messages: {len(messages)} total\n"
                # Última mensagem
                last_msg = messages[-1]
                formatted += f"   └─ Last: {DebugFormatter.format_message(last_msg, indent=1)}\n"
        
        # Estatísticas importantes
        if 'candidate_urls' in update:
            count = len(update['candidate_urls'])
            formatted += f"   🔍 Articles Found: {count}\n"
        
        if 'processed_articles' in update:
            count = len(update['processed_articles'])
            formatted += f"   ✅ Articles Processed: {count}\n"
        
        if 'failed_urls' in update:
            count = len(update['failed_urls'])
            if count > 0:
                formatted += f"   ❌ Failed URLs: {count}\n"
        
        if 'current_phase' in update:
            phase = update['current_phase']
            formatted += f"   📍 Phase: [bold]{phase}[/bold]\n"
        
        if 'quality_scores' in update:
            scores = update['quality_scores']
            if scores:
                formatted += f"   📊 Quality Scores: {scores}\n"
        
        return formatted
    
    @staticmethod
    def format_error(error: Exception, context: str = "") -> str:
        """Formata erro para exibição"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_type = type(error).__name__
        
        formatted = f"[dim]{timestamp}[/dim] ❌ [bold red]Error in {context}[/bold red]\n"
        formatted += f"   🏷️ Type: {error_type}\n"
        formatted += f"   💬 Message: {str(error)}\n"
        
        return formatted


class ProgressTracker:
    """Rastreia progresso da meta-análise"""
    
    def __init__(self):
        self.start_time = time.time()
        self.phases = []
        self.current_phase = None
        self.articles_found = 0
        self.articles_processed = 0
        self.articles_failed = 0
        self.quality_scores = {}
        self.agent_activities = {}
    
    def update(self, state_update: Dict[str, Any], node_name: str):
        """Atualiza tracking com nova informação"""
        if 'current_phase' in state_update:
            new_phase = state_update['current_phase']
            if new_phase != self.current_phase:
                self.current_phase = new_phase
                self.phases.append({
                    'phase': new_phase,
                    'timestamp': datetime.now(),
                    'elapsed': time.time() - self.start_time
                })
        
        if 'candidate_urls' in state_update:
            self.articles_found = len(state_update['candidate_urls'])
        
        if 'processed_articles' in state_update:
            self.articles_processed = len(state_update['processed_articles'])
        
        if 'failed_urls' in state_update:
            self.articles_failed = len(state_update['failed_urls'])
        
        if 'quality_scores' in state_update:
            self.quality_scores.update(state_update['quality_scores'])
        
        # Rastrear atividade do agente
        if node_name not in self.agent_activities:
            self.agent_activities[node_name] = 0
        self.agent_activities[node_name] += 1
    
    def get_summary_table(self) -> Table:
        """Gera tabela de resumo do progresso"""
        table = Table(title="Meta-Analysis Progress Summary", show_header=True, header_style="bold magenta")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        # Tempo decorrido
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed:.1f}s"
        table.add_row("Elapsed Time", elapsed_str, "⏱️")
        
        # Fase atual
        current_phase = self.current_phase or "starting"
        table.add_row("Current Phase", current_phase.title(), "📍")
        
        # Artigos
        table.add_row("Articles Found", str(self.articles_found), "🔍")
        table.add_row("Articles Processed", str(self.articles_processed), "✅")
        if self.articles_failed > 0:
            table.add_row("Articles Failed", str(self.articles_failed), "❌")
        
        # Scores de qualidade
        if self.quality_scores:
            avg_score = sum(self.quality_scores.values()) / len(self.quality_scores)
            table.add_row("Avg Quality Score", f"{avg_score:.2f}", "📊")
        
        # Atividade dos agentes
        if self.agent_activities:
            most_active = max(self.agent_activities, key=self.agent_activities.get)
            table.add_row("Most Active Agent", most_active.title(), "🤖")
        
        return table


class MetanalystCLI:
    """Interface CLI principal para o Metanalyst-Agent"""
    
    def __init__(self, debug: bool = False, use_persistent_storage: bool = True):
        self.debug = debug
        self.use_persistent_storage = use_persistent_storage
        self.console = console
        self.progress_tracker = None
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar agente (lazy)
        self._agent = None
    
    def _setup_logging(self):
        """Configura sistema de logging"""
        # Configuração base
        log_level = "DEBUG" if self.debug else "INFO"
        log_file = settings.logs_dir / "cli.log"
        setup_logging(
            log_level=log_level,
            log_file=log_file,
            enable_console=True
        )
        
        # Logger específico para CLI com Rich handler
        cli_logger.handlers.clear()
        
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        formatter = logging.Formatter(
            "%(message)s",
            datefmt="[%X]"
        )
        rich_handler.setFormatter(formatter)
        
        cli_logger.addHandler(rich_handler)
        cli_logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        cli_logger.propagate = False
    
    @property
    def agent(self) -> MetanalystAgent:
        """Lazy loading do agente"""
        if self._agent is None:
            with self.console.status("[bold blue]Inicializando Metanalyst-Agent...", spinner="dots"):
                self._agent = MetanalystAgent(
                    use_persistent_storage=self.use_persistent_storage,
                    debug=self.debug
                )
            self.console.print("✅ [green]Metanalyst-Agent inicializado com sucesso![/green]")
        return self._agent
    
    def print_banner(self):
        """Exibe banner do sistema"""
        banner = """
╭─────────────────────────────────────────────────────────────────╮
│                     🔬 METANALYST-AGENT CLI                     │
│                                                                 │
│              Sistema Automatizado de Meta-Análise              │
│                     com Agentes Multi-LLM                      │
╰─────────────────────────────────────────────────────────────────╯
        """
        
        self.console.print(Panel(
            banner,
            style="bold blue",
            border_style="blue"
        ))
        
        # Informações do sistema
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column("Key", style="cyan")
        info_table.add_column("Value", style="white")
        
        storage_type = "PostgreSQL" if self.use_persistent_storage else "In-Memory"
        debug_status = "ON" if self.debug else "OFF"
        
        info_table.add_row("🗄️ Storage:", storage_type)
        info_table.add_row("🧠 Model:", settings.openai_model)
        info_table.add_row("🐛 Debug:", debug_status)
        info_table.add_row("📁 Logs:", str(settings.logs_dir))
        
        self.console.print(info_table)
        self.console.print()
    
    async def interactive_session(self):
        """Sessão interativa principal"""
        self.print_banner()
        
        while True:
            try:
                # Menu principal
                self.console.print("\n[bold cyan]Opções disponíveis:[/bold cyan]")
                self.console.print("1. 🔬 Nova Meta-Análise")
                self.console.print("2. 📊 Listar Análises Anteriores")
                self.console.print("3. ⚙️ Configurações")
                self.console.print("4. 🔍 Debug/Teste")
                self.console.print("5. ❓ Ajuda")
                self.console.print("6. 🚪 Sair")
                
                choice = Prompt.ask("\n[bold yellow]Escolha uma opção", choices=["1", "2", "3", "4", "5", "6"])
                
                if choice == "1":
                    await self.run_meta_analysis()
                elif choice == "2":
                    await self.list_analyses()
                elif choice == "3":
                    await self.show_settings()
                elif choice == "4":
                    await self.debug_mode()
                elif choice == "5":
                    self.show_help()
                elif choice == "6":
                    if Confirm.ask("\n[yellow]Tem certeza que deseja sair?[/yellow]"):
                        break
                
            except KeyboardInterrupt:
                self.console.print("\n\n[yellow]Operação cancelada pelo usuário.[/yellow]")
                if Confirm.ask("[yellow]Deseja sair?[/yellow]"):
                    break
                continue
            except Exception as e:
                cli_logger.error(f"Erro na sessão interativa: {e}")
                self.console.print(f"[red]❌ Erro inesperado: {e}[/red]")
                continue
        
        self.console.print("\n[green]👋 Obrigado por usar o Metanalyst-Agent![/green]")
    
    async def run_meta_analysis(self):
        """Executa uma nova meta-análise"""
        self.console.print("\n[bold blue]🔬 Nova Meta-Análise[/bold blue]")
        
        # Coletar informações do usuário
        query = Prompt.ask("\n[cyan]Descreva sua meta-análise[/cyan]")
        
        if not query.strip():
            self.console.print("[red]Query não pode estar vazia![/red]")
            return
        
        # Configurações opcionais
        self.console.print("\n[dim]Configurações avançadas (pressione Enter para usar padrões):[/dim]")
        
        max_articles = Prompt.ask("Máximo de artigos", default="50")
        try:
            max_articles = int(max_articles)
        except ValueError:
            max_articles = 50
        
        quality_threshold = Prompt.ask("Threshold de qualidade (0.0-1.0)", default="0.8")
        try:
            quality_threshold = float(quality_threshold)
        except ValueError:
            quality_threshold = 0.8
        
        max_time_minutes = Prompt.ask("Tempo máximo (minutos)", default="30")
        try:
            max_time_minutes = int(max_time_minutes)
        except ValueError:
            max_time_minutes = 30
        
        # Modo de streaming
        stream_modes = ["values", "updates", "debug"]
        stream_mode = Prompt.ask(
            "Modo de debug",
            choices=stream_modes,
            default="updates"
        )
        
        # Executar meta-análise
        self.console.print(f"\n[green]🚀 Iniciando meta-análise...[/green]")
        self.console.print(f"[dim]Query: {query}[/dim]")
        self.console.print(f"[dim]Máx. artigos: {max_articles}, Qualidade: {quality_threshold}, Tempo: {max_time_minutes}min[/dim]")
        
        # Inicializar tracking
        self.progress_tracker = ProgressTracker()
        
        try:
            await self.stream_meta_analysis(
                query=query,
                max_articles=max_articles,
                quality_threshold=quality_threshold,
                max_time_minutes=max_time_minutes,
                stream_mode=stream_mode
            )
        except Exception as e:
            cli_logger.error(f"Erro na meta-análise: {e}")
            self.console.print(f"\n[red]❌ Erro durante execução: {e}[/red]")
            
            if self.debug:
                import traceback
                self.console.print("[dim]Stack trace:[/dim]")
                self.console.print(traceback.format_exc())
    
    async def stream_meta_analysis(
        self,
        query: str,
        max_articles: int = 50,
        quality_threshold: float = 0.8,
        max_time_minutes: int = 30,
        stream_mode: str = "updates"
    ):
        """Stream meta-análise com logs detalhados"""
        
        start_time = time.time()
        
        # Configurar agente
        agent = self.agent
        
        # Gerar PICO
        from .agents.orchestrator_agent import generate_pico_from_query
        pico = generate_pico_from_query(query)
        
        # Mostrar PICO extraído
        pico_panel = Panel(
            f"**Population:** {pico['P']}\n"
            f"**Intervention:** {pico['I']}\n"
            f"**Comparison:** {pico['C']}\n"
            f"**Outcome:** {pico['O']}",
            title="📋 PICO Framework",
            border_style="green"
        )
        self.console.print(pico_panel)
        
        # Estado inicial
        from .state.meta_analysis_state import create_initial_state
        
        meta_analysis_id = str(uuid.uuid4())
        thread_id = f"thread_{meta_analysis_id[:8]}"
        
        config_params = {
            "max_articles": max_articles,
            "quality_threshold": quality_threshold,
            "quality_thresholds": {
                "researcher": quality_threshold,
                "processor": quality_threshold,
                "analyst": quality_threshold,
                "writer": quality_threshold,
                "reviewer": quality_threshold * 1.1,
            },
        }
        
        initial_state = create_initial_state(
            research_question=query,
            meta_analysis_id=meta_analysis_id,
            thread_id=thread_id,
            config=config_params
        )
        
        initial_state.update({
            "pico": pico,
            "research_question": query,
            "current_phase": "search",
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=f"Vou conduzir uma meta-análise com base no PICO extraído. Começando busca por literatura...")
            ]
        })
        
        # Configuração de execução
        config = {
            "recursion_limit": settings.default_recursion_limit,
            "configurable": {
                "thread_id": initial_state["thread_id"],
                "checkpoint_ns": "meta_analysis",
            }
        }
        
        cli_logger.info(f"Iniciando meta-análise - ID: {meta_analysis_id}")
        cli_logger.info(f"Thread ID: {thread_id}")
        cli_logger.info(f"Stream Mode: {stream_mode}")
        
        # Progress bar setup
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            task = progress.add_task("[green]Processando meta-análise...", total=None)
            
            try:
                # Stream com timeout
                async with asyncio.timeout(max_time_minutes * 60):
                    final_state = None
                    
                    async for state_update in agent.graph.astream(
                        initial_state,
                        config,
                        stream_mode=stream_mode
                    ):
                        final_state = state_update
                        
                        # Processar diferentes tipos de stream
                        if stream_mode == "debug":
                            await self._process_debug_chunk(state_update)
                        elif stream_mode == "updates":
                            await self._process_updates_chunk(state_update)
                        else:  # values
                            await self._process_values_chunk(state_update)
                        
                        # Atualizar progress tracker
                        if isinstance(state_update, dict):
                            node_name = "unknown"
                            if stream_mode == "updates" and state_update:
                                node_name = next(iter(state_update.keys()))
                            elif "current_agent" in state_update:
                                node_name = state_update["current_agent"]
                            
                            self.progress_tracker.update(state_update, node_name)
                        
                        # Atualizar descrição do progress
                        current_phase = getattr(self.progress_tracker, 'current_phase', 'processing')
                        articles_found = getattr(self.progress_tracker, 'articles_found', 0)
                        articles_processed = getattr(self.progress_tracker, 'articles_processed', 0)
                        
                        progress.update(
                            task,
                            description=f"[green]Fase: {current_phase} | Encontrados: {articles_found} | Processados: {articles_processed}"
                        )
                        
                        # Verificar condições de parada
                        if isinstance(state_update, dict):
                            if state_update.get("final_report") or state_update.get("force_stop"):
                                break
                
                # Exibir resumo final
                await self._show_final_summary(final_state, start_time)
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                self.console.print(f"\n[yellow]⏰ Meta-análise interrompida por timeout após {elapsed:.1f}s[/yellow]")
                
                # Tentar recuperar estado parcial
                if self.progress_tracker:
                    summary_table = self.progress_tracker.get_summary_table()
                    self.console.print(summary_table)
                
            except GraphRecursionError as e:
                self.console.print(f"\n[red]🔄 Limite de recursão atingido: {e}[/red]")
                cli_logger.error(f"Recursion limit reached: {e}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                error_msg = DebugFormatter.format_error(e, "meta-analysis execution")
                self.console.print(error_msg)
                cli_logger.error(f"Meta-analysis failed after {elapsed:.1f}s: {e}")
                
                if self.debug:
                    import traceback
                    cli_logger.debug(traceback.format_exc())
    
    async def _process_debug_chunk(self, chunk: Dict[str, Any]):
        """Processa chunk no modo debug (máximo detalhe)"""
        # Debug mode tem estrutura: {'type': 'node', 'timestamp': ..., 'step': ..., 'payload': ...}
        
        if isinstance(chunk, dict):
            chunk_type = chunk.get('type', 'unknown')
            timestamp = chunk.get('timestamp', datetime.now().isoformat())
            
            if chunk_type == 'node':
                node_name = chunk.get('payload', {}).get('name', 'unknown')
                state = chunk.get('payload', {}).get('state', {})
                
                self.console.print(f"\n[dim]{timestamp}[/dim] 🔄 [bold magenta]Debug - Node '{node_name}'[/bold magenta]")
                
                # Log estado completo em debug
                if self.debug and state:
                    cli_logger.debug(f"Full state for node {node_name}: {json.dumps(state, default=str, indent=2)}")
                
                # Mostrar informações relevantes
                if 'messages' in state and state['messages']:
                    last_msg = state['messages'][-1]
                    formatted_msg = DebugFormatter.format_message(last_msg)
                    self.console.print(f"   {formatted_msg}")
                
                # Estatísticas
                if 'candidate_urls' in state:
                    count = len(state['candidate_urls'])
                    self.console.print(f"   🔍 Articles: {count}")
                
                if 'processed_articles' in state:
                    count = len(state['processed_articles'])
                    self.console.print(f"   ✅ Processed: {count}")
        
        else:
            # Fallback para outros tipos
            self.console.print(f"[dim]Debug chunk: {str(chunk)[:200]}...[/dim]")
    
    async def _process_updates_chunk(self, chunk: Dict[str, Any]):
        """Processa chunk no modo updates (por agente)"""
        # Updates mode: {node_name: {state_update}}
        
        if isinstance(chunk, dict):
            for node_name, state_update in chunk.items():
                if isinstance(state_update, dict):
                    formatted = DebugFormatter.format_state_update(state_update, node_name)
                    self.console.print(formatted)
                    
                    # Log detalhado das mensagens
                    if 'messages' in state_update:
                        messages = state_update['messages']
                        for msg in messages[-2:]:  # Últimas 2 mensagens
                            formatted_msg = DebugFormatter.format_message(msg)
                            cli_logger.info(formatted_msg)
                else:
                    self.console.print(f"[dim]{node_name}: {str(state_update)[:100]}...[/dim]")
    
    async def _process_values_chunk(self, chunk: Dict[str, Any]):
        """Processa chunk no modo values (estado completo)"""
        # Values mode: estado completo a cada step
        
        if isinstance(chunk, dict):
            current_agent = chunk.get('current_agent', 'system')
            current_phase = chunk.get('current_phase', 'unknown')
            
            # Mostrar progresso básico
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.print(f"[dim]{timestamp}[/dim] 🔄 [bold]{current_agent}[/bold] ({current_phase})")
            
            # Última mensagem
            if 'messages' in chunk and chunk['messages']:
                last_msg = chunk['messages'][-1]
                formatted_msg = DebugFormatter.format_message(last_msg)
                self.console.print(f"   {formatted_msg}")
                
                # Log completo da mensagem
                cli_logger.info(f"[{current_agent}] {last_msg.content}")
            
            # Estatísticas resumidas
            stats = []
            if 'candidate_urls' in chunk:
                stats.append(f"Found: {len(chunk['candidate_urls'])}")
            if 'processed_articles' in chunk:
                stats.append(f"Processed: {len(chunk['processed_articles'])}")
            if 'failed_urls' in chunk:
                failed_count = len(chunk['failed_urls'])
                if failed_count > 0:
                    stats.append(f"Failed: {failed_count}")
            
            if stats:
                stats_str = " | ".join(stats)
                self.console.print(f"   📊 {stats_str}")
    
    async def _show_final_summary(self, final_state: Dict[str, Any], start_time: float):
        """Mostra resumo final da meta-análise"""
        elapsed = time.time() - start_time
        
        self.console.print("\n" + "="*80)
        self.console.print("[bold green]✅ META-ANÁLISE CONCLUÍDA![/bold green]")
        self.console.print("="*80)
        
        # Resumo executivo
        if self.progress_tracker:
            summary_table = self.progress_tracker.get_summary_table()
            self.console.print(summary_table)
        
        # Detalhes do resultado
        if final_state:
            results_panel = self._format_results_panel(final_state, elapsed)
            self.console.print(results_panel)
            
            # Salvar resultados se solicitado
            if Confirm.ask("\n[cyan]Deseja salvar os resultados?[/cyan]"):
                await self._save_results(final_state)
        
        cli_logger.info(f"Meta-analysis completed in {elapsed:.1f}s")
    
    def _format_results_panel(self, state: Dict[str, Any], elapsed: float) -> Panel:
        """Formata painel com resultados finais"""
        content = f"**⏱️ Tempo Total:** {elapsed:.1f}s\n"
        content += f"**🆔 ID:** {state.get('meta_analysis_id', 'N/A')}\n"
        content += f"**📍 Fase Final:** {state.get('current_phase', 'N/A')}\n"
        
        # Estatísticas de artigos
        articles_found = len(state.get('candidate_urls', []))
        articles_processed = len(state.get('processed_articles', []))
        articles_failed = len(state.get('failed_urls', []))
        
        content += f"**🔍 Artigos Encontrados:** {articles_found}\n"
        content += f"**✅ Artigos Processados:** {articles_processed}\n"
        if articles_failed > 0:
            content += f"**❌ Artigos Falharam:** {articles_failed}\n"
        
        # Scores de qualidade
        if state.get('quality_scores'):
            scores = state['quality_scores']
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            content += f"**📊 Score Médio:** {avg_score:.2f}\n"
        
        # Relatório final
        if state.get('final_report'):
            content += f"**📝 Relatório:** Gerado com sucesso\n"
        elif state.get('draft_report'):
            content += f"**📝 Relatório:** Rascunho disponível\n"
        
        return Panel(content, title="🎯 Resultados Finais", border_style="green")
    
    async def _save_results(self, state: Dict[str, Any]):
        """Salva resultados em arquivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metanalysis_{timestamp}.json"
        filepath = settings.data_dir / filename
        
        try:
            # Preparar dados para salvamento
            results = {
                "meta_analysis_id": state.get('meta_analysis_id'),
                "timestamp": timestamp,
                "pico": state.get('pico', {}),
                "research_question": state.get('research_question'),
                "statistics": {
                    "articles_found": len(state.get('candidate_urls', [])),
                    "articles_processed": len(state.get('processed_articles', [])),
                    "articles_failed": len(state.get('failed_urls', [])),
                },
                "quality_scores": state.get('quality_scores', {}),
                "final_report": state.get('final_report'),
                "draft_report": state.get('draft_report'),
                "citations": state.get('citations', []),
                "execution_summary": {
                    "final_phase": state.get('current_phase'),
                    "total_iterations": state.get('global_iterations', 0),
                }
            }
            
            # Salvar arquivo JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.console.print(f"[green]✅ Resultados salvos em: {filepath}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Erro ao salvar: {e}[/red]")
            cli_logger.error(f"Error saving results: {e}")
    
    async def list_analyses(self):
        """Lista análises anteriores"""
        self.console.print("\n[bold blue]📊 Análises Anteriores[/bold blue]")
        
        try:
            # Buscar arquivos de resultados
            results_files = list(settings.data_dir.glob("metanalysis_*.json"))
            
            if not results_files:
                self.console.print("[yellow]Nenhuma análise anterior encontrada.[/yellow]")
                return
            
            # Criar tabela
            table = Table(title="Análises Anteriores", show_header=True, header_style="bold cyan")
            table.add_column("Data", style="cyan")
            table.add_column("ID", style="white")
            table.add_column("Pergunta", style="green")
            table.add_column("Artigos", style="yellow")
            table.add_column("Status", style="magenta")
            
            for file_path in sorted(results_files, reverse=True)[:10]:  # Últimas 10
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    timestamp = data.get('timestamp', 'N/A')
                    analysis_id = data.get('meta_analysis_id', 'N/A')[:8]
                    question = data.get('research_question', 'N/A')[:50] + "..."
                    articles = data.get('statistics', {}).get('articles_processed', 0)
                    status = "✅ Completa" if data.get('final_report') else "📝 Rascunho"
                    
                    table.add_row(timestamp, analysis_id, question, str(articles), status)
                    
                except Exception as e:
                    cli_logger.warning(f"Error reading file {file_path}: {e}")
                    continue
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]❌ Erro ao listar análises: {e}[/red]")
            cli_logger.error(f"Error listing analyses: {e}")
    
    async def show_settings(self):
        """Mostra configurações atuais"""
        self.console.print("\n[bold blue]⚙️ Configurações do Sistema[/bold blue]")
        
        settings_table = Table(show_header=True, header_style="bold magenta")
        settings_table.add_column("Configuração", style="cyan")
        settings_table.add_column("Valor", style="white")
        settings_table.add_column("Descrição", style="dim")
        
        # Configurações principais
        settings_table.add_row(
            "Storage Type",
            "PostgreSQL" if self.use_persistent_storage else "In-Memory",
            "Tipo de armazenamento de dados"
        )
        
        settings_table.add_row(
            "Debug Mode",
            "ON" if self.debug else "OFF",
            "Logging detalhado e debug"
        )
        
        settings_table.add_row(
            "Model",
            settings.openai_model,
            "Modelo LLM principal"
        )
        
        settings_table.add_row(
            "Database URL",
            settings.database_url if settings.database_url else "Not configured",
            "URL do banco de dados"
        )
        
        settings_table.add_row(
            "Recursion Limit",
            str(settings.default_recursion_limit),
            "Limite de iterações do grafo"
        )
        
        settings_table.add_row(
            "Data Directory",
            str(settings.data_dir),
            "Diretório de dados"
        )
        
        settings_table.add_row(
            "Logs Directory",
            str(settings.logs_dir),
            "Diretório de logs"
        )
        
        self.console.print(settings_table)
        
        # Opções de configuração
        if Confirm.ask("\n[cyan]Deseja modificar alguma configuração?[/cyan]"):
            await self._modify_settings()
    
    async def _modify_settings(self):
        """Permite modificar configurações"""
        self.console.print("\n[bold yellow]🔧 Modificar Configurações[/bold yellow]")
        self.console.print("[dim]Nota: Algumas mudanças requerem reinicialização.[/dim]")
        
        options = [
            "1. Toggle Debug Mode",
            "2. Toggle Storage Type",
            "3. Change Recursion Limit",
            "4. Voltar"
        ]
        
        for option in options:
            self.console.print(option)
        
        choice = Prompt.ask("\nEscolha", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            self.debug = not self.debug
            self._setup_logging()  # Reconfigurar logging
            status = "ON" if self.debug else "OFF"
            self.console.print(f"[green]Debug mode: {status}[/green]")
            
        elif choice == "2":
            self.use_persistent_storage = not self.use_persistent_storage
            storage_type = "PostgreSQL" if self.use_persistent_storage else "In-Memory"
            self.console.print(f"[green]Storage type: {storage_type}[/green]")
            self.console.print("[yellow]⚠️ Reinicie o CLI para aplicar mudança de storage.[/yellow]")
            
        elif choice == "3":
            current_limit = settings.default_recursion_limit
            new_limit = Prompt.ask(f"Novo limite de recursão (atual: {current_limit})", default=str(current_limit))
            try:
                settings.default_recursion_limit = int(new_limit)
                self.console.print(f"[green]Limite de recursão: {new_limit}[/green]")
            except ValueError:
                self.console.print("[red]Valor inválido![/red]")
    
    async def debug_mode(self):
        """Modo debug/teste"""
        self.console.print("\n[bold blue]🔍 Modo Debug/Teste[/bold blue]")
        
        debug_options = [
            "1. 🧪 Testar Conexão com Agente",
            "2. 🔍 Testar Busca Tavily",
            "3. 📊 Verificar Estado do Sistema",
            "4. 🗄️ Testar Banco de Dados",
            "5. 📝 Gerar Log de Teste",
            "6. Voltar"
        ]
        
        for option in debug_options:
            self.console.print(option)
        
        choice = Prompt.ask("\nEscolha", choices=["1", "2", "3", "4", "5", "6"])
        
        if choice == "1":
            await self._test_agent_connection()
        elif choice == "2":
            await self._test_tavily_search()
        elif choice == "3":
            await self._check_system_status()
        elif choice == "4":
            await self._test_database()
        elif choice == "5":
            await self._generate_test_log()
    
    async def _test_agent_connection(self):
        """Testa conexão com agente"""
        self.console.print("\n[cyan]🧪 Testando conexão com agente...[/cyan]")
        
        try:
            with self.console.status("[bold blue]Inicializando agente...", spinner="dots"):
                agent = self.agent
            
            self.console.print("[green]✅ Agente inicializado com sucesso![/green]")
            
            # Testar PICO generation
            test_query = "Eficácia da meditação mindfulness para ansiedade"
            
            from .agents.orchestrator_agent import generate_pico_from_query
            pico = generate_pico_from_query(test_query)
            
            self.console.print(f"[green]✅ PICO gerado:[/green]")
            pico_table = Table(show_header=False)
            pico_table.add_column("Campo", style="cyan")
            pico_table.add_column("Valor", style="white")
            
            pico_table.add_row("Population", pico['P'])
            pico_table.add_row("Intervention", pico['I'])
            pico_table.add_row("Comparison", pico['C'])
            pico_table.add_row("Outcome", pico['O'])
            
            self.console.print(pico_table)
            
        except Exception as e:
            self.console.print(f"[red]❌ Erro no teste: {e}[/red]")
            cli_logger.error(f"Agent connection test failed: {e}")
    
    async def _test_tavily_search(self):
        """Testa busca Tavily"""
        self.console.print("\n[cyan]🔍 Testando Tavily Search...[/cyan]")
        
        query = Prompt.ask("Query de teste", default="mindfulness meditation anxiety")
        
        try:
            # Testar importação e configuração
            from .tools.research_tools import search_literature
            
            with self.console.status(f"[bold blue]Buscando: {query}...", spinner="dots"):
                # Simular chamada da ferramenta
                results = await asyncio.create_task(
                    asyncio.to_thread(search_literature, query, max_results=5)
                )
            
            if isinstance(results, list) and results:
                self.console.print(f"[green]✅ Encontrados {len(results)} resultados[/green]")
                
                # Mostrar primeiros resultados
                results_table = Table(title="Primeiros Resultados", show_header=True, header_style="bold green")
                results_table.add_column("Título", style="cyan")
                results_table.add_column("Domínio", style="yellow")
                results_table.add_column("Score", style="white")
                
                for result in results[:3]:
                    title = result.get('title', 'N/A')[:50] + "..."
                    domain = result.get('source_domain', 'N/A')
                    score = f"{result.get('score', 0):.2f}"
                    results_table.add_row(title, domain, score)
                
                self.console.print(results_table)
            else:
                self.console.print("[yellow]⚠️ Nenhum resultado encontrado[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Erro no teste Tavily: {e}[/red]")
            cli_logger.error(f"Tavily search test failed: {e}")
    
    async def _check_system_status(self):
        """Verifica status do sistema"""
        self.console.print("\n[cyan]📊 Verificando status do sistema...[/cyan]")
        
        status_table = Table(title="Status do Sistema", show_header=True, header_style="bold blue")
        status_table.add_column("Componente", style="cyan")
        status_table.add_column("Status", style="white")
        status_table.add_column("Detalhes", style="dim")
        
        # Verificar diretórios
        dirs_to_check = [
            ("Data Directory", settings.data_dir),
            ("Logs Directory", settings.logs_dir),
            ("Checkpoints Directory", settings.data_dir / "checkpoints"),
            ("Vector Store Directory", settings.data_dir / "vector_store")
        ]
        
        for name, path in dirs_to_check:
            if path.exists():
                status = "✅ OK"
                details = f"Exists: {path}"
            else:
                status = "❌ Missing"
                details = f"Not found: {path}"
            
            status_table.add_row(name, status, details)
        
        # Verificar variáveis de ambiente
        env_vars = [
            ("OPENAI_API_KEY", "OpenAI API Key"),
            ("TAVILY_API_KEY", "Tavily API Key"),
            ("DATABASE_URL", "Database URL"),
        ]
        
        for env_var, description in env_vars:
            import os
            value = os.getenv(env_var)
            if value:
                status = "✅ Set"
                details = f"Configured ({len(value)} chars)"
            else:
                status = "❌ Missing"
                details = "Not configured"
            
            status_table.add_row(description, status, details)
        
        # Verificar imports importantes
        imports_to_check = [
            ("langchain_core", "LangChain Core"),
            ("langgraph", "LangGraph"),
            ("rich", "Rich Console"),
            ("psycopg2", "PostgreSQL Driver"),
        ]
        
        for module_name, description in imports_to_check:
            try:
                __import__(module_name)
                status = "✅ Available"
                details = "Import successful"
            except ImportError as e:
                status = "❌ Missing"
                details = f"Import error: {str(e)[:50]}"
            
            status_table.add_row(description, status, details)
        
        self.console.print(status_table)
    
    async def _test_database(self):
        """Testa conexão com banco de dados"""
        self.console.print("\n[cyan]🗄️ Testando banco de dados...[/cyan]")
        
        if not self.use_persistent_storage:
            self.console.print("[yellow]⚠️ Usando storage in-memory, não PostgreSQL[/yellow]")
            return
        
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from langgraph.store.postgres import PostgresStore
            
            database_url = settings.database_url
            if not database_url:
                self.console.print("[red]❌ DATABASE_URL não configurado[/red]")
                return
            
            # Testar checkpointer
            with self.console.status("[bold blue]Testando PostgresSaver...", spinner="dots"):
                async with PostgresSaver.from_conn_string(database_url) as checkpointer:
                    # Tentar operação básica
                    configs = list(checkpointer.list({}))
                    self.console.print(f"[green]✅ PostgresSaver: {len(configs)} configs[/green]")
            
            # Testar store
            with self.console.status("[bold blue]Testando PostgresStore...", spinner="dots"):
                async with PostgresStore.from_conn_string(database_url) as store:
                    # Tentar operação básica
                    test_namespace = ("test", "connection")
                    store.put(test_namespace, "test_key", {"test": True})
                    result = store.get(test_namespace, "test_key")
                    if result and result.value.get("test"):
                        self.console.print("[green]✅ PostgresStore: OK[/green]")
                    else:
                        self.console.print("[yellow]⚠️ PostgresStore: Unexpected result[/yellow]")
                    
                    # Limpar teste
                    store.delete(test_namespace, ["test_key"])
            
        except Exception as e:
            self.console.print(f"[red]❌ Erro no teste de banco: {e}[/red]")
            cli_logger.error(f"Database test failed: {e}")
    
    async def _generate_test_log(self):
        """Gera log de teste para verificar sistema de logging"""
        self.console.print("\n[cyan]📝 Gerando log de teste...[/cyan]")
        
        # Logs em diferentes níveis
        cli_logger.debug("🐛 Este é um log DEBUG")
        cli_logger.info("ℹ️ Este é um log INFO")
        cli_logger.warning("⚠️ Este é um log WARNING")
        cli_logger.error("❌ Este é um log ERROR")
        
        # Verificar arquivo de log
        log_file = settings.logs_dir / "cli.log"
        if log_file.exists():
            size = log_file.stat().st_size
            self.console.print(f"[green]✅ Log file: {log_file} ({size} bytes)[/green]")
        else:
            self.console.print("[yellow]⚠️ Log file não encontrado[/yellow]")
        
        self.console.print("[green]✅ Logs de teste gerados![/green]")
    
    def show_help(self):
        """Mostra ajuda do sistema"""
        self.console.print("\n[bold blue]❓ Ajuda do Metanalyst-Agent CLI[/bold blue]")
        
        help_content = """
## 🔬 Sobre o Sistema

O Metanalyst-Agent é um sistema automatizado de meta-análise que utiliza múltiplos agentes LLM 
especializados para conduzir análises sistemáticas da literatura médica.

## 🚀 Funcionalidades Principais

### 1. Meta-Análise Automatizada
- **Busca Inteligente**: Usa Tavily para encontrar artigos relevantes
- **Extração de Dados**: Processa conteúdo e extrai dados estatísticos
- **Análise Estatística**: Calcula effect sizes, forest plots, etc.
- **Geração de Relatórios**: Cria relatórios estruturados em HTML

### 2. Sistema Multi-Agente
- **Orchestrator**: Coordena todo o processo
- **Researcher**: Busca literatura científica
- **Processor**: Extrai e processa artigos
- **Analyst**: Realiza análises estatísticas
- **Writer**: Gera relatórios finais

### 3. Modos de Debug
- **values**: Estado completo a cada step
- **updates**: Updates por agente/nó
- **debug**: Máximo detalhamento

## 📊 Framework PICO

O sistema extrai automaticamente o framework PICO da sua pergunta:
- **P**opulation: População estudada
- **I**ntervention: Intervenção/exposição
- **C**omparison: Comparação/controle
- **O**utcome: Desfecho/resultado

## ⚙️ Configurações

- **Storage**: PostgreSQL (persistente) ou In-Memory (temporário)
- **Debug**: Controla nível de logging detalhado
- **Recursion Limit**: Limite máximo de iterações

## 🔍 Troubleshooting

1. **Erro de API**: Verifique OPENAI_API_KEY e TAVILY_API_KEY
2. **Erro de DB**: Configure DATABASE_URL para PostgreSQL
3. **Timeout**: Aumente tempo máximo ou reduza máximo de artigos
4. **Loops**: Use modo debug para identificar problemas

## 📞 Suporte

Para dúvidas técnicas, verifique:
1. Logs em `logs/cli.log`
2. Estado do sistema no menu Debug
3. Configurações no menu Settings
        """
        
        help_md = Markdown(help_content)
        self.console.print(help_md)


# Funções CLI principais

@click.command()
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Habilitar modo debug com logs detalhados"
)
@click.option(
    "--storage",
    "-s",
    type=click.Choice(["memory", "postgres"]),
    default="postgres",
    help="Tipo de storage (memory=temporário, postgres=persistente)"
)
@click.option(
    "--query",
    "-q",
    type=str,
    help="Executar meta-análise diretamente com query fornecida"
)
@click.option(
    "--max-articles",
    "-m",
    type=int,
    default=50,
    help="Máximo de artigos para processar"
)
@click.option(
    "--quality-threshold",
    "-t",
    type=float,
    default=0.8,
    help="Threshold de qualidade (0.0-1.0)"
)
@click.option(
    "--max-time",
    type=int,
    default=30,
    help="Tempo máximo em minutos"
)
@click.option(
    "--stream-mode",
    type=click.Choice(["values", "updates", "debug"]),
    default="updates",
    help="Modo de streaming de debug"
)
def cli_main(
    debug: bool,
    storage: str,
    query: Optional[str],
    max_articles: int,
    quality_threshold: float,
    max_time: int,
    stream_mode: str
):
    """
    CLI do Metanalyst-Agent - Sistema Automatizado de Meta-Análise
    
    Execute meta-análises automáticas com agentes LLM especializados.
    """
    
    # Configurar storage
    use_persistent = storage == "postgres"
    
    # Inicializar CLI
    cli = MetanalystCLI(debug=debug, use_persistent_storage=use_persistent)
    
    async def run():
        if query:
            # Modo direto com query fornecida
            cli.print_banner()
            
            cli.console.print(f"\n[green]🚀 Executando meta-análise direta...[/green]")
            cli.console.print(f"[dim]Query: {query}[/dim]")
            
            # Inicializar progress tracker
            cli.progress_tracker = ProgressTracker()
            
            await cli.stream_meta_analysis(
                query=query,
                max_articles=max_articles,
                quality_threshold=quality_threshold,
                max_time_minutes=max_time,
                stream_mode=stream_mode
            )
        else:
            # Modo interativo
            await cli.interactive_session()
    
    # Executar
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        cli.console.print("\n\n[yellow]👋 Operação cancelada. Até mais![/yellow]")
        sys.exit(0)
    except Exception as e:
        cli.console.print(f"\n[red]❌ Erro fatal: {e}[/red]")
        if debug:
            import traceback
            cli.console.print("[dim]Stack trace:[/dim]")
            cli.console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
