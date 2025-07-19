"""
Arquivo principal do metanalyst-agent.
Configura e executa o sistema multi-agente de meta-análise.
"""

import logging
import sys
import os
from typing import Dict, Any
import asyncio

# Adicionar o diretório src ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain_core.messages import HumanMessage

from models.state import MetaAnalysisState, create_initial_state
from agents.orchestrator import orchestrator_node
from agents.researcher import researcher_agent
from agents.processor import processor_agent
from utils.config import Config, setup_logging, validate_environment


logger = logging.getLogger(__name__)


def create_metanalyst_graph():
    """
    Cria o grafo LangGraph para o sistema de meta-análise.
    
    Returns:
        Grafo compilado do LangGraph
    """
    logger.info("Criando grafo LangGraph para meta-análise")
    
    # Configurar persistência PostgreSQL
    db_url = Config.DATABASE_URL
    
    try:
        # Inicializar store e checkpointer
        store = PostgresStore.from_conn_string(db_url)
        checkpointer = PostgresSaver.from_conn_string(db_url)
        
        # Criar grafo de estado
        builder = StateGraph(MetaAnalysisState)
        
        # Adicionar nó orquestrador (hub central)
        builder.add_node("orchestrator", orchestrator_node)
        
        # Adicionar agentes especializados (spokes)
        builder.add_node("researcher", researcher_agent)
        builder.add_node("processor", processor_agent)
        
        # TODO: Adicionar outros agentes
        # builder.add_node("retriever", retriever_agent)
        # builder.add_node("analyst", analyst_agent)
        # builder.add_node("writer", writer_agent)
        # builder.add_node("reviewer", reviewer_agent)
        # builder.add_node("editor", editor_agent)
        
        # Definir fluxo - todos começam no orquestrador
        builder.add_edge(START, "orchestrator")
        
        # Todos os agentes retornam ao orquestrador (hub-and-spoke)
        builder.add_edge("researcher", "orchestrator")
        builder.add_edge("processor", "orchestrator")
        # builder.add_edge("retriever", "orchestrator")
        # builder.add_edge("analyst", "orchestrator")
        # builder.add_edge("writer", "orchestrator")
        # builder.add_edge("reviewer", "orchestrator")
        # builder.add_edge("editor", "orchestrator")
        
        # Compilar grafo com persistência
        graph = builder.compile(
            checkpointer=checkpointer,
            store=store
        )
        
        logger.info("Grafo LangGraph criado com sucesso")
        return graph
        
    except Exception as e:
        logger.error(f"Erro ao criar grafo: {e}")
        raise


async def run_metanalysis(user_request: str) -> Dict[str, Any]:
    """
    Executa uma meta-análise completa.
    
    Args:
        user_request: Solicitação do usuário em linguagem natural
        
    Returns:
        Resultado da meta-análise
    """
    logger.info(f"Iniciando meta-análise: {user_request}")
    
    try:
        # Criar grafo
        graph = create_metanalyst_graph()
        
        # Criar estado inicial
        initial_state = create_initial_state(
            user_request=user_request,
            config=Config.get_llm_config()
        )
        
        # Adicionar mensagem inicial
        initial_state["messages"] = [
            HumanMessage(content=user_request)
        ]
        
        # Configuração para execução
        config = {
            "configurable": {
                "thread_id": initial_state["thread_id"],
                "checkpoint_ns": "metanalysis"
            }
        }
        
        logger.info(f"Executando meta-análise com thread_id: {initial_state['thread_id']}")
        
        # Executar grafo
        final_state = await graph.ainvoke(initial_state, config)
        
        # Gerar relatório de resultado
        result = {
            "meta_analysis_id": final_state["meta_analysis_id"],
            "status": "completed" if final_state["current_phase"] == "completed" else "failed",
            "pico": final_state.get("pico", {}),
            "articles_processed": len(final_state.get("processed_articles", [])),
            "chunks_created": final_state.get("chunk_count", 0),
            "execution_time": final_state.get("execution_time", {}),
            "final_report": final_state.get("final_report"),
            "messages": [msg.content for msg in final_state.get("messages", [])],
            "agent_logs": final_state.get("agent_logs", [])
        }
        
        logger.info(f"Meta-análise concluída: {result['status']}")
        return result
        
    except Exception as e:
        logger.error(f"Erro na execução da meta-análise: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "meta_analysis_id": None
        }


def save_report_to_file(result: Dict[str, Any]) -> str:
    """
    Salva o relatório final em arquivo HTML.
    
    Args:
        result: Resultado da meta-análise
        
    Returns:
        Caminho do arquivo salvo
    """
    try:
        import os
        from datetime import datetime
        
        # Criar diretório de outputs
        os.makedirs("outputs", exist_ok=True)
        
        # Nome do arquivo
        meta_analysis_id = result.get("meta_analysis_id", "unknown")
        filename = f"outputs/meta_analysis_report_{meta_analysis_id}.html"
        
        # Criar HTML básico se não há relatório final
        if result.get("final_report"):
            html_content = result["final_report"]
        else:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Meta-análise - {meta_analysis_id}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .pico {{ background: #e8f4fd; padding: 15px; border-radius: 5px; }}
                    .stats {{ background: #f0f8f0; padding: 15px; border-radius: 5px; }}
                    .error {{ background: #ffe6e6; padding: 15px; border-radius: 5px; color: #d00; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Relatório de Meta-análise</h1>
                    <p><strong>ID:</strong> {meta_analysis_id}</p>
                    <p><strong>Status:</strong> {result.get('status', 'unknown')}</p>
                    <p><strong>Gerado em:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>PICO</h2>
                    <div class="pico">
                        <p><strong>População:</strong> {result.get('pico', {}).get('population', 'N/A')}</p>
                        <p><strong>Intervenção:</strong> {result.get('pico', {}).get('intervention', 'N/A')}</p>
                        <p><strong>Comparação:</strong> {result.get('pico', {}).get('comparison', 'N/A')}</p>
                        <p><strong>Desfecho:</strong> {result.get('pico', {}).get('outcome', 'N/A')}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Estatísticas</h2>
                    <div class="stats">
                        <p><strong>Artigos processados:</strong> {result.get('articles_processed', 0)}</p>
                        <p><strong>Chunks criados:</strong> {result.get('chunks_created', 0)}</p>
                        <p><strong>Tempo de execução:</strong> {sum(result.get('execution_time', {}).values()):.2f}s</p>
                    </div>
                </div>
                
                {f'<div class="section error"><h2>Erro</h2><p>{result.get("error")}</p></div>' if result.get('error') else ''}
                
                <div class="section">
                    <h2>Log de Execução</h2>
                    <ul>
                        {chr(10).join(f'<li>{msg}</li>' for msg in result.get('messages', []))}
                    </ul>
                </div>
            </body>
            </html>
            """
        
        # Salvar arquivo
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Relatório salvo em: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Erro ao salvar relatório: {e}")
        return ""


async def main():
    """Função principal."""
    # Configurar logging
    setup_logging()
    
    # Validar ambiente
    if not validate_environment():
        logger.error("Ambiente não configurado corretamente")
        return 1
    
    # Obter solicitação do usuário
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        user_request = input("Digite sua solicitação de meta-análise: ")
    
    if not user_request.strip():
        logger.error("Solicitação não pode estar vazia")
        return 1
    
    try:
        # Executar meta-análise
        result = await run_metanalysis(user_request)
        
        # Salvar relatório
        report_file = save_report_to_file(result)
        
        # Exibir resultado
        print("\n" + "="*60)
        print("🎉 META-ANÁLISE CONCLUÍDA")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"ID: {result.get('meta_analysis_id', 'N/A')}")
        print(f"Artigos processados: {result.get('articles_processed', 0)}")
        print(f"Chunks criados: {result.get('chunks_created', 0)}")
        if report_file:
            print(f"Relatório: {report_file}")
        print("="*60)
        
        return 0 if result['status'] == 'completed' else 1
        
    except KeyboardInterrupt:
        logger.info("Execução interrompida pelo usuário")
        return 1
    except Exception as e:
        logger.error(f"Erro na execução: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)