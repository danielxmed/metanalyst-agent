"""
Grafo principal do sistema multi-agente de meta-análise.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from ..models.state import MetaAnalysisState
from ..models.config import config
from ..agents import (
    supervisor_agent,
    research_agent,
    processor_agent,
    vectorizer_agent,
    retriever_agent,
    analyst_agent,
    writer_agent,
    reviewer_agent,
    editor_agent
)

def build_meta_analysis_graph(checkpointer=None, store=None):
    """
    Constrói o grafo completo do sistema multi-agente de meta-análise.
    
    Args:
        checkpointer: Instância de checkpointer para persistência de estado
        store: Instância de store para memória de longo prazo
    
    Returns:
        Grafo compilado pronto para execução
    """
    
    # Criar o grafo principal
    builder = StateGraph(MetaAnalysisState)
    
    # Adicionar o supervisor como nó central
    builder.add_node("supervisor", supervisor_agent)
    
    # Adicionar todos os agentes especializados
    builder.add_node("researcher", research_agent)
    builder.add_node("processor", processor_agent)
    builder.add_node("vectorizer", vectorizer_agent)
    builder.add_node("retriever", retriever_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("reviewer", reviewer_agent)
    builder.add_node("editor", editor_agent)
    
    # Definir ponto de entrada
    builder.add_edge(START, "supervisor")
    
    # Todos os agentes retornam ao supervisor após execução
    # Isso permite que o supervisor monitore progresso e decida próximos passos
    for agent_name in [
        "researcher", "processor", "vectorizer", "retriever", 
        "analyst", "writer", "reviewer", "editor"
    ]:
        builder.add_edge(agent_name, "supervisor")
    
    # Compilar grafo com persistência
    if checkpointer and store:
        return builder.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=[]  # Permitir execução completa
        )
    elif checkpointer:
        return builder.compile(
            checkpointer=checkpointer,
            interrupt_before=[]
        )
    else:
        return builder.compile()

def create_default_graph():
    """
    Cria grafo com configuração padrão usando PostgreSQL.
    """
    
    # Configurar persistência PostgreSQL
    DB_URI = config.postgres_url
    
    try:
        # Inicializar checkpointer e store
        checkpointer = PostgresSaver.from_conn_string(DB_URI)
        store = PostgresStore.from_conn_string(DB_URI)
        
        # Construir grafo
        graph = build_meta_analysis_graph(
            checkpointer=checkpointer,
            store=store
        )
        
        return graph, checkpointer, store
        
    except Exception as e:
        # Fallback para versão sem persistência
        print(f"Aviso: Não foi possível conectar ao PostgreSQL: {e}")
        print("Usando versão sem persistência...")
        
        graph = build_meta_analysis_graph()
        return graph, None, None

def create_memory_graph():
    """
    Cria grafo com persistência em memória para desenvolvimento.
    """
    
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    
    # Inicializar em memória
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    # Construir grafo
    graph = build_meta_analysis_graph(
        checkpointer=checkpointer,
        store=store
    )
    
    return graph, checkpointer, store

# Função de conveniência para executar meta-análise
def run_meta_analysis(
    user_query: str,
    thread_id: str = None,
    use_memory: bool = False
):
    """
    Executa uma meta-análise completa baseada na query do usuário.
    
    Args:
        user_query: Solicitação de meta-análise do usuário
        thread_id: ID único para a sessão (gerado automaticamente se None)
        use_memory: Se True, usa persistência em memória
    
    Returns:
        Gerador com os resultados da execução
    """
    
    import uuid
    from datetime import datetime
    
    # Gerar thread_id se não fornecido
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    # Criar grafo apropriado
    if use_memory:
        graph, checkpointer, store = create_memory_graph()
    else:
        graph, checkpointer, store = create_default_graph()
    
    # Configuração da execução
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "metanalysis",
        }
    }
    
    # Estado inicial
    initial_state = {
        "messages": [{
            "role": "user",
            "content": user_query
        }],
        "meta_analysis_id": str(uuid.uuid4()),
        "thread_id": thread_id,
        "current_phase": "initialization",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "pico": {},
        "candidate_urls": [],
        "processed_articles": [],
        "statistical_analysis": None,
        "final_report": None
    }
    
    # Executar grafo
    try:
        for chunk in graph.stream(
            initial_state,
            config,
            stream_mode="values"
        ):
            yield chunk
            
    except Exception as e:
        yield {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().isoformat()
        }

# Função para visualizar o grafo
def visualize_graph():
    """
    Gera visualização do grafo para debug e documentação.
    """
    
    try:
        # Criar grafo simples sem persistência
        graph = build_meta_analysis_graph()
        
        # Tentar gerar imagem do grafo
        img_data = graph.get_graph().draw_mermaid_png()
        
        # Salvar imagem
        with open("metanalysis_graph.png", "wb") as f:
            f.write(img_data)
        
        print("Grafo salvo como 'metanalysis_graph.png'")
        
    except Exception as e:
        print(f"Erro ao visualizar grafo: {e}")
        print("Estrutura do grafo:")
        print("START → supervisor → [researcher, processor, vectorizer, retriever, analyst, writer, reviewer, editor] → supervisor → END")

if __name__ == "__main__":
    # Exemplo de uso
    print("Testando sistema metanalyst-agent...")
    
    # Visualizar grafo
    visualize_graph()
    
    # Exemplo de execução
    example_query = """
    Realize uma meta-análise sobre a eficácia da meditação mindfulness 
    versus terapia cognitivo-comportamental para tratamento de ansiedade 
    em adultos. Inclua forest plot e análise de heterogeneidade.
    """
    
    print(f"\nExemplo de query: {example_query}")
    print("\nPara executar, use:")
    print("for result in run_meta_analysis(query): print(result)")