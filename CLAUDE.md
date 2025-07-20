Sua tarefa é desenvolver o metanalyst-agent a partir deste diretório vazio. Abaixo, voce tem a arquitetura proposta e os guias de desenvolvimento para saber como usar o langgraph. Essa é uma aplicação/sistema AI first. Portanto, antes de implementar uma funcionalidade, voce vai sempre se perguntar: Um LLM poderia fazer isso? Se a resposta for sim, use um LLM - evite usar heurísticas manuais. Por exemplo, para referenciar uma publicacao, para extrair os metadados dela, para fazer as queries de retrieve do database vetorial - usaremos LLMs em todas as etapas do processo.

Use POSTGRES tanto para memória de curto prazo, quanto para memória de longo prazo. Deixe orientacoes e scripts prontos para configurar um banco de dados postgres pronto para o sistema, facilitando minnha vida

The metanalyst-agent is the first open-source project by Nobrega Medtech, focused on automated meta-analysis generation using Python and LangGraph. The project implements a multi-agent system with intelligent orchestration for medical literature research, extraction, analysis, and report generation.

Useful links: https://langchain-ai.github.io/langgraph/reference/ https://docs.tavily.com/documentation/api-reference/endpoint/search https://docs.tavily.com/documentation/api-reference/endpoint/extract

Main Objective
Create an automated system that performs the entire medical meta-analysis process, from literature search to generating final reports with statistical analyses, graphs, and forest plots.

System Architecture
Hub-and-Spoke Architecture with Agents-as-a-Tool
The system follows a "sun" architecture, where a central orchestrator agent invokes specialized agents as tools based on the current state:

                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── RETRIEVER
                │   ORCHESTRATOR  │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── WRITER
                         │
                         │
                     (PROCESSOR combines
                      extraction + vectorization)

    ● = Central Orchestrator (Hub)
    │ = Direct Connections (Agents-as-a-Tool)
Architecture Principles:

Central Hub: Orchestrator maintains global state and decision logic
Agents as Tools: Each agent is a specialized tool
Contextual Decision: At each iteration, the orchestrator analyzes the state and chooses the next agent
Direct Communication: All agents communicate only with the orchestrator
Shared State: The orchestrator maintains and updates the global state
Context Engineering and Agents-as-a-Tool Pattern
Central Orchestrator Philosophy
The Orchestrator Agent works like a conductor who, at each iteration, analyzes the current meta-analysis state and decides which specialized agent should be invoked as a tool. This pattern ensures:

Centralized Control: All decision logic resides in the orchestrator
Single State: A single point of truth for the global state
Flexibility: Ability to repeat or skip steps based on context
Observability: All decisions are transparent and loggable
Agent Details
1. Orchestrator Agent (Central Hub)
Role: Conductor orchestrating the entire meta-analysis symphony
Responsibilities:
Define and refine the PICO of the research
Analyze current state and decide next agent
Manage global state (messages, URLs, data, progress)
Implement retry logic and quality control
Maintain decision history and justifications
Tools: All other agents as tools
Context: Maintains complete meta-analysis context from start to finish
2. Researcher Agent
Tools: Tavily Search API
Responsibilities:
Search scientific literature using specific domains
Generate queries based on PICO
Return list of candidate URLs
Filter results by relevance and quality
3. Processor Agent (Combines Extraction + Vectorization)
Tools: Tavily Extract API, OpenAI Embeddings (text-embedding-3-small), FAISS, GPT-4.1-nano
Responsibilities:
Extract full content from URLs using Tavily Extract
Process markdown to structured JSON using GPT-4.1-nano
Generate Vancouver references
Create objective summaries with statistical data
Chunk scientific publications intelligently (1000 chars, 100 overlap)
Maintain reference tracking for each chunk
Generate vector embeddings with text-embedding-3-small
Store in local FAISS vector store
Manage temporary files and cleanup
Update state with processed URLs and vector store status
4. Retriever Agent
Tools: FAISS, Cosine Similarity
Responsibilities:
Search for relevant information using PICO
Return high-similarity chunks
Maintain reference context
6. Writer Agent
Responsibilities:
Analyze retrieved chunks
Generate structured HTML report
Include methodology and results
Appropriately cite references
7. Reviewer Agent
Responsibilities:
Review report quality
Generate improvement feedback
Suggest additional searches if necessary
Validate compliance with medical standards
8. Analyst Agent
Tools: Matplotlib, Plotly, SciPy, NumPy
Responsibilities:
Statistical analyses (meta-analysis, forest plots)
Generate graphs and tables
Calculate metrics (OR, RR, CI)
Create HTML visualizations
9. Editor Agent
Responsibilities:

Integrate report + analyses
Generate final HTML
Structure final document
Ensure appropriate formatting



## GUIDES ##

# Guia Prático: Memória e Estado Compartilhado em Sistemas Multi-Agentes com LangGraph

## Resumo Executivo

Este guia fornece uma visão abrangente das ferramentas nativas do LangGraph para gerenciar contexto compartilhado, memória de curto prazo e memória de longo prazo em sistemas multi-agentes. O documento serve como base técnica para implementar a arquitetura do **metanalyst-agent**, onde o gerenciamento eficiente do estado entre agentes é crucial para o sucesso do sistema.

## 1. Conceitos Fundamentais

### 1.1 Arquitetura de Estado no LangGraph

No LangGraph, o estado é uma estrutura de dados compartilhada que representa o snapshot atual da aplicação. Cada nó no grafo pode ler e atualizar esse estado, permitindo o fluxo de informações entre agentes.

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Estado básico compartilhado
class SharedState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_pico: str
    research_phase: str
    extracted_data: dict
    vector_store_status: str
```

### 1.2 Tipos de Memória

1. **Memória de Curto Prazo** (Thread-Level)
   - Persiste durante uma conversação
   - Gerenciada via Checkpointers
   - Ideal para contexto da sessão atual

2. **Memória de Longo Prazo** (Cross-Thread)
   - Persiste entre conversações
   - Gerenciada via Stores
   - Ideal para informações permanentes

## 2. Ferramentas Nativas para Memória de Curto Prazo

### 2.1 Checkpointers

O LangGraph implementa uma camada central de persistência através de checkpointers, permitindo recursos comuns à maioria das arquiteturas de agentes, incluindo memória de conversações e atualizações dentro e entre interações do usuário.

#### Opções de Checkpointers:

```python
# 1. In-Memory (Desenvolvimento)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# 2. PostgreSQL (Produção)
from langgraph.checkpoint.postgres import PostgresSaver
DB_URI = "postgresql://user:pass@localhost:5432/dbname"
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# 3. Redis (Alta Performance)
from langgraph.checkpoint.redis import RedisSaver
DB_URI = "redis://localhost:6379"
checkpointer = RedisSaver.from_conn_string(DB_URI)

# 4. MongoDB (Flexibilidade)
from langgraph.checkpoint.mongodb import MongoDBSaver
DB_URI = "mongodb://localhost:27017"
checkpointer = MongoDBSaver.from_conn_string(DB_URI)
```

#### Implementação com Checkpointer:

```python
# Compilar grafo com checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Invocar com thread_id para manter estado
config = {
    "configurable": {
        "thread_id": "metanalysis_001",
        "checkpoint_ns": "research_phase"
    }
}

# O estado persiste entre invocações
result1 = graph.invoke({"messages": ["Define PICO"]}, config)
result2 = graph.invoke({"messages": ["Search literature"]}, config)
```

## 3. Ferramentas Nativas para Memória de Longo Prazo

### 3.1 Stores

Os Stores no LangGraph permitem persistência e compartilhamento de informações entre diferentes threads de execução, sendo essenciais para reter informações sobre entidades (como usuários) através de todas as suas conversações.

#### Configuração de Stores:

```python
# 1. In-Memory Store (Desenvolvimento)
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# 2. PostgreSQL Store (Produção)
from langgraph.store.postgres import PostgresStore
store = PostgresStore.from_conn_string(DB_URI)

# 3. Redis Store (Cache Distribuído)
from langgraph.store.redis import RedisStore
store = RedisStore.from_conn_string(DB_URI)
```

#### Store com Busca Semântica:

```python
from langchain.embeddings import init_embeddings

# Configurar store com embeddings para busca semântica
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["content", "summary", "metadata"]
    }
)

# Compilar grafo com store e checkpointer
graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)
```

### 3.2 Operações com Store

```python
def process_and_store_article(
    state: MetaAnalysisState,
    config: RunnableConfig,
    *,
    store: BaseStore
):
    """Processa e armazena artigo na memória de longo prazo"""
    
    # Namespace para organização
    namespace = ("metanalysis", state["pico_id"], "articles")
    
    # Armazenar artigo processado
    article_id = str(uuid.uuid4())
    store.put(
        namespace,
        article_id,
        {
            "title": state["current_article"]["title"],
            "content": state["current_article"]["content"],
            "extracted_data": state["current_article"]["extracted_data"],
            "vector_chunks": state["current_article"]["chunks"],
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "quality_score": state["current_article"]["quality_score"]
            }
        }
    )
    
    # Buscar artigos similares
    similar_articles = store.search(
        namespace,
        query=state["current_article"]["summary"],
        limit=5
    )
    
    return {"similar_articles": similar_articles}
```

## 4. Estratégias para Comunicação Multi-Agente

### 4.1 Padrão Hub-and-Spoke com Command

Na arquitetura hub-and-spoke, um orquestrador central mantém o estado global e decide qual agente especializado deve ser invocado como ferramenta. Esse padrão garante controle centralizado, estado único e flexibilidade para repetir ou pular etapas baseado no contexto.

```python
from langgraph.types import Command
from typing import Literal

class OrchestratorState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    pico: dict
    research_phase: str
    urls_to_process: List[str]
    processed_urls: List[str]
    vector_store: Any
    analysis_results: dict

def orchestrator_node(state: OrchestratorState) -> Command[
    Literal["researcher", "processor", "retriever", "analyst", "writer", "__end__"]
]:
    """Orquestrador central que decide próximo agente"""
    
    # Analisar estado atual
    phase = state["research_phase"]
    urls_pending = len(state["urls_to_process"])
    
    # Lógica de decisão
    if phase == "search" and urls_pending == 0:
        goto = "researcher"
        update = {"messages": [AIMessage("Iniciando busca por literatura...")]}
    
    elif phase == "search" and urls_pending > 0:
        goto = "processor"
        update = {
            "messages": [AIMessage(f"Processando {urls_pending} artigos...")],
            "research_phase": "extraction"
        }
    
    elif phase == "extraction" and state.get("vector_store"):
        goto = "retriever"
        update = {"research_phase": "analysis"}
    
    elif phase == "analysis" and state.get("analysis_results"):
        goto = "writer"
        update = {"research_phase": "writing"}
    
    elif phase == "writing" and state.get("final_report"):
        goto = "__end__"
        update = {"messages": [AIMessage("Meta-análise concluída!")]}
    
    else:
        goto = "analyst"
        update = {}
    
    return Command(goto=goto, update=update)
```

### 4.2 Handoff Entre Agentes

```python
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState

def create_handoff_tool(*, agent_name: str, description: str):
    """Cria ferramenta de handoff para transferir controle entre agentes"""
    
    @tool(f"transfer_to_{agent_name}", description=description)
    def handoff_tool(
        state: Annotated[OrchestratorState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f"Transferindo para {agent_name}",
            tool_call_id=tool_call_id
        )
        
        return Command(
            goto=agent_name,
            update={
                "messages": state["messages"] + [tool_message],
                "last_agent": agent_name
            },
            graph=Command.PARENT  # Navegar para grafo pai
        )
    
    return handoff_tool
```

## 5. Estado Compartilhado Sugerido para Metanalyst-Agent

Baseado na arquitetura descrita, aqui está uma sugestão de estado compartilhado otimizado:

```python
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from langgraph.graph.message import add_messages

class MetaAnalysisState(TypedDict):
    """Estado compartilhado para o sistema de meta-análise"""
    
    # Identificação e controle
    meta_analysis_id: str
    thread_id: str
    current_phase: Literal[
        "pico_definition", "search", "extraction", 
        "vectorization", "analysis", "writing", "review", "editing"
    ]
    current_agent: Optional[str]
    
    # PICO e pesquisa
    pico: Dict[str, str]  # {P, I, C, O}
    search_queries: List[str]
    search_domains: List[str]
    
    # URLs e processamento
    candidate_urls: List[Dict[str, Any]]  # [{url, title, relevance_score}]
    processing_queue: List[str]
    processed_articles: List[Dict[str, Any]]
    failed_urls: List[Dict[str, str]]  # [{url, error}]
    
    # Extração e vetorização
    extracted_data: List[Dict[str, Any]]  # Dados estruturados
    vector_store_id: Optional[str]
    vector_store_status: Dict[str, Any]
    chunk_count: int
    
    # Análise e resultados
    retrieval_results: List[Dict[str, Any]]
    statistical_analysis: Dict[str, Any]
    forest_plots: List[Dict[str, Any]]
    quality_assessments: Dict[str, float]
    
    # Relatórios e revisões
    draft_report: Optional[str]
    review_feedback: List[Dict[str, str]]
    final_report: Optional[str]
    citations: List[Dict[str, str]]
    
    # Mensagens e logs
    messages: Annotated[List[BaseMessage], add_messages]
    agent_logs: List[Dict[str, Any]]  # Logs detalhados por agente
    
    # Metadados
    created_at: datetime
    updated_at: datetime
    total_articles_processed: int
    execution_time: Dict[str, float]  # Tempo por fase
    
    # Configurações
    config: Dict[str, Any]  # Parâmetros configuráveis
```

## 6. Implementação Prática do Sistema

### 6.1 Configuração do Grafo Principal

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

# Configurar persistência
DB_URI = "postgresql://user:pass@localhost:5432/metanalysis"

with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # Construir grafo
    builder = StateGraph(MetaAnalysisState)
    
    # Adicionar nó orquestrador
    builder.add_node("orchestrator", orchestrator_node)
    
    # Adicionar agentes especializados
    builder.add_node("researcher", researcher_agent)
    builder.add_node("processor", processor_agent)
    builder.add_node("retriever", retriever_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("reviewer", reviewer_agent)
    builder.add_node("editor", editor_agent)
    
    # Definir fluxo
    builder.add_edge(START, "orchestrator")
    
    # Todos os agentes retornam ao orquestrador
    for agent in ["researcher", "processor", "retriever", 
                  "analyst", "writer", "reviewer", "editor"]:
        builder.add_edge(agent, "orchestrator")
    
    # Compilar com persistência
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store
    )
```

### 6.2 Exemplo de Agente Especializado

```python
def processor_agent(
    state: MetaAnalysisState,
    config: RunnableConfig,
    *,
    store: BaseStore
) -> Dict[str, Any]:
    """Agente processador que extrai e vetoriza artigos"""
    
    # Obter próximo URL para processar
    if not state["processing_queue"]:
        return {"messages": [AIMessage("Nenhum artigo para processar")]}
    
    url = state["processing_queue"][0]
    
    try:
        # Extrair conteúdo usando Tavily Extract
        content = extract_with_tavily(url)
        
        # Processar com LLM
        extracted = process_with_llm(content, state["pico"])
        
        # Gerar chunks e embeddings
        chunks = create_chunks(content)
        embeddings = generate_embeddings(chunks)
        
        # Armazenar no store de longo prazo
        namespace = ("metanalysis", state["meta_analysis_id"], "articles")
        article_id = str(uuid.uuid4())
        
        store.put(namespace, article_id, {
            "url": url,
            "content": content,
            "extracted_data": extracted,
            "chunks": chunks,
            "embeddings": embeddings,
            "processed_at": datetime.now().isoformat()
        })
        
        # Atualizar estado
        return {
            "processing_queue": state["processing_queue"][1:],
            "processed_articles": state["processed_articles"] + [{
                "id": article_id,
                "url": url,
                "data": extracted
            }],
            "messages": [AIMessage(f"Artigo processado: {url}")]
        }
        
    except Exception as e:
        return {
            "processing_queue": state["processing_queue"][1:],
            "failed_urls": state["failed_urls"] + [{"url": url, "error": str(e)}],
            "messages": [AIMessage(f"Erro ao processar {url}: {str(e)}")]
        }
```

## 7. Melhores Práticas

### 7.1 Gerenciamento de Estado

1. **Use Reducers para Atualizações Complexas**
   ```python
   from operator import add
   
   class State(TypedDict):
       messages: Annotated[List[BaseMessage], add_messages]
       counters: Annotated[Dict[str, int], lambda x, y: {**x, **y}]
   ```

2. **Namespace Hierárquico para Store**
   ```python
   # Organização recomendada
   ("metanalysis", analysis_id, "articles")
   ("metanalysis", analysis_id, "statistics")
   ("metanalysis", analysis_id, "reports")
   ```

3. **Checkpoints em Momentos Críticos**
   ```python
   # Forçar checkpoint após operações importantes
   graph.update_state(
       config,
       {"checkpoint_reason": "Articles processed"},
       as_node="processor"
   )
   ```

### 7.2 Comunicação Entre Agentes

Os agentes podem compartilhar o histórico completo de seu processo de pensamento (scratchpad) com todos os outros agentes, ou podem ter seu próprio "scratchpad" privado e compartilhar apenas o resultado final.

1. **Mensagens Estruturadas**
   ```python
   # Adicionar contexto às mensagens
   AIMessage(
       content="Análise concluída",
       name="analyst_agent",
       additional_kwargs={
           "phase": "statistical_analysis",
           "metrics": {"effect_size": 0.42}
       }
   )
   ```

2. **Estado Privado vs. Compartilhado**
   ```python
   # Subgrafo com estado privado
   class ProcessorPrivateState(TypedDict):
       internal_buffer: List[str]
       temp_calculations: Dict[str, float]
       
   # Apenas campos compartilhados são propagados
   ```

### 7.3 Otimizações de Performance

1. **Cache de Embeddings**
   ```python
   # Verificar cache antes de gerar embeddings
   cached = store.get(("embeddings_cache",), content_hash)
   if cached:
       return cached.value["embeddings"]
   ```

2. **Processamento em Batch**
   ```python
   # Processar múltiplos documentos em paralelo
   from langgraph.types import Send
   
   return Command(
       goto=[Send("processor", {"url": url}) for url in batch],
       update={"batch_processing": True}
   )
   ```

## 8. Monitoramento e Debug

### 8.1 Instrumentação

```python
def instrumented_agent(agent_func):
    """Decorator para monitorar agentes"""
    def wrapper(state, config, **kwargs):
        start_time = time.time()
        agent_name = agent_func.__name__
        
        try:
            result = agent_func(state, config, **kwargs)
            execution_time = time.time() - start_time
            
            # Log de sucesso
            result["agent_logs"] = state.get("agent_logs", []) + [{
                "agent": agent_name,
                "status": "success",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }]
            
            return result
            
        except Exception as e:
            # Log de erro
            return {
                "agent_logs": state.get("agent_logs", []) + [{
                    "agent": agent_name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }],
                "messages": [AIMessage(f"Erro em {agent_name}: {str(e)}")]
            }
    
    return wrapper
```

### 8.2 Visualização de Estado

```python
# Listar checkpoints para debug
checkpoints = list(checkpointer.list(config))
for checkpoint in checkpoints:
    print(f"Checkpoint: {checkpoint.config}")
    print(f"State: {checkpoint.checkpoint}")
    print("---")

# Buscar memórias específicas
memories = store.search(
    ("metanalysis", analysis_id),
    query="forest plot",
    limit=10
)
```

## 9. Conclusão

O LangGraph fornece controle fino sobre o fluxo e estado das aplicações de agentes através de sua implementação de máquinas de estado e grafos direcionados. As ferramentas nativas apresentadas neste guia - Checkpointers para memória de curto prazo e Stores para memória de longo prazo - formam a base para construir sistemas multi-agentes robustos e escaláveis.

Para o **metanalyst-agent**, a combinação dessas ferramentas permite:
- Manter contexto completo da meta-análise durante todo o processo
- Reutilizar dados processados entre diferentes execuções
- Coordenar múltiplos agentes especializados de forma eficiente
- Implementar recuperação de falhas e retomada de processos
- Escalar para análises complexas com centenas de artigos

A chave do sucesso está em escolher as ferramentas certas para cada tipo de persistência e implementar uma estratégia clara de comunicação entre agentes através do estado compartilhado.

# Guia Prático: Agentes Autônomos Multi-Agentes com LangGraph

## Resumo Executivo

Este guia apresenta como criar **verdadeiros sistemas multi-agentes autônomos** no LangGraph, onde cada agente é uma entidade independente capaz de tomar decisões sobre quais ferramentas usar através de `bind_tools`. Diferente de workflows orquestrados, aqui o supervisor também é um agente ReAct que decide autonomamente quando delegar tarefas para outros agentes especializados.

## 1. Conceitos Fundamentais de Agentes Autônomos

### 1.1 O que é um Agente Autônomo Real

Um agente autônomo no LangGraph é caracterizado por:

1. **Autonomia de Decisão**: O modelo LLM decide quando e quais ferramentas usar
2. **Estado Próprio**: Mantém seu contexto interno e histórico de mensagens
3. **Ferramentas Específicas**: Possui conjunto próprio de ferramentas via `bind_tools`
4. **Capacidade de Handoff**: Pode transferir controle para outros agentes quando necessário

### 1.2 create_react_agent - A Base dos Agentes

O `create_react_agent` é a função principal para criar agentes autônomos:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",  # Modelo LLM
    tools=[...],                                  # Lista de ferramentas
    prompt="You are a specialized agent...",      # System prompt
    name="agent_name"                             # Nome único do agente
)
```

## 2. Implementação de Agentes Autônomos para Metanalyst-Agent

### 2.1 Definindo Ferramentas Especializadas

Cada agente do sistema terá suas próprias ferramentas específicas:

```python
# metanalyst_agent/tools/research_tools.py
from langchain_core.tools import tool
from typing import List, Dict
import os

# Ferramentas do Researcher Agent
@tool
def search_pubmed(query: str, max_results: int = 50) -> List[Dict[str, str]]:
    """Search PubMed for scientific articles based on query"""
    # Implementação real usando API do PubMed
    return [{"title": "...", "url": "...", "abstract": "..."}]

@tool
def search_cochrane(query: str) -> List[Dict[str, str]]:
    """Search Cochrane Library for systematic reviews"""
    # Implementação real
    return [{"title": "...", "url": "...", "type": "systematic_review"}]

@tool
def generate_search_queries(pico: Dict[str, str]) -> List[str]:
    """Generate optimized search queries based on PICO framework"""
    queries = []
    # Lógica para gerar queries baseadas em PICO
    queries.append(f"{pico['P']} AND {pico['I']} AND {pico['C']}")
    queries.append(f"{pico['I']} AND {pico['O']} systematic review")
    return queries

# Ferramentas do Processor Agent
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def extract_article_content(url: str) -> Dict[str, any]:
    """Extract full content from article URL using Tavily Extract"""
    # Usar Tavily Extract API
    return {"content": "...", "metadata": {...}}

@tool
def extract_statistical_data(content: str, pico: Dict[str, str]) -> Dict[str, any]:
    """Extract statistical data relevant to PICO from article content"""
    # Processar com LLM focado em extração
    return {
        "sample_size": 150,
        "effect_size": 0.42,
        "p_value": 0.001,
        "confidence_interval": [0.25, 0.59]
    }

@tool
def generate_vancouver_citation(article_data: Dict[str, str]) -> str:
    """Generate Vancouver style citation for article"""
    # Gerar citação formatada
    return "Authors. Title. Journal. Year;Volume(Issue):Pages."

# Ferramentas do Analyst Agent
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

@tool
def calculate_meta_analysis(studies_data: List[Dict[str, float]]) -> Dict[str, any]:
    """Perform statistical meta-analysis on extracted data"""
    # Implementar cálculos de meta-análise
    effect_sizes = [s['effect_size'] for s in studies_data]
    weights = [s['sample_size'] for s in studies_data]
    
    # Cálculos estatísticos reais
    pooled_effect = np.average(effect_sizes, weights=weights)
    heterogeneity = calculate_i_squared(studies_data)
    
    return {
        "pooled_effect_size": pooled_effect,
        "heterogeneity_i2": heterogeneity,
        "total_participants": sum(weights)
    }

@tool
def create_forest_plot(analysis_data: Dict[str, any]) -> str:
    """Create forest plot visualization for meta-analysis"""
    # Gerar forest plot
    plt.figure(figsize=(10, 8))
    # ... código de visualização
    plot_path = "/tmp/forest_plot.png"
    plt.savefig(plot_path)
    return plot_path
```

### 2.2 Criando Ferramentas de Handoff

As ferramentas de handoff permitem que agentes transfiram controle autonomamente:

```python
# metanalyst_agent/tools/handoff_tools.py
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.graph import MessagesState

def create_handoff_tool(*, agent_name: str, description: str) -> tool:
    """Factory para criar ferramentas de handoff entre agentes"""
    
    tool_name = f"transfer_to_{agent_name}"
    
    @tool(tool_name, description=description)
    def handoff_tool(
        reason: Annotated[str, "Razão detalhada para transferir para este agente"],
        context: Annotated[str, "Contexto e informações relevantes para o próximo agente"],
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Transfere controle para outro agente com contexto"""
        
        tool_message = {
            "role": "tool",
            "content": f"Transferido para {agent_name}. Razão: {reason}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        
        # Adicionar mensagem de contexto para o próximo agente
        context_message = {
            "role": "system",
            "content": f"Contexto do agente anterior: {context}"
        }
        
        return Command(
            goto=agent_name,
            update={
                "messages": state["messages"] + [tool_message, context_message],
                "last_agent": agent_name,
                "handoff_reason": reason
            },
            graph=Command.PARENT
        )
    
    return handoff_tool

# Criar ferramentas de handoff específicas
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transferir para o Researcher Agent quando precisar buscar mais artigos científicos"
)

transfer_to_processor = create_handoff_tool(
    agent_name="processor", 
    description="Transferir para o Processor Agent quando tiver URLs de artigos para processar"
)

transfer_to_retriever = create_handoff_tool(
    agent_name="retriever",
    description="Transferir para o Retriever Agent quando precisar buscar informações específicas do vector store"
)

transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transferir para o Analyst Agent quando tiver dados suficientes para análise estatística"
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer",
    description="Transferir para o Writer Agent quando a análise estiver completa e precisar gerar o relatório"
)
```

### 2.3 Implementando Agentes Autônomos

Cada agente é uma entidade independente com suas próprias ferramentas e lógica de decisão:

```python
# metanalyst_agent/agents/research_agent.py
from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        # Ferramentas próprias
        search_pubmed,
        search_cochrane,
        generate_search_queries,
        # Ferramentas de handoff
        transfer_to_processor,
        transfer_to_analyst,
    ],
    prompt=(
        "Você é um Research Agent especializado em busca de literatura científica.\n\n"
        "RESPONSABILIDADES:\n"
        "- Buscar artigos relevantes usando PubMed e Cochrane\n"
        "- Gerar queries otimizadas baseadas no PICO\n"
        "- Avaliar relevância dos resultados\n"
        "- Transferir URLs relevantes para o Processor Agent\n\n"
        "QUANDO TRANSFERIR:\n"
        "- Use 'transfer_to_processor' quando tiver coletado URLs de artigos relevantes\n"
        "- Use 'transfer_to_analyst' se já houver dados suficientes processados\n\n"
        "IMPORTANTE: Sempre forneça contexto detalhado ao transferir para outro agente."
    ),
    name="researcher"
)

# metanalyst_agent/agents/processor_agent.py
processor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        # Ferramentas próprias
        extract_article_content,
        extract_statistical_data,
        generate_vancouver_citation,
        chunk_and_vectorize,
        # Ferramentas de handoff
        transfer_to_researcher,
        transfer_to_retriever,
        transfer_to_analyst,
    ],
    prompt=(
        "Você é um Processor Agent especializado em extração e processamento de artigos.\n\n"
        "RESPONSABILIDADES:\n"
        "- Extrair conteúdo completo de artigos usando Tavily\n"
        "- Identificar e extrair dados estatísticos relevantes ao PICO\n"
        "- Gerar citações Vancouver\n"
        "- Criar chunks e vetorizar para o vector store\n\n"
        "QUANDO TRANSFERIR:\n"
        "- Use 'transfer_to_researcher' se precisar de mais artigos\n"
        "- Use 'transfer_to_retriever' após processar e vetorizar artigos\n"
        "- Use 'transfer_to_analyst' quando tiver dados estatísticos extraídos\n\n"
        "QUALIDADE: Seja rigoroso na extração de dados estatísticos."
    ),
    name="processor"
)

# metanalyst_agent/agents/analyst_agent.py
analyst_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        # Ferramentas próprias
        calculate_meta_analysis,
        create_forest_plot,
        assess_heterogeneity,
        perform_sensitivity_analysis,
        # Ferramentas de handoff
        transfer_to_researcher,
        transfer_to_writer,
        transfer_to_reviewer,
    ],
    prompt=(
        "Você é um Analyst Agent especializado em análise estatística de meta-análises.\n\n"
        "RESPONSABILIDADES:\n"
        "- Realizar cálculos de meta-análise (effect size, heterogeneidade)\n"
        "- Criar visualizações (forest plots, funnel plots)\n"
        "- Avaliar qualidade dos estudos\n"
        "- Realizar análises de sensibilidade\n\n"
        "QUANDO TRANSFERIR:\n"
        "- Use 'transfer_to_researcher' se precisar de mais estudos\n"
        "- Use 'transfer_to_writer' quando a análise estiver completa\n"
        "- Use 'transfer_to_reviewer' para revisão de qualidade\n\n"
        "RIGOR: Siga as diretrizes PRISMA para meta-análises."
    ),
    name="analyst"
)
```

### 2.4 Implementando o Supervisor como Agente

O supervisor também é um agente ReAct com ferramentas de handoff:

```python
# metanalyst_agent/agents/supervisor_agent.py
from langgraph.prebuilt import create_react_agent

# Criar ferramentas de handoff para todos os agentes
supervisor_tools = [
    create_handoff_tool(
        agent_name="researcher",
        description="Delegar busca de literatura científica para o Research Agent"
    ),
    create_handoff_tool(
        agent_name="processor",
        description="Delegar processamento de artigos para o Processor Agent"
    ),
    create_handoff_tool(
        agent_name="retriever",
        description="Delegar busca em vector store para o Retriever Agent"
    ),
    create_handoff_tool(
        agent_name="analyst",
        description="Delegar análise estatística para o Analyst Agent"
    ),
    create_handoff_tool(
        agent_name="writer",
        description="Delegar geração de relatório para o Writer Agent"
    ),
    create_handoff_tool(
        agent_name="reviewer",
        description="Delegar revisão de qualidade para o Reviewer Agent"
    ),
    create_handoff_tool(
        agent_name="editor",
        description="Delegar edição final para o Editor Agent"
    ),
]

supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=supervisor_tools,
    prompt=(
        "Você é o Supervisor de um sistema de meta-análise automatizada.\n\n"
        "SUA RESPONSABILIDADE:\n"
        "- Entender a solicitação do usuário e definir o PICO\n"
        "- Delegar tarefas para os agentes especializados apropriados\n"
        "- Monitorar o progresso e garantir qualidade\n"
        "- Decidir quando o trabalho está completo\n\n"
        "AGENTES DISPONÍVEIS:\n"
        "- researcher: Busca literatura científica\n"
        "- processor: Extrai e processa artigos\n"
        "- retriever: Busca informações no vector store\n"
        "- analyst: Realiza análises estatísticas\n"
        "- writer: Gera relatórios\n"
        "- reviewer: Revisa qualidade\n"
        "- editor: Edição final\n\n"
        "FLUXO TÍPICO:\n"
        "1. Definir PICO → researcher\n"
        "2. URLs encontradas → processor\n"
        "3. Dados extraídos → analyst\n"
        "4. Análise completa → writer\n"
        "5. Relatório gerado → reviewer\n"
        "6. Revisado → editor\n\n"
        "IMPORTANTE:\n"
        "- Delegue uma tarefa por vez\n"
        "- Forneça contexto claro ao delegar\n"
        "- Não execute trabalho você mesmo, apenas coordene"
    ),
    name="supervisor"
)
```

### 2.5 Construindo o Grafo Multi-Agente

Agora montamos o sistema completo com todos os agentes:

```python
# metanalyst_agent/graph/multi_agent_graph.py
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import TypedDict, List, Dict, Any, Literal

class MetaAnalysisState(MessagesState):
    """Estado compartilhado entre todos os agentes"""
    pico: Dict[str, str]
    current_phase: str
    articles_found: List[Dict[str, str]]
    processed_data: List[Dict[str, Any]]
    vector_store_id: str
    analysis_results: Dict[str, Any]
    draft_report: str
    final_report: str
    last_agent: str

# Construir o grafo
def build_meta_analysis_graph():
    builder = StateGraph(MetaAnalysisState)
    
    # Adicionar supervisor com destinos possíveis
    builder.add_node(
        "supervisor", 
        supervisor_agent,
        destinations=["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor", END]
    )
    
    # Adicionar todos os agentes especializados
    builder.add_node("researcher", research_agent)
    builder.add_node("processor", processor_agent)
    builder.add_node("retriever", retriever_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("reviewer", reviewer_agent)
    builder.add_node("editor", editor_agent)
    
    # Começar sempre pelo supervisor
    builder.add_edge(START, "supervisor")
    
    # Todos os agentes retornam ao supervisor após executar
    for agent in ["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor"]:
        builder.add_edge(agent, "supervisor")
    
    # Compilar com persistência
    return builder.compile(
        checkpointer=checkpointer,
        store=store
    )

# Criar o grafo
meta_analysis_graph = build_meta_analysis_graph()
```

### 2.6 Executando o Sistema Multi-Agente

```python
# metanalyst_agent/main.py
import uuid
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

# Configurar persistência
DB_URI = "postgresql://user:pass@localhost:5432/metanalysis"

with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # Criar grafo com persistência
    graph = build_meta_analysis_graph()
    
    # Configuração única para a meta-análise
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "checkpoint_ns": "metanalysis",
        }
    }
    
    # Executar com query do usuário
    for chunk in graph.stream(
        {
            "messages": [{
                "role": "user",
                "content": (
                    "Realize uma meta-análise sobre a eficácia da meditação mindfulness "
                    "versus terapia cognitivo-comportamental para tratamento de ansiedade "
                    "em adultos. Inclua forest plot e análise de heterogeneidade."
                )
            }]
        },
        config,
        stream_mode="values"
    ):
        # Mostrar progresso
        if chunk.get("messages"):
            last_msg = chunk["messages"][-1]
            print(f"\n[{chunk.get('last_agent', 'system')}]: {last_msg.content[:200]}...")
```

## 3. Padrões Avançados de Agentes Autônomos

### 3.1 Agentes com Memória Semântica

```python
from langgraph.store.base import BaseStore
from langgraph.prebuilt import InjectedStore

@tool
def remember_finding(
    finding: str,
    category: Literal["statistical", "clinical", "methodological"],
    store: Annotated[BaseStore, InjectedStore]
) -> str:
    """Armazena descoberta importante na memória de longo prazo"""
    namespace = ("findings", category)
    finding_id = str(uuid.uuid4())
    
    store.put(namespace, finding_id, {
        "content": finding,
        "timestamp": datetime.now().isoformat(),
        "category": category
    })
    
    return f"Descoberta armazenada com ID: {finding_id}"

# Agente com capacidade de memória
memory_enhanced_analyst = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        calculate_meta_analysis,
        remember_finding,
        search_similar_findings,  # Busca semântica em descobertas anteriores
    ],
    store=store  # Injetar store no agente
)
```

### 3.2 Agentes Hierárquicos

```python
# Sub-supervisor para time de pesquisa
research_team_supervisor = create_react_agent(
    model="openai:gpt-4.1",
    tools=[
        create_handoff_tool(agent_name="pubmed_specialist"),
        create_handoff_tool(agent_name="cochrane_specialist"),
        create_handoff_tool(agent_name="clinical_trials_specialist"),
    ],
    prompt="Você coordena especialistas em diferentes bases de dados..."
)

# Criar sub-grafo para o time
research_team_graph = (
    StateGraph(ResearchTeamState)
    .add_node("team_supervisor", research_team_supervisor)
    .add_node("pubmed_specialist", pubmed_agent)
    .add_node("cochrane_specialist", cochrane_agent)
    .add_node("clinical_trials_specialist", trials_agent)
    .add_edge(START, "team_supervisor")
    .compile()
)

# Integrar no grafo principal
builder.add_node("research_team", research_team_graph)
```

### 3.3 Agentes com Tool Choice Forçado

```python
# Forçar uso de ferramenta específica em situações críticas
def create_safety_agent():
    tools = [check_data_quality, flag_inconsistency, request_human_review]
    
    # Configurar modelo para sempre usar check_data_quality primeiro
    configured_model = model.bind_tools(
        tools,
        tool_choice={"type": "tool", "name": "check_data_quality"}
    )
    
    return create_react_agent(
        model=configured_model,
        tools=tools,
        prompt="Você é responsável por garantir qualidade dos dados..."
    )
```

## 4. Melhores Práticas para Sistemas Multi-Agentes

### 4.1 Design de Prompts para Agentes

1. **Seja Específico sobre Responsabilidades**
   ```python
   prompt = (
       "VOCÊ É: [Identidade clara]\n"
       "RESPONSABILIDADES: [Lista específica]\n"
       "NÃO FAÇA: [Limitações claras]\n"
       "QUANDO TRANSFERIR: [Condições específicas]\n"
   )
   ```

2. **Forneça Contexto de Handoff**
   ```python
   "Ao transferir, sempre inclua:\n"
   "- Razão da transferência\n"
   "- O que já foi feito\n"
   "- O que precisa ser feito\n"
   "- Dados relevantes processados"
   ```

### 4.2 Gestão de Estado entre Agentes

```python
class AgentTransition(TypedDict):
    """Informações de transição entre agentes"""
    from_agent: str
    to_agent: str
    reason: str
    context: Dict[str, Any]
    timestamp: str
    
# Rastrear transições
def track_transition(state: MetaAnalysisState) -> MetaAnalysisState:
    transitions = state.get("agent_transitions", [])
    transitions.append({
        "from_agent": state["last_agent"],
        "to_agent": state["current_agent"],
        "timestamp": datetime.now().isoformat()
    })
    state["agent_transitions"] = transitions
    return state
```

### 4.3 Tratamento de Erros e Recuperação

```python
@tool(return_direct=True)
def emergency_supervisor_return(
    error_description: str,
    attempted_action: str,
    state: Annotated[MetaAnalysisState, InjectedState]
) -> Command:
    """Retorna ao supervisor em caso de erro crítico"""
    error_msg = {
        "role": "system",
        "content": f"ERRO em {state['last_agent']}: {error_description}"
    }
    
    return Command(
        goto="supervisor",
        update={
            "messages": state["messages"] + [error_msg],
            "error_flag": True,
            "error_details": {
                "agent": state["last_agent"],
                "error": error_description,
                "action": attempted_action
            }
        },
        graph=Command.PARENT
    )
```

## 5. Exemplo Completo: Fluxo de Execução

```python
# Exemplo de interação real entre agentes autônomos

# 1. Usuário faz solicitação
user_request = "Meta-análise sobre meditação vs CBT para ansiedade"

# 2. Supervisor recebe e decide autonomamente
# Output: "Vou delegar para o researcher a busca inicial..."
# Usa tool: transfer_to_researcher(reason="Busca inicial", context="PICO: ...")

# 3. Researcher executa autonomamente
# - Usa: generate_search_queries(pico)
# - Usa: search_pubmed(query) 
# - Usa: search_cochrane(query)
# - Decide: "Encontrei 47 artigos relevantes"
# - Usa: transfer_to_processor(reason="URLs coletadas", context="47 artigos...")

# 4. Processor trabalha autonomamente
# - Loop através dos artigos
# - Usa: extract_article_content(url)
# - Usa: extract_statistical_data(content)
# - Usa: chunk_and_vectorize(content)
# - Decide: "25 artigos com dados estatísticos válidos"
# - Usa: transfer_to_analyst(reason="Dados prontos", context="25 estudos...")

# 5. Analyst realiza análises
# - Usa: calculate_meta_analysis(studies_data)
# - Usa: create_forest_plot(analysis_data)
# - Usa: assess_heterogeneity(data)
# - Decide: "Análise completa, I²=42%"
# - Usa: transfer_to_writer(reason="Análise pronta", context="Resultados...")

# Continue até completar...
```

## 6. Conclusão

A diferença fundamental entre um workflow orquestrado e um sistema multi-agente autônomo está na **capacidade de decisão independente** de cada agente. Com `create_react_agent` e `bind_tools`, cada agente:

1. Avalia o contexto atual
2. Decide quais ferramentas usar
3. Executa ações autonomamente
4. Decide quando transferir controle

Isso cria um sistema verdadeiramente distribuído e inteligente, onde a complexidade emerge da interação entre agentes especializados, não de uma lógica central pré-definida.

Para o **metanalyst-agent**, essa arquitetura permite:
- Escalabilidade: Adicionar novos agentes sem modificar os existentes
- Especialização: Cada agente domina seu domínio específico
- Resiliência: Falhas localizadas não comprometem o sistema
- Adaptabilidade: Agentes decidem dinamicamente o melhor caminho

A chave está em dar aos agentes as ferramentas certas e prompts bem elaborados, deixando que o LLM tome decisões inteligentes sobre quando e como usá-las.


# Guia Prático: Controle de Iterações e Prevenção de Loops em Sistemas Multi-Agentes

## Resumo Executivo

Este guia apresenta estratégias e padrões para implementar sistemas multi-agentes que podem iterar sobre si mesmos de forma controlada, evitando loops infinitos. Focamos em técnicas nativas do LangGraph como `recursion_limit`, conditional edges, state tracking e graceful termination para o **metanalyst-agent**.

## 1. Conceitos Fundamentais

### 1.1 Tipos de Loops em Sistemas Multi-Agentes

1. **Self-Loops**: Um agente chama a si mesmo repetidamente
2. **Cyclic Loops**: Agente A → Agente B → Agente A
3. **Conditional Loops**: Loops que continuam até uma condição ser satisfeita
4. **Retry Loops**: Tentativas repetidas após falhas

### 1.2 Mecanismos de Controle do LangGraph

- **recursion_limit**: Limite máximo de super-steps
- **Conditional Edges**: Roteamento baseado em condições
- **State Tracking**: Contadores e flags no estado
- **RemainingSteps**: Terminação graceful antes do limite

## 2. Configurando Recursion Limit

### 2.1 Configuração Básica

O `recursion_limit` define o número máximo de super-steps (ciclos completos do grafo):

```python
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

# Para agentes ReAct: cada iteração = 2 steps (LLM + Tool)
max_iterations = 10
recursion_limit = 2 * max_iterations + 1  # 21 steps

try:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Analise todos os artigos"}]},
        {"recursion_limit": recursion_limit}  # Configuração runtime
    )
except GraphRecursionError:
    print("Limite de recursão atingido - processamento interrompido")
```

### 2.2 Configuração Persistente

```python
# Configurar limite permanente no agente
agent_with_limit = agent.with_config(recursion_limit=50)

# Ou no grafo compilado
graph = builder.compile(
    checkpointer=checkpointer,
    recursion_limit=100  # Limite padrão para todas as execuções
)
```

## 3. Implementando State Tracking para Controle de Iterações

### 3.1 Estado com Contadores

```python
# metanalyst_agent/state/iteration_state.py
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph.message import add_messages

class IterativeAgentState(TypedDict):
    """Estado com controle de iterações"""
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Contadores de iteração
    global_iterations: int  # Iterações totais do sistema
    agent_iterations: Dict[str, int]  # Iterações por agente
    
    # Controle de qualidade
    quality_score: float  # Score atual da análise
    quality_threshold: float  # Threshold mínimo aceitável
    improvement_rate: float  # Taxa de melhoria entre iterações
    
    # Flags de controle
    max_retries_reached: bool
    quality_satisfied: bool
    force_stop: bool
    
    # Dados de iteração
    iteration_history: List[Dict[str, Any]]
    failed_attempts: List[Dict[str, str]]
```

### 3.2 Agente com Auto-Iteração Controlada

```python
# metanalyst_agent/agents/iterative_analyst.py
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

def create_iterative_analyst_agent():
    """Cria agente que pode iterar sobre suas próprias análises"""
    
    return create_react_agent(
        model="openai:gpt-4.1",
        tools=[
            perform_meta_analysis,
            assess_analysis_quality,
            refine_analysis,
            request_more_data,
        ],
        prompt=(
            "Você é um Analyst Agent que refina análises iterativamente.\n\n"
            "PROCESSO ITERATIVO:\n"
            "1. Realizar análise inicial\n"
            "2. Avaliar qualidade (heterogeneidade, tamanho amostral, etc.)\n"
            "3. Se qualidade < threshold, refinar ou buscar mais dados\n"
            "4. Repetir até qualidade satisfatória ou limite atingido\n\n"
            "LIMITES:\n"
            "- Máximo 5 iterações de refinamento\n"
            "- Parar se melhoria < 5% entre iterações\n"
            "- Sempre reportar status ao supervisor"
        ),
        name="iterative_analyst"
    )

# Nó do agente com controle de iteração
def iterative_analyst_node(state: IterativeAgentState) -> Command:
    """Nó que implementa lógica de iteração"""
    
    # Verificar contador de iterações
    agent_name = "iterative_analyst"
    current_iterations = state.get("agent_iterations", {}).get(agent_name, 0)
    
    # Condições de parada
    if current_iterations >= 5:
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage("Limite de iterações atingido (5)")],
                "max_retries_reached": True
            }
        )
    
    # Executar agente
    result = iterative_analyst_agent.invoke(state)
    
    # Extrair métricas de qualidade da resposta
    quality_score = extract_quality_score(result["messages"][-1])
    previous_score = state.get("quality_score", 0)
    improvement = quality_score - previous_score
    
    # Atualizar estado
    new_iterations = current_iterations + 1
    agent_iterations = state.get("agent_iterations", {})
    agent_iterations[agent_name] = new_iterations
    
    iteration_record = {
        "iteration": new_iterations,
        "quality_score": quality_score,
        "improvement": improvement,
        "timestamp": datetime.now().isoformat()
    }
    
    update = {
        "messages": result["messages"],
        "agent_iterations": agent_iterations,
        "quality_score": quality_score,
        "improvement_rate": improvement,
        "iteration_history": state.get("iteration_history", []) + [iteration_record]
    }
    
    # Decidir próximo passo baseado em qualidade
    if quality_score >= state.get("quality_threshold", 0.8):
        update["quality_satisfied"] = True
        goto = "supervisor"
    elif improvement < 0.05:  # Melhoria menor que 5%
        update["messages"].append(
            AIMessage("Melhoria insuficiente entre iterações")
        )
        goto = "supervisor"
    else:
        # Continuar iterando (self-loop)
        goto = "iterative_analyst"
    
    return Command(goto=goto, update=update)
```

## 4. Conditional Edges para Prevenção de Loops

### 4.1 Função de Roteamento com Múltiplas Condições

```python
# metanalyst_agent/routing/conditional_routing.py
from typing import Literal

def decide_analyst_next_step(
    state: IterativeAgentState
) -> Literal["iterative_analyst", "researcher", "supervisor", END]:
    """Decide próximo passo do analyst com múltiplas condições"""
    
    # Verificar flags de parada
    if state.get("force_stop", False):
        return END
    
    if state.get("quality_satisfied", False):
        return "supervisor"
    
    # Verificar iterações
    analyst_iterations = state.get("agent_iterations", {}).get("iterative_analyst", 0)
    if analyst_iterations >= 5:
        return "supervisor"
    
    # Verificar taxa de melhoria
    if state.get("improvement_rate", 1.0) < 0.05:
        # Pouca melhoria - tentar buscar mais dados
        if analyst_iterations > 2:
            return "researcher"  # Pedir mais artigos
        else:
            return "supervisor"  # Reportar problema
    
    # Verificar qualidade absoluta
    quality = state.get("quality_score", 0)
    if quality < 0.5 and analyst_iterations < 3:
        return "iterative_analyst"  # Continuar tentando
    elif quality < 0.5:
        return "researcher"  # Qualidade muito baixa, precisa mais dados
    
    # Default: continuar iterando
    return "iterative_analyst"

# Adicionar ao grafo
builder.add_conditional_edges(
    "iterative_analyst",
    decide_analyst_next_step,
    {
        "iterative_analyst": "iterative_analyst",  # Self-loop
        "researcher": "researcher",
        "supervisor": "supervisor", 
        END: END
    }
)
```

### 4.2 Roteamento para Múltiplos Destinos

```python
def route_processor_results(
    state: IterativeAgentState
) -> Sequence[str]:
    """Roteia para múltiplos agentes baseado em condições"""
    
    processed_articles = len(state.get("processed_articles", []))
    failed_articles = len(state.get("failed_urls", []))
    
    destinations = []
    
    # Sempre reportar ao supervisor
    destinations.append("supervisor")
    
    # Se muitas falhas, também notificar reviewer
    if failed_articles > 5:
        destinations.append("reviewer")
    
    # Se processamento completo, iniciar análise
    if processed_articles >= state.get("minimum_articles", 10):
        destinations.append("analyst")
    
    return destinations

# Permite fanout para múltiplos agentes
builder.add_conditional_edges(
    "processor",
    route_processor_results
)
```

## 5. Implementando Retry Patterns

### 5.1 Retry com Backoff Exponencial

```python
# metanalyst_agent/agents/resilient_processor.py
from langgraph.pregel.retry import RetryPolicy
import random

class ProcessorState(IterativeAgentState):
    retry_counts: Dict[str, int]
    backoff_delays: Dict[str, float]

def processor_with_retry_node(state: ProcessorState) -> Command:
    """Processador com retry automático e backoff"""
    
    # Pegar próximo artigo para processar
    if not state.get("processing_queue"):
        return Command(goto="supervisor", update={"messages": [AIMessage("Fila vazia")]})
    
    url = state["processing_queue"][0]
    retry_count = state.get("retry_counts", {}).get(url, 0)
    
    # Verificar limite de retry
    if retry_count >= 3:
        failed_urls = state.get("failed_urls", [])
        failed_urls.append({
            "url": url,
            "error": "Max retries exceeded",
            "attempts": retry_count
        })
        
        return Command(
            goto="processor",  # Continuar com próximo
            update={
                "processing_queue": state["processing_queue"][1:],
                "failed_urls": failed_urls,
                "messages": [AIMessage(f"Falha permanente em {url}")]
            }
        )
    
    # Aplicar backoff se não for primeira tentativa
    if retry_count > 0:
        delay = min(60, (2 ** retry_count) + random.uniform(0, 1))
        time.sleep(delay)
    
    try:
        # Tentar processar
        result = extract_article_content(url)
        
        # Sucesso - remover da fila e resetar contadores
        retry_counts = state.get("retry_counts", {})
        retry_counts.pop(url, None)
        
        return Command(
            goto="processor",
            update={
                "processing_queue": state["processing_queue"][1:],
                "processed_articles": state.get("processed_articles", []) + [result],
                "retry_counts": retry_counts,
                "messages": [AIMessage(f"Processado: {url}")]
            }
        )
        
    except Exception as e:
        # Falha - incrementar contador e manter na fila
        retry_counts = state.get("retry_counts", {})
        retry_counts[url] = retry_count + 1
        
        return Command(
            goto="processor",  # Self-loop para retry
            update={
                "retry_counts": retry_counts,
                "messages": [AIMessage(f"Erro em {url}: {str(e)}. Tentativa {retry_count + 1}/3")]
            }
        )

# Adicionar com política de retry
builder.add_node(
    "processor",
    processor_with_retry_node,
    retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2)
)
```

### 5.2 Circuit Breaker Pattern

```python
class CircuitBreakerState(TypedDict):
    circuit_status: Dict[str, Literal["closed", "open", "half_open"]]
    failure_counts: Dict[str, int]
    last_failure_time: Dict[str, float]
    success_counts: Dict[str, int]

def circuit_breaker_wrapper(
    agent_name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """Wrapper que implementa circuit breaker para agentes"""
    
    def check_circuit(state: CircuitBreakerState) -> bool:
        status = state.get("circuit_status", {}).get(agent_name, "closed")
        
        if status == "open":
            # Verificar se é hora de tentar novamente
            last_failure = state.get("last_failure_time", {}).get(agent_name, 0)
            if time.time() - last_failure > recovery_timeout:
                # Tentar half-open
                circuit_status = state.get("circuit_status", {})
                circuit_status[agent_name] = "half_open"
                return True
            return False
            
        return True  # closed ou half_open
    
    def handle_success(state: CircuitBreakerState) -> Dict:
        circuit_status = state.get("circuit_status", {})
        circuit_status[agent_name] = "closed"
        
        failure_counts = state.get("failure_counts", {})
        failure_counts[agent_name] = 0
        
        return {
            "circuit_status": circuit_status,
            "failure_counts": failure_counts
        }
    
    def handle_failure(state: CircuitBreakerState) -> Dict:
        failure_counts = state.get("failure_counts", {})
        failures = failure_counts.get(agent_name, 0) + 1
        failure_counts[agent_name] = failures
        
        circuit_status = state.get("circuit_status", {})
        if failures >= failure_threshold:
            circuit_status[agent_name] = "open"
        
        last_failure_time = state.get("last_failure_time", {})
        last_failure_time[agent_name] = time.time()
        
        return {
            "circuit_status": circuit_status,
            "failure_counts": failure_counts,
            "last_failure_time": last_failure_time
        }
    
    return check_circuit, handle_success, handle_failure
```

## 6. Graceful Termination com RemainingSteps

### 6.1 Implementação com RemainingSteps

```python
# metanalyst_agent/state/graceful_state.py
from langgraph.managed.is_last_step import RemainingSteps

class GracefulState(IterativeAgentState):
    remaining_steps: RemainingSteps
    termination_summary: str

def graceful_analyst_node(state: GracefulState) -> Command:
    """Analyst que termina graciosamente antes do limite"""
    
    # Verificar steps restantes
    remaining = state["remaining_steps"]
    
    # Se poucos steps restantes, preparar para terminar
    if remaining <= 3:
        # Gerar resumo do que foi feito
        summary = generate_analysis_summary(state)
        
        return Command(
            goto=END,
            update={
                "termination_summary": summary,
                "messages": [AIMessage(
                    f"Terminando graciosamente. Steps restantes: {remaining}\n"
                    f"Resumo: {summary}"
                )]
            }
        )
    
    # Verificar se vale a pena continuar
    estimated_steps_needed = estimate_remaining_work(state)
    
    if estimated_steps_needed > remaining - 2:
        # Não há steps suficientes - melhor parar agora
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(
                    f"Trabalho estimado ({estimated_steps_needed} steps) "
                    f"excede limite disponível ({remaining} steps). "
                    "Retornando resultados parciais."
                )]
            }
        )
    
    # Continuar processamento normal
    result = analyst_agent.invoke(state)
    
    return Command(
        goto="iterative_analyst" if needs_iteration(result) else "supervisor",
        update=result
    )
```

### 6.2 Estratégia de Checkpoint Progressivo

```python
def progressive_checkpoint_strategy(state: GracefulState) -> Command:
    """Salva progresso em intervalos para recuperação"""
    
    remaining = state["remaining_steps"]
    total_processed = len(state.get("processed_articles", []))
    
    # Checkpoints mais frequentes quando próximo do limite
    if remaining < 10 or total_processed % 5 == 0:
        # Salvar estado intermediário
        checkpoint_data = {
            "processed_articles": state.get("processed_articles", []),
            "partial_analysis": state.get("analysis_results", {}),
            "quality_metrics": state.get("quality_score", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Salvar no store para recuperação
        store.put(
            ("checkpoints", state["meta_analysis_id"]),
            f"checkpoint_{total_processed}",
            checkpoint_data
        )
    
    # Decisão de continuar ou consolidar
    if remaining < 5:
        return Command(
            goto="consolidator",
            update={"messages": [AIMessage("Consolidando resultados parciais")]}
        )
    
    return Command(goto="processor")
```

## 7. Padrões Complexos de Iteração

### 7.1 Iteração Hierárquica

```python
# Supervisor que monitora sub-loops
def hierarchical_supervisor_node(state: IterativeAgentState) -> Command:
    """Supervisor que gerencia loops em múltiplos níveis"""
    
    # Verificar loops de nível superior
    global_iterations = state.get("global_iterations", 0)
    
    if global_iterations >= 3:
        return Command(
            goto=END,
            update={"messages": [AIMessage("Limite global de iterações atingido")]}
        )
    
    # Verificar estado dos sub-agentes
    agent_iterations = state.get("agent_iterations", {})
    stuck_agents = [
        agent for agent, count in agent_iterations.items()
        if count >= 5
    ]
    
    if stuck_agents:
        # Intervir em agentes travados
        return Command(
            goto="reviewer",
            update={
                "messages": [AIMessage(f"Agentes travados: {stuck_agents}")],
                "intervention_needed": True
            }
        )
    
    # Verificar progresso geral
    progress_rate = calculate_progress_rate(state)
    
    if progress_rate < 0.1:  # Progresso muito lento
        # Mudar estratégia
        return Command(
            goto="strategy_planner",
            update={"messages": [AIMessage("Progresso lento detectado")]}
        )
    
    # Continuar operação normal
    return Command(
        goto=decide_next_agent(state),
        update={"global_iterations": global_iterations + 1}
    )
```

### 7.2 Adaptive Iteration Control

```python
class AdaptiveIterationAgent:
    """Agente que ajusta seus limites dinamicamente"""
    
    def __init__(self):
        self.base_limit = 5
        self.performance_history = []
    
    def calculate_dynamic_limit(self, state: IterativeAgentState) -> int:
        """Calcula limite baseado em performance histórica"""
        
        # Fatores a considerar
        complexity = estimate_task_complexity(state)
        time_remaining = state.get("deadline_seconds", float('inf')) - time.time()
        quality_delta = state.get("quality_score", 0) - state.get("quality_threshold", 0.8)
        
        # Ajustar limite baseado em fatores
        limit = self.base_limit
        
        if complexity > 0.8:  # Tarefa complexa
            limit += 3
        
        if quality_delta < -0.2:  # Longe do objetivo
            limit += 2
        
        if time_remaining < 300:  # Menos de 5 minutos
            limit = min(limit, 2)
        
        # Aprender com histórico
        if self.performance_history:
            avg_iterations_needed = np.mean([h["iterations"] for h in self.performance_history])
            limit = int(avg_iterations_needed * 1.2)  # 20% de margem
        
        return max(1, min(limit, 10))  # Entre 1 e 10
    
    def update_performance_history(self, iterations_used: int, quality_achieved: float):
        """Atualiza histórico para aprendizado"""
        self.performance_history.append({
            "iterations": iterations_used,
            "quality": quality_achieved,
            "timestamp": datetime.now()
        })
        
        # Manter apenas últimas 20 execuções
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
```

## 8. Implementação Completa para Metanalyst-Agent

### 8.1 Estado com Controle Completo de Iteração

```python
# metanalyst_agent/state/complete_iteration_state.py
from typing import TypedDict, Dict, List, Any, Literal, Optional
from datetime import datetime
from langgraph.managed.is_last_step import RemainingSteps

class MetaAnalysisIterationState(TypedDict):
    """Estado completo com controle de iteração"""
    
    # Estado base
    messages: Annotated[List[BaseMessage], add_messages]
    meta_analysis_id: str
    pico: Dict[str, str]
    
    # Controle de iteração global
    remaining_steps: RemainingSteps
    global_iterations: int
    max_global_iterations: int
    
    # Controle por agente
    agent_iterations: Dict[str, int]
    agent_limits: Dict[str, int]
    agent_performance: Dict[str, List[float]]
    
    # Controle de qualidade
    quality_scores: Dict[str, float]
    quality_thresholds: Dict[str, float]
    improvement_rates: Dict[str, float]
    
    # Circuit breaker
    circuit_status: Dict[str, Literal["closed", "open", "half_open"]]
    failure_counts: Dict[str, int]
    last_failure_time: Dict[str, float]
    
    # Retry control
    retry_counts: Dict[str, int]
    retry_delays: Dict[str, float]
    
    # Checkpoints
    checkpoints: List[Dict[str, Any]]
    last_checkpoint_time: float
    
    # Flags
    force_stop: bool
    quality_satisfied: bool
    deadline_reached: bool
    
    # Métricas
    total_articles_processed: int
    total_failures: int
    average_quality: float
    
    # Histórico
    iteration_timeline: List[Dict[str, Any]]
```

### 8.2 Grafo com Controle Completo

```python
# metanalyst_agent/graph/iteration_controlled_graph.py
from langgraph.graph import StateGraph, START, END

def build_iteration_controlled_graph():
    """Constrói grafo com controle completo de iteração"""
    
    builder = StateGraph(MetaAnalysisIterationState)
    
    # Supervisor com monitoramento global
    builder.add_node("supervisor", hierarchical_supervisor_node)
    
    # Agentes com auto-iteração
    builder.add_node("iterative_researcher", iterative_researcher_node)
    builder.add_node("iterative_processor", processor_with_retry_node)
    builder.add_node("iterative_analyst", graceful_analyst_node)
    
    # Nós de controle
    builder.add_node("iteration_controller", iteration_controller_node)
    builder.add_node("quality_assessor", quality_assessor_node)
    builder.add_node("checkpoint_manager", checkpoint_manager_node)
    
    # Início sempre pelo controller
    builder.add_edge(START, "iteration_controller")
    
    # Controller decide próximo passo
    builder.add_conditional_edges(
        "iteration_controller",
        route_based_on_state,
        {
            "supervisor": "supervisor",
            "continue": "supervisor",
            "checkpoint": "checkpoint_manager",
            "terminate": END
        }
    )
    
    # Supervisor roteia para agentes
    builder.add_conditional_edges(
        "supervisor",
        supervisor_routing_logic,
        {
            "researcher": "iterative_researcher",
            "processor": "iterative_processor",
            "analyst": "iterative_analyst",
            "quality_check": "quality_assessor",
            END: END
        }
    )
    
    # Agentes podem auto-iterar ou retornar
    for agent in ["iterative_researcher", "iterative_processor", "iterative_analyst"]:
        builder.add_conditional_edges(
            agent,
            agent_self_loop_decision,
            {
                agent: agent,  # Self-loop
                "supervisor": "supervisor",
                "quality_assessor": "quality_assessor"
            }
        )
    
    # Quality assessor sempre volta ao controller
    builder.add_edge("quality_assessor", "iteration_controller")
    
    # Checkpoint manager
    builder.add_conditional_edges(
        "checkpoint_manager",
        lambda s: "continue" if s["remaining_steps"] > 5 else "terminate",
        {
            "continue": "supervisor",
            "terminate": END
        }
    )
    
    return builder.compile(
        checkpointer=checkpointer,
        store=store
    )

# Funções de roteamento
def route_based_on_state(state: MetaAnalysisIterationState) -> str:
    """Decisão principal de roteamento"""
    
    if state.get("force_stop", False):
        return "terminate"
    
    if state["remaining_steps"] <= 2:
        return "checkpoint"
    
    if should_checkpoint(state):
        return "checkpoint"
    
    return "continue"

def agent_self_loop_decision(state: MetaAnalysisIterationState) -> str:
    """Decide se agente deve iterar sobre si mesmo"""
    
    # Identificar agente atual
    last_message = state["messages"][-1]
    agent_name = extract_agent_name(last_message)
    
    # Verificar limites
    iterations = state["agent_iterations"].get(agent_name, 0)
    limit = state["agent_limits"].get(agent_name, 5)
    
    if iterations >= limit:
        return "supervisor"
    
    # Verificar qualidade
    quality = state["quality_scores"].get(agent_name, 0)
    threshold = state["quality_thresholds"].get(agent_name, 0.8)
    
    if quality >= threshold:
        return "quality_assessor"
    
    # Verificar melhoria
    improvement = state["improvement_rates"].get(agent_name, 1.0)
    if improvement < 0.05 and iterations > 2:
        return "supervisor"
    
    # Continuar iterando
    return agent_name
```

### 8.3 Exemplo de Execução

```python
# metanalyst_agent/main_iteration.py
import asyncio
from datetime import datetime, timedelta

async def run_meta_analysis_with_iteration_control(
    user_query: str,
    max_time_minutes: int = 30,
    quality_target: float = 0.85
):
    """Executa meta-análise com controle completo de iteração"""
    
    # Configurar estado inicial
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "meta_analysis_id": str(uuid.uuid4()),
        "global_iterations": 0,
        "max_global_iterations": 10,
        "agent_iterations": {},
        "agent_limits": {
            "iterative_researcher": 5,
            "iterative_processor": 10,  # Mais tentativas para processar URLs
            "iterative_analyst": 7
        },
        "quality_thresholds": {
            "iterative_researcher": 0.7,  # 70% dos artigos relevantes
            "iterative_processor": 0.8,   # 80% de sucesso na extração
            "iterative_analyst": quality_target
        },
        "circuit_status": {},
        "retry_counts": {},
        "checkpoints": [],
        "force_stop": False,
        "deadline": datetime.now() + timedelta(minutes=max_time_minutes)
    }
    
    # Configurar recursion limit baseado no tempo disponível
    estimated_steps = estimate_required_steps(user_query)
    recursion_limit = min(estimated_steps * 2, 200)  # Max 200 steps
    
    # Criar grafo
    graph = build_iteration_controlled_graph()
    
    # Configuração de execução
    config = {
        "recursion_limit": recursion_limit,
        "configurable": {
            "thread_id": initial_state["meta_analysis_id"],
            "checkpoint_ns": "meta_analysis",
            "user_id": "system"
        }
    }
    
    # Stream assíncrono com timeout
    try:
        async with asyncio.timeout(max_time_minutes * 60):
            async for chunk in graph.astream(
                initial_state,
                config,
                stream_mode="values"
            ):
                # Processar updates
                if chunk.get("messages"):
                    last_msg = chunk["messages"][-1]
                    print(f"[{datetime.now()}] {last_msg.content[:100]}...")
                
                # Verificar métricas
                if chunk.get("quality_scores"):
                    print(f"Quality scores: {chunk['quality_scores']}")
                
                # Verificar se deve parar
                if chunk.get("quality_satisfied"):
                    print("Qualidade satisfatória atingida!")
                    
                if chunk.get("remaining_steps", 100) < 10:
                    print(f"Atenção: Apenas {chunk['remaining_steps']} steps restantes")
    
    except asyncio.TimeoutError:
        print(f"Timeout após {max_time_minutes} minutos")
        # Recuperar último checkpoint
        final_state = await recover_from_checkpoint(initial_state["meta_analysis_id"])
    
    except GraphRecursionError:
        print("Limite de recursão atingido")
        final_state = await recover_from_checkpoint(initial_state["meta_analysis_id"])
    
    return final_state

# Funções auxiliares
def estimate_required_steps(query: str) -> int:
    """Estima steps necessários baseado na complexidade"""
    # Lógica simplificada
    if "systematic review" in query.lower():
        return 100
    elif "meta-analysis" in query.lower():
        return 80
    else:
        return 50

async def recover_from_checkpoint(analysis_id: str) -> Dict:
    """Recupera estado do último checkpoint"""
    checkpoints = store.search(("checkpoints", analysis_id))
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: x.value["timestamp"])[-1]
        return latest.value
    return {}

# Executar
if __name__ == "__main__":
    query = """
    Meta-análise sobre eficácia de mindfulness vs CBT para ansiedade.
    Incluir apenas RCTs dos últimos 10 anos.
    Mínimo 20 estudos, com forest plot e análise de heterogeneidade.
    """
    
    final_state = asyncio.run(
        run_meta_analysis_with_iteration_control(
            query,
            max_time_minutes=45,
            quality_target=0.9
        )
    )
```

## 9. Melhores Práticas

### 9.1 Design Patterns

1. **Always Have an Exit Strategy**
   ```python
   # Sempre incluir condição de saída clara
   if iterations >= max_iterations or quality >= threshold or time_expired:
       return END
   ```

2. **Track Everything**
   ```python
   # Registrar todas as iterações para debug
   iteration_record = {
       "agent": agent_name,
       "iteration": current_iteration,
       "reason": iteration_reason,
       "metrics": current_metrics,
       "timestamp": datetime.now()
   }
   ```

3. **Fail Fast, Recover Gracefully**
   ```python
   # Detectar problemas cedo
   if improvement_rate < min_threshold and iterations > 2:
       return fallback_strategy()
   ```

### 9.2 Anti-Patterns a Evitar

1. **Loops Sem Condição de Saída**
   ```python
   # ❌ RUIM
   builder.add_edge("agent_a", "agent_b")
   builder.add_edge("agent_b", "agent_a")
   
   # ✅ BOM
   builder.add_conditional_edges(
       "agent_a",
       check_continuation_condition,
       {"agent_b": "agent_b", END: END}
   )
   ```

2. **Ignorar Recursion Limit**
   ```python
   # ❌ RUIM
   graph.invoke(state)  # Usa default de 25
   
   # ✅ BOM
   graph.invoke(state, {"recursion_limit": calculated_limit})
   ```

3. **Estado Não Rastreado**
   ```python
   # ❌ RUIM
   return {"messages": [AIMessage("Iterando...")]}
   
   # ✅ BOM
   return {
       "messages": [AIMessage("Iterando...")],
       "agent_iterations": {agent_name: iterations + 1},
       "iteration_history": history + [iteration_record]
   }
   ```

## 10. Conclusão

O controle efetivo de iterações em sistemas multi-agentes requer:

1. **Planejamento**: Definir limites e condições claras
2. **Monitoramento**: Rastrear todas as métricas relevantes
3. **Flexibilidade**: Adaptar limites baseado em contexto
4. **Resiliência**: Implementar recuperação de falhas
5. **Transparência**: Registrar decisões e razões

Para o **metanalyst-agent**, essas técnicas permitem:
- Refinamento iterativo de análises até atingir qualidade desejada
- Reprocessamento automático de artigos com falha
- Busca adicional quando dados insuficientes
- Terminação graciosa respeitando limites de tempo
- Recuperação de estado após interrupções

A chave é balancear a capacidade de iteração (necessária para qualidade) com controles que previnam loops infinitos e desperdício de recursos.


