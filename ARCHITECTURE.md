# Arquitetura Técnica - Metanalyst-Agent

## Visão Geral

O metanalyst-agent implementa uma arquitetura **Hub-and-Spoke** usando LangGraph, onde um agente orquestrador central coordena agentes especializados para executar diferentes fases da meta-análise médica.

## Arquitetura Hub-and-Spoke

```
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
```

### Princípios da Arquitetura

1. **Hub Central**: O orquestrador mantém estado global e lógica de decisão
2. **Agents-as-Tools**: Cada agente é uma ferramenta especializada
3. **Decisão Contextual**: A cada iteração, o orquestrador analisa o estado e escolhe o próximo agente
4. **Comunicação Direta**: Todos os agentes se comunicam apenas com o orquestrador
5. **Estado Compartilhado**: O orquestrador mantém e atualiza o estado global

## Componentes do Sistema

### 1. Estado Compartilhado (`MetaAnalysisState`)

O estado é a estrutura central que mantém todas as informações da meta-análise:

```python
class MetaAnalysisState(TypedDict):
    # Identificação e controle
    meta_analysis_id: str
    thread_id: str
    current_phase: Literal[...]
    current_agent: Optional[str]
    
    # PICO e pesquisa
    pico: Dict[str, str]
    search_queries: List[str]
    candidate_urls: List[Dict[str, Any]]
    
    # Processamento e vetorização
    processed_articles: List[Dict[str, Any]]
    vector_store_id: Optional[str]
    chunk_count: int
    
    # Análise e relatórios
    statistical_analysis: Dict[str, Any]
    final_report: Optional[str]
    
    # Mensagens e logs
    messages: Annotated[List[BaseMessage], add_messages]
    agent_logs: List[Dict[str, Any]]
```

### 2. Agente Orquestrador Central

O orquestrador é o "condutor" que analisa o estado atual e decide qual agente especializado invocar:

```python
def orchestrator_node(state: MetaAnalysisState) -> Command:
    """Decide próxima ação baseada no estado atual."""
    phase = state["current_phase"]
    
    if phase == "pico_definition":
        return Command(goto="researcher", update=...)
    elif phase == "search":
        return Command(goto="processor", update=...)
    # ... lógica de decisão
```

### 3. Agentes Especializados

#### Researcher Agent
- **Responsabilidade**: Busca literatura científica
- **Ferramentas**: Tavily Search API
- **Entrada**: Estrutura PICO
- **Saída**: Lista de URLs candidatas com scores de relevância

#### Processor Agent
- **Responsabilidade**: Extração de dados e vetorização
- **Ferramentas**: Tavily Extract API, OpenAI Embeddings, FAISS
- **Entrada**: URLs para processar
- **Saída**: Dados estruturados e chunks vetorizados

#### Retriever Agent (Futuro)
- **Responsabilidade**: Busca semântica no vector store
- **Ferramentas**: FAISS, Cosine Similarity
- **Entrada**: Queries baseadas em PICO
- **Saída**: Chunks relevantes com contexto

#### Analyst Agent (Futuro)
- **Responsabilidade**: Análises estatísticas
- **Ferramentas**: SciPy, NumPy, Matplotlib
- **Entrada**: Dados extraídos dos estudos
- **Saída**: Meta-análises, forest plots, estatísticas

#### Writer Agent (Futuro)
- **Responsabilidade**: Geração de relatórios
- **Entrada**: Análises estatísticas e dados
- **Saída**: Relatório estruturado em HTML

#### Reviewer Agent (Futuro)
- **Responsabilidade**: Revisão de qualidade
- **Entrada**: Relatório draft
- **Saída**: Feedback e sugestões

#### Editor Agent (Futuro)
- **Responsabilidade**: Edição final
- **Entrada**: Relatório revisado
- **Saída**: Relatório final publicável

## Gerenciamento de Memória

### Memória de Curto Prazo (PostgreSQL Checkpointer)

- **Escopo**: Thread-level (uma conversação)
- **Persistência**: Durante a execução da meta-análise
- **Uso**: Estado da sessão, progresso atual, mensagens

```python
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)
graph = builder.compile(checkpointer=checkpointer)
```

### Memória de Longo Prazo (PostgreSQL Store)

- **Escopo**: Cross-thread (entre execuções)
- **Persistência**: Permanente
- **Uso**: Artigos processados, vector stores, análises reutilizáveis

```python
store = PostgresStore.from_conn_string(DATABASE_URL)

# Namespace hierárquico
namespace = ("metanalysis", analysis_id, "articles", article_id)
store.put(namespace, "data", article_data)
```

### Organização de Namespaces

```
metanalysis/
├── {analysis_id}/
│   ├── articles/
│   │   ├── {article_id}/
│   │   │   ├── data          # Dados completos do artigo
│   │   │   └── metadata      # Metadados para busca
│   │   └── ...
│   ├── vector_store/
│   │   └── metadata          # Metadados do vector store
│   ├── statistics/
│   │   └── results           # Resultados de análises
│   └── reports/
│       └── final             # Relatório final
```

## Fluxo de Execução

### 1. Inicialização
```python
initial_state = create_initial_state(user_request)
config = {"configurable": {"thread_id": state["thread_id"]}}
```

### 2. Iteração Automática
O orquestrador itera automaticamente:

1. **Análise do Estado**: Examina `current_phase` e dados disponíveis
2. **Decisão**: Escolhe próximo agente baseado na lógica de negócio
3. **Invocação**: Executa agente especializado como ferramenta
4. **Atualização**: Agente atualiza o estado global
5. **Retorno**: Controle volta ao orquestrador
6. **Repetição**: Processo continua até conclusão

### 3. Fases da Meta-análise

```
pico_definition → search → extraction → vectorization → 
analysis → writing → review → editing → completed
```

## Tecnologias Utilizadas

### Core Framework
- **LangGraph**: Orquestração de agentes e fluxo de estado
- **LangChain**: Integração com LLMs e ferramentas
- **PostgreSQL**: Persistência de estado e memória de longo prazo

### APIs Externas
- **OpenAI**: LLM para extração de dados e embeddings
- **Tavily**: Busca e extração de literatura científica

### Processamento de Dados
- **FAISS**: Vector store local para busca semântica
- **Pydantic**: Validação e serialização de dados
- **NumPy/SciPy**: Análises estatísticas (futuro)

### Infraestrutura
- **Docker**: Containerização do PostgreSQL
- **Python 3.11+**: Runtime principal

## Padrões de Design

### 1. Command Pattern
```python
return Command(
    goto="next_agent",
    update={"state_field": "new_value"}
)
```

### 2. State Machine
Cada fase representa um estado com transições bem definidas.

### 3. Strategy Pattern
Diferentes estratégias de busca, extração e análise baseadas no contexto.

### 4. Observer Pattern
Sistema de logs para monitoramento de ações dos agentes.

## Escalabilidade e Performance

### Otimizações Implementadas

1. **Batch Processing**: Embeddings gerados em lotes
2. **Caching**: Vector stores persistentes para reutilização
3. **Lazy Loading**: Dados carregados sob demanda
4. **Connection Pooling**: Pool de conexões PostgreSQL

### Limitações Atuais

1. **Processamento Sequencial**: Artigos processados um por vez
2. **Vector Store Local**: FAISS armazenado localmente
3. **Single Thread**: Execução em thread única

### Melhorias Futuras

1. **Processamento Paralelo**: Múltiplos artigos simultaneamente
2. **Vector Store Distribuído**: PostgreSQL com pgvector
3. **Cache Distribuído**: Redis para embeddings
4. **Monitoramento**: Métricas detalhadas de performance

## Configuração e Deployment

### Variáveis de Ambiente
```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
DATABASE_URL=postgresql://...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### Docker Setup
```bash
./docker/setup_postgres.sh  # Setup automático
```

### Monitoramento
- Logs estruturados por agente
- Métricas de tempo de execução
- Tracking de qualidade dos dados

Esta arquitetura garante flexibilidade, escalabilidade e manutenibilidade, permitindo fácil adição de novos agentes e modificação de comportamentos sem afetar o sistema como um todo.