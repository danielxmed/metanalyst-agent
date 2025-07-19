# MetAnalyst Agent

Sistema multi-agente autônomo para geração automatizada de meta-análises médicas usando LangGraph e LLMs.

## 🏗️ Arquitetura

Sistema hub-and-spoke com agentes verdadeiramente autônomos:

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR  
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── RETRIEVER
                │   SUPERVISOR    │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── WRITER
                         │
                         │
                     VECTORIZER

    ● = Supervisor Agent (Hub)
    │ = Handoff Tools (Agents-as-a-Tool)
```

### Agentes Autônomos

- **Supervisor**: Coordena e delega tarefas usando handoff tools
- **Researcher**: Busca literatura usando Tavily com domínios médicos específicos
- **Processor**: Extrai conteúdo e dados estatísticos dos artigos
- **Vectorizer**: Cria embeddings e gerencia vector store
- **Retriever**: Busca informações relevantes no vector store
- **Analyst**: Realiza análises estatísticas e gera visualizações
- **Writer**: Gera relatórios estruturados em HTML
- **Reviewer**: Revisa qualidade e sugere melhorias
- **Editor**: Finaliza e formata o documento

## 🚀 Setup Rápido

### 1. Configurar PostgreSQL

```bash
# Iniciar PostgreSQL com Docker
docker run --name metanalyst-postgres \
  -e POSTGRES_PASSWORD=metanalyst123 \
  -e POSTGRES_USER=metanalyst \
  -e POSTGRES_DB=metanalyst \
  -p 5432:5432 \
  -d postgres:15

# Aguardar inicialização
sleep 10

# Criar tabelas do LangGraph
python -c "
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
import asyncio

async def setup_db():
    DB_URI = 'postgresql://metanalyst:metanalyst123@localhost:5432/metanalyst'
    
    # Setup checkpointer tables
    async with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
    
    # Setup store tables  
    async with PostgresStore.from_conn_string(DB_URI) as store:
        await store.setup()
    
    print('Database setup complete!')

asyncio.run(setup_db())
"
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente

```bash
cp .env.example .env
# Editar .env com suas chaves de API
```

### 4. Executar

```python
from metanalyst_agent import MetAnalystGraph

# Criar e executar meta-análise
graph = MetAnalystGraph()
result = graph.run_meta_analysis(
    "Eficácia da meditação mindfulness versus terapia cognitivo-comportamental para ansiedade"
)
```

## 🔧 Configuração

### Variáveis de Ambiente Obrigatórias

```env
# APIs LLM (pelo menos uma)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Tavily Search
TAVILY_API_KEY=tvly-...

# Database
DATABASE_URL=postgresql://metanalyst:metanalyst123@localhost:5432/metanalyst
```

## 📊 Domínios de Pesquisa

O sistema busca literatura nos seguintes domínios de alta qualidade:

- **Periódicos de Alto Impacto**: NEJM, JAMA, The Lancet, BMJ
- **Bases Científicas**: PubMed, PMC, SciELO
- **Bibliotecas Especializadas**: Cochrane Library

## 🎯 Funcionalidades

- ✅ Busca automatizada de literatura médica
- ✅ Extração de dados estatísticos
- ✅ Análise de meta-análise com forest plots
- ✅ Avaliação de heterogeneidade (I²)
- ✅ Relatórios HTML com citações Vancouver
- ✅ Memória persistente entre sessões
- ✅ Sistema de qualidade e revisão

## 📈 Exemplo de Uso

```python
import asyncio
from metanalyst_agent import create_meta_analysis_system

async def main():
    # Criar sistema
    system = create_meta_analysis_system()
    
    # Executar meta-análise
    config = {"configurable": {"thread_id": "analysis_001"}}
    
    async for chunk in system.astream(
        {
            "messages": [{
                "role": "user", 
                "content": "Meta-análise sobre estatinas na prevenção cardiovascular"
            }]
        },
        config
    ):
        print(f"[{chunk.get('current_agent', 'system')}]: {chunk['messages'][-1].content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Monitoramento

O sistema registra todas as transições entre agentes e decisões tomadas:

```python
# Visualizar histórico de execução
history = system.get_execution_history(thread_id="analysis_001")
for step in history:
    print(f"{step.timestamp}: {step.from_agent} → {step.to_agent}")
    print(f"Razão: {step.reason}")
```

## 🧪 Testes

```bash
# Executar testes
pytest tests/ -v

# Teste específico de agente
pytest tests/test_researcher_agent.py -v
```

## 📚 Documentação

- [Guia de Agentes](docs/agents.md)
- [API Reference](docs/api.md)
- [Exemplos](examples/)

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

**Nobrega Medtech** - Primeiro projeto open-source focado em automação de meta-análises médicas.