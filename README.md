# Metanalyst-Agent

O primeiro projeto open-source da Nobrega Medtech, focado na geração automatizada de meta-análises usando Python e LangGraph. O projeto implementa um sistema multi-agente com orquestração inteligente para pesquisa de literatura médica, extração, análise e geração de relatórios.

## Arquitetura

Sistema multi-agente autônomo com arquitetura hub-and-spoke, onde cada agente é uma entidade independente capaz de tomar decisões sobre quais ferramentas usar através de `bind_tools`.

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── RETRIEVER
                │   SUPERVISOR   │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── WRITER
                         │
                         │
                     (PROCESSOR combines
                      extraction + vectorization)
```

## Setup Rápido

### 1. Pré-requisitos
- Python 3.11+
- Docker
- PostgreSQL (via Docker)

### 2. Configurar PostgreSQL

```bash
# Criar e iniciar container PostgreSQL
docker run --name metanalyst-postgres \
  -e POSTGRES_DB=metanalysis \
  -e POSTGRES_USER=metanalyst \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  -d postgres:15

# Verificar se está rodando
docker ps | grep metanalyst-postgres
```

### 3. Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar o sistema
pip install -e .
```

### 4. Configurar Variáveis de Ambiente

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar .env com suas chaves de API:
```

**Obrigatório - Configure estas variáveis:**
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here
POSTGRES_URL=postgresql://metanalyst:secure_password@localhost:5432/metanalysis
```

### 5. Executar

```bash
# Exemplo básico
python run_example.py

# Ou usar programaticamente
python -c "
from metanalyst_agent import MetanalystAgent
agent = MetanalystAgent()
result = agent.run('Meta-análise sobre mindfulness vs CBT para ansiedade')
print(result)
"
```

## Agentes Autônomos

### Supervisor Agent
- **Responsabilidade**: Coordenação central e delegação inteligente
- **Ferramentas**: Handoff tools para todos os agentes especializados
- **Decisões**: Analisa contexto e delega tarefas autonomamente

### Researcher Agent  
- **Responsabilidade**: Busca de literatura científica
- **Ferramentas**: PubMed API, Cochrane Library, geração de queries PICO
- **Autonomia**: Decide quando buscar mais artigos ou transferir para processamento

### Processor Agent
- **Responsabilidade**: Extração e vetorização de artigos
- **Ferramentas**: Tavily Extract, OpenAI Embeddings, FAISS
- **Autonomia**: Processa artigos e decide quando dados são suficientes

### Retriever Agent
- **Responsabilidade**: Busca semântica no vector store
- **Ferramentas**: FAISS, busca por similaridade
- **Autonomia**: Recupera informações relevantes baseadas em queries

### Analyst Agent
- **Responsabilidade**: Análises estatísticas e meta-análise
- **Ferramentas**: SciPy, NumPy, Matplotlib, forest plots
- **Autonomia**: Realiza cálculos e decide quando análise está completa

### Writer Agent
- **Responsabilidade**: Geração de relatórios estruturados
- **Ferramentas**: Templates HTML, citações Vancouver
- **Autonomia**: Cria relatórios e solicita revisão quando necessário

### Reviewer Agent
- **Responsabilidade**: Revisão de qualidade e conformidade
- **Ferramentas**: Checklist PRISMA, validação de dados
- **Autonomia**: Avalia qualidade e sugere melhorias

### Editor Agent
- **Responsabilidade**: Edição final e formatação
- **Ferramentas**: Formatação HTML, integração de gráficos
- **Autonomia**: Finaliza documento e decide quando está pronto

## Características Técnicas

- **Memória Persistente**: PostgreSQL para checkpoints e store
- **Busca Semântica**: FAISS com OpenAI embeddings
- **APIs Externas**: Tavily para extração, PubMed para busca
- **Visualizações**: Forest plots, funnel plots, gráficos estatísticos
- **Padrões Médicos**: Conformidade com PRISMA e Vancouver

## Exemplo de Uso

```python
from metanalyst_agent import MetanalystAgent

# Inicializar sistema
agent = MetanalystAgent()

# Executar meta-análise
result = agent.run(
    query="Meta-análise sobre eficácia da meditação mindfulness "
          "versus terapia cognitivo-comportamental para ansiedade em adultos"
)

# Verificar resultado
if result["success"]:
    print(f"✅ Meta-análise concluída!")
    print(f"📄 Relatório: {result['final_report_path']}")
    print(f"📊 Estudos incluídos: {result['studies_in_analysis']}")
    print(f"🎯 Effect size: {result['meta_analysis_results']['pooled_effect_size']}")
else:
    print(f"❌ Erro: {result['error']}")
```

### Teste de Configuração

Antes de usar, execute o teste de configuração:

```bash
python test_setup.py
```

### Execução Passo a Passo

1. **Configure o ambiente:**
   ```bash
   # Inicie PostgreSQL
   docker run --name metanalyst-postgres \
     -e POSTGRES_DB=metanalysis \
     -e POSTGRES_USER=metanalyst \
     -e POSTGRES_PASSWORD=secure_password \
     -p 5432:5432 -d postgres:15
   
   # Configure .env com suas API keys
   cp .env.example .env
   # Edite .env com OPENAI_API_KEY e TAVILY_API_KEY
   ```

2. **Teste a configuração:**
   ```bash
   python test_setup.py
   ```

3. **Execute exemplos:**
   ```bash
   python run_example.py
   ```

## Contribuição

Este é o primeiro projeto open-source da Nobrega Medtech. Contribuições são bem-vindas!

## Licença

MIT License - veja LICENSE para detalhes.