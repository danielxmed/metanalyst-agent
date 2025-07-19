# 🤖 Metanalyst-Agent

O **primeiro projeto open-source da Nobrega Medtech** para **meta-análises automatizadas** usando Python e LangGraph. Sistema multi-agente autônomo que realiza o processo completo de meta-análise médica seguindo diretrizes PRISMA.

## 🎯 Visão Geral

O Metanalyst-Agent implementa uma arquitetura **hub-and-spoke** com agentes especializados que trabalham de forma autônoma para produzir meta-análises de alta qualidade científica, desde a busca de literatura até a geração de relatórios finais formatados.

### ✨ Características Principais

- 🧠 **Agentes Autônomos**: Cada agente decide autonomamente quais ferramentas usar via `create_react_agent`
- 🏥 **AI-First Approach**: LLMs são usados em todas as etapas possíveis do processo
- 📊 **Diretrizes PRISMA**: Aderência completa às diretrizes internacionais
- 🔄 **Arquitetura Hub-and-Spoke**: Supervisor central coordena agentes especializados
- 💾 **Persistência Inteligente**: Estado compartilhado com checkpointers e stores
- 🌐 **Bases Médicas**: Integração com PubMed, Cochrane, NEJM, JAMA, Lancet, BMJ

## 🏗️ Arquitetura do Sistema

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── VECTORIZER
                │   SUPERVISOR    │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── RETRIEVER
                         │
                         │
                     WRITER

    ● = Supervisor (Hub Central)
    │ = Comunicação via Handoff Tools
```

### 🤖 Agentes Especializados

| Agente | Responsabilidade | Ferramentas Principais |
|--------|------------------|------------------------|
| **Supervisor** | Coordenação central e tomada de decisões | Handoff tools para todos os agentes |
| **Researcher** | Busca sistemática de literatura | Tavily Search, geração de queries PICO |
| **Processor** | Extração e processamento de artigos | Tavily Extract, LLM para dados estatísticos |
| **Vectorizer** | Vetorização para busca semântica | OpenAI Embeddings, FAISS |
| **Retriever** | Recuperação inteligente de informações | Busca semântica, ranking por LLM |
| **Analyst** | Análise estatística e visualizações | Meta-análise, forest plots, heterogeneidade |
| **Writer** | Geração de relatórios PRISMA | Seções estruturadas, formatação HTML |
| **Reviewer** | Controle de qualidade científica | Avaliação de qualidade, validação estatística |
| **Editor** | Formatação final profissional | HTML/CSS científico, múltiplos formatos |

## 🚀 Instalação

### Pré-requisitos

- Python 3.9+
- PostgreSQL (opcional, para persistência)
- Chaves de API: OpenAI, Tavily

### Instalação via pip

```bash
pip install -r requirements.txt
```

### Configuração de Variáveis de Ambiente

```bash
# APIs obrigatórias
export OPENAI_API_KEY="sua_chave_openai"
export TAVILY_API_KEY="sua_chave_tavily"

# PostgreSQL (opcional)
export POSTGRES_URL="postgresql://user:pass@localhost:5432/metanalysis"
```

## 📖 Uso

### Execução via Linha de Comando

```bash
# Executar meta-análise
python -m metanalyst_agent "Eficácia da metformina vs placebo para prevenção de diabetes tipo 2"

# Mostrar exemplos
python -m metanalyst_agent --examples

# Visualizar grafo
python -m metanalyst_agent --visualize

# Modo desenvolvimento (memória)
python -m metanalyst_agent --memory "Sua query aqui"
```

### Execução via Python

```python
from metanalyst_agent import run_meta_analysis

query = """
Realize uma meta-análise sobre a eficácia da meditação mindfulness 
versus terapia cognitivo-comportamental para tratamento de ansiedade 
em adultos. Inclua forest plot e análise de heterogeneidade.
"""

for result in run_meta_analysis(query):
    if result.get("final_report"):
        print("✅ Meta-análise concluída!")
        print(result["final_report"])
        break
```

### Modo Interativo

```bash
python -m metanalyst_agent
# Digite suas queries interativamente
```

## 🔧 Configuração Avançada

### Personalizar Configurações

```python
from metanalyst_agent.models.config import config

# Ajustar configurações
config.search.max_results = 100
config.search.min_relevance_score = 0.8
```

### Usar Persistência Customizada

```python
from metanalyst_agent import build_meta_analysis_graph
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = build_meta_analysis_graph(checkpointer=checkpointer)
```

## 📊 Exemplo de Fluxo Completo

1. **Definição PICO**: Supervisor analisa query e define estrutura PICO
2. **Busca de Literatura**: Researcher busca artigos em bases médicas
3. **Processamento**: Processor extrai conteúdo e dados estatísticos
4. **Vetorização**: Vectorizer cria embeddings para busca semântica
5. **Recuperação**: Retriever busca informações relevantes
6. **Análise**: Analyst realiza meta-análise e gera visualizações
7. **Redação**: Writer gera relatório seguindo PRISMA
8. **Revisão**: Reviewer avalia qualidade e sugere melhorias
9. **Edição**: Editor produz versão final formatada

## 🛠️ Ferramentas por Categoria

### 🔍 Pesquisa e Coleta
- `search_medical_literature`: Busca em bases médicas
- `generate_search_queries`: Queries otimizadas PICO
- `evaluate_article_relevance`: Avaliação de relevância

### 🔄 Processamento
- `extract_article_content`: Extração via Tavily
- `extract_statistical_data`: Dados estatísticos via LLM
- `assess_article_quality`: Avaliação metodológica

### 🎯 Vetorização e Busca
- `create_text_chunks`: Chunking inteligente
- `generate_embeddings`: OpenAI embeddings
- `retrieve_relevant_chunks`: Busca semântica

### 📈 Análise Estatística
- `calculate_meta_analysis`: Cálculos de meta-análise
- `create_forest_plot`: Visualizações
- `assess_heterogeneity`: Análise de heterogeneidade

### ✍️ Geração de Relatórios
- `generate_report_section`: Seções PRISMA
- `format_html_report`: Formatação profissional
- `assess_report_quality`: Controle de qualidade

## 🔗 Handoff Tools

Sistema de transferência inteligente entre agentes:

```python
transfer_to_researcher    # → Buscar mais literatura
transfer_to_processor     # → Processar artigos encontrados  
transfer_to_vectorizer    # → Criar vector store
transfer_to_retriever     # → Buscar informações
transfer_to_analyst       # → Análise estatística
transfer_to_writer        # → Gerar relatório
transfer_to_reviewer      # → Revisar qualidade
transfer_to_editor        # → Formatação final
```

## 📋 Estado Compartilhado

O sistema mantém estado completo da meta-análise:

```python
{
    "meta_analysis_id": "uuid",
    "current_phase": "analysis",
    "pico": {"P": "...", "I": "...", "C": "...", "O": "..."},
    "processed_articles": [...],
    "statistical_analysis": {...},
    "final_report": "...",
    # ... e muito mais
}
```

## 🔬 Exemplos de Queries

### Meta-análise Farmacológica
```
Realize uma meta-análise sobre a eficácia da metformina versus placebo 
para prevenção de diabetes tipo 2 em adultos com pré-diabetes. 
Inclua análise de heterogeneidade e forest plot.
```

### Meta-análise Psicológica
```
Conduza uma meta-análise comparando terapia cognitivo-comportamental 
versus mindfulness para tratamento de depressão em adultos. 
Avalie qualidade da evidência e heterogeneidade.
```

### Meta-análise Cirúrgica
```
Faça uma meta-análise comparando cirurgia laparoscópica versus 
cirurgia aberta para apendicectomia. Foque em tempo de recuperação 
e complicações pós-operatórias.
```

## 🧪 Desenvolvimento

### Estrutura do Projeto

```
metanalyst_agent/
├── agents/           # Agentes autônomos
├── tools/           # Ferramentas especializadas
├── models/          # Estado e configuração
├── graph/           # Grafo principal
└── main.py         # Interface principal
```

### Executar Testes

```bash
pytest tests/
```

### Contribuir

1. Fork o repositório
2. Crie branch para feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para branch (`git push origin feature/nova-feature`)
5. Abra Pull Request

## 🏥 Diretrizes Médicas

O sistema segue rigorosamente:

- **PRISMA**: Preferred Reporting Items for Systematic Reviews and Meta-Analyses
- **Cochrane**: Métodos de meta-análise
- **GRADE**: Avaliação de qualidade da evidência
- **Vancouver**: Estilo de citações médicas

## 📈 Métricas de Qualidade

- ✅ Aderência PRISMA completa
- ✅ Validação estatística automática
- ✅ Controle de qualidade multi-camadas
- ✅ Rastreabilidade de todas as fontes
- ✅ Transparência metodológica

## 🤝 Suporte

- 📧 Email: suporte@nobregamedtech.com
- 🐛 Issues: [GitHub Issues](https://github.com/nobregamedtech/metanalyst-agent/issues)
- 📖 Docs: [Documentação Completa](https://docs.nobregamedtech.com/metanalyst-agent)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **LangGraph**: Framework multi-agente
- **OpenAI**: Modelos de linguagem e embeddings
- **Tavily**: APIs de busca e extração
- **Comunidade científica**: Diretrizes e padrões

---

**Nobrega Medtech** - Inovação em tecnologia médica através de IA

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://langchain.ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Medical](https://img.shields.io/badge/Medical-PRISMA-red.svg)](https://prisma-statement.org)