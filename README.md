# Metanalyst Agent - Automated Medical Meta-Analysis System

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

O primeiro projeto open-source da Nobrega Medtech, focado na geração automatizada de meta-análises médicas usando Python e LangGraph.

## 🎯 Visão Geral

O **Metanalyst Agent** é um sistema multi-agente com arquitetura hub-and-spoke que automatiza todo o processo de meta-análise médica, desde a busca de literatura até a geração de relatórios finais com análises estatísticas e gráficos.

### Características Principais

- 🤖 **Interface Agêntica**: Interação em linguagem natural com o orchestrator
- 🔄 **Iteração Automática**: O orchestrator itera sobre si mesmo sem intervenção manual
- 📚 **Busca Inteligente**: Integração com Tavily API para busca em 70+ bases médicas
- 📊 **Análise Estatística**: Geração automática de forest plots e análises estatísticas
- 📋 **Relatórios HTML**: Geração de relatórios profissionais em HTML
- 🧠 **Multi-Agente**: Arquitetura com 9 agentes especializados

## 🚀 Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/danielxmed/metanalyst-agent.git
cd metanalyst-agent

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com suas API keys
```

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# APIs necessárias
TAVILY_API_KEY=tvly-your-api-key
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Configurações do sistema
MAX_PAPERS_PER_SEARCH=15
CHUNK_SIZE=500
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=claude-3-5-sonnet-20241022
```

### APIs Necessárias

1. **Tavily API**: Para busca e extração de literatura científica
2. **OpenAI API**: Para embeddings e alguns modelos de linguagem
3. **Anthropic API**: Para o modelo principal Claude

## 🤖 Interface Agêntica

### Execução Automática

```bash
# Executa workflow agêntico com interação automática
python -m src.cli run

# Ou com pedido direto
python -m src.cli run -r "Meta-análise sobre eficácia da metformina em diabéticos"

# Controlar número máximo de iterações
python -m src.cli run -r "Análise de estatinas" -i 25
```

### Exemplos de Pedidos

```bash
# Diabetes e metformina
python -m src.cli run -r "Fazer meta-análise sobre eficácia da metformina em diabéticos tipo 2"

# Estatinas e prevenção cardiovascular
python -m src.cli run -r "Analisar impacto de estatinas na prevenção cardiovascular"

# Aspirina vs placebo
python -m src.cli run -r "Meta-análise de aspirina vs placebo para prevenção de AVC"
```

### Comandos Disponíveis

```bash
# Verificar status do sistema
python -m src.cli status

# Executar exemplo pré-definido
python -m src.cli example

# Obter ajuda
python -m src.cli --help
```

## 🏗️ Arquitetura do Sistema

### Hub-and-Spoke com Orchestrator Agêntico

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── EXTRACTOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── VECTORIZER
                │   ORCHESTRATOR  │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── RETRIEVER
                         │
                         │
                     WRITER

    ● = Orchestrator Central (Hub)
    │ = Conexões Diretas (Agents-as-a-Tool)
```

### Agentes Especializados

1. **Orchestrator**: Condutor central com decisões baseadas em LLM
2. **Researcher**: Busca literatura científica usando Tavily
3. **Extractor**: Extrai conteúdo de URLs e processa para JSON
4. **Vectorizer**: Cria embeddings e vector store com FAISS
5. **Retriever**: Busca informações relevantes no vector store
6. **Writer**: Gera relatórios estruturados em HTML
7. **Reviewer**: Revisa qualidade e sugere melhorias
8. **Analyst**: Realiza análises estatísticas e forest plots
9. **Editor**: Integra relatório final com análises

## 🔄 Fluxo de Trabalho Agêntico

### 1. Inicialização
```bash
python -m src.cli run -r "Sua solicitação em linguagem natural"
```

### 2. Iteração Automática
O orchestrator:
- Analisa o estado atual
- Toma decisões inteligentes
- Invoca agentes especializados
- Atualiza o estado global
- Continua até completar o workflow

### 3. Saída
- Relatório HTML profissional
- Gráficos e análises estatísticas
- Log completo do processo

## 📊 Exemplo de Saída

```
🎉 WORKFLOW COMPLETED
===============================================
✅ Status: Complete
🔄 Total Iterations: 12
📊 Final report: outputs/meta_analysis_report_abc123.html
```

## 🛠️ Desenvolvimento

### Estrutura do Projeto

```
metanalyst-agent/
├── src/
│   ├── agents/          # Agentes especializados
│   │   ├── orchestrator.py
│   │   ├── researcher.py
│   │   └── ...
│   ├── tools/           # Ferramentas dos agentes
│   │   ├── tavily_tools.py
│   │   ├── orchestrator_tools.py
│   │   └── ...
│   ├── models/          # Esquemas de dados
│   │   ├── state.py
│   │   └── schemas.py
│   ├── utils/           # Utilitários
│   │   └── config.py
│   ├── cli.py           # Interface CLI agêntica
│   └── main.py          # Ponto de entrada
├── tests/               # Testes unitários
├── outputs/             # Relatórios gerados
├── data/                # Dados e vector stores
└── docs/                # Documentação
```

### Executar Testes

```bash
# Executar todos os testes
pytest tests/

# Teste específico
pytest tests/test_orchestrator.py -v

# Cobertura de código
pytest --cov=src tests/
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📚 Documentação

- [CONTEXT.md](CONTEXT.md): Contexto técnico detalhado
- [Tavily API Docs](https://docs.tavily.com/): Documentação da API
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/): Framework multi-agente

## 🔒 Segurança

- Nunca commite suas API keys
- Use o arquivo `.env` para configurações sensíveis
- Mantenha as dependências atualizadas

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🏥 Nobrega Medtech

Primeiro projeto open-source da Nobrega Medtech, focado em democratizar o acesso a ferramentas avançadas de análise médica através de inteligência artificial.

---

**Importante**: Este sistema é para fins de pesquisa e educação. Sempre consulte profissionais médicos qualificados para decisões clínicas.
