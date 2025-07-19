# Metanalyst-Agent

O primeiro projeto open-source da Nobrega Medtech, focado na geração automatizada de meta-análises usando Python e LangGraph. O projeto implementa um sistema multi-agente com orquestração inteligente para pesquisa de literatura médica, extração, análise e geração de relatórios.

## Arquitetura

Sistema Hub-and-Spoke com Agents-as-a-Tool, onde um orquestrador central invoca agentes especializados baseado no estado atual:

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
```

## Setup Local (macOS)

### 1. Pré-requisitos

- Python 3.11+
- Docker Desktop
- Git

### 2. Clonar e Configurar Projeto

```bash
# Clonar repositório (se aplicável)
git clone https://github.com/seu-usuario/metanalyst-agent.git
cd metanalyst-agent

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Configurar PostgreSQL com Docker

```bash
# Executar script de setup automático
./docker/setup_postgres.sh
```

Este script irá:
- Criar e configurar container PostgreSQL
- Criar tabelas necessárias para LangGraph
- Testar a conexão
- Exibir informações de conexão

### 4. Configurar Variáveis de Ambiente

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar .env com suas chaves de API
nano .env
```

Configure as seguintes variáveis:
```bash
OPENAI_API_KEY=sk-sua-chave-openai-aqui
TAVILY_API_KEY=tvly-sua-chave-tavily-aqui
DATABASE_URL=postgresql://metanalyst:metanalyst123@localhost:5432/metanalyst
```

### 5. Testar Instalação

```bash
# Executar testes básicos
python -m pytest tests/test_basic_functionality.py -v

# Verificar configuração
python -c "from src.utils.config import validate_environment; print('✅ OK' if validate_environment() else '❌ Erro')"
```

### 6. Executar Sistema

```bash
# Método 1: Linha de comando
python -m src.main "Eficácia da aspirina na prevenção de AVC"

# Método 2: Interativo
python -m src.main
# Digite sua solicitação quando solicitado

# Exemplo de solicitações:
# - "Meta-análise sobre metformina em diabetes tipo 2"
# - "Eficácia de estatinas na prevenção cardiovascular"
# - "Aspirina vs placebo para prevenção de AVC"
```

## Comandos Docker Úteis

```bash
# Parar container
docker stop metanalyst-postgres

# Iniciar container existente
docker start metanalyst-postgres

# Remover container (dados serão perdidos)
docker rm metanalyst-postgres

# Ver logs do container
docker logs metanalyst-postgres

# Backup do banco
docker exec metanalyst-postgres pg_dump -U metanalyst metanalyst > backup.sql

# Restaurar backup
docker exec -i metanalyst-postgres psql -U metanalyst metanalyst < backup.sql
```

## Estrutura do Projeto

```
metanalyst-agent/
├── src/
│   ├── agents/          # Agentes especializados
│   ├── models/          # Modelos de estado e schemas
│   ├── tools/           # Ferramentas para agentes
│   ├── utils/           # Utilitários e configurações
│   └── main.py          # Ponto de entrada
├── tests/               # Testes
├── docker/              # Configurações Docker
├── requirements.txt     # Dependências Python
└── README.md
```

## Agentes

- **Orchestrator**: Condutor central que decide qual agente invocar
- **Researcher**: Busca literatura científica usando Tavily
- **Processor**: Extrai e vetoriza artigos científicos
- **Retriever**: Busca informações relevantes usando PICO
- **Writer**: Gera relatórios estruturados em HTML
- **Reviewer**: Revisa qualidade e sugere melhorias
- **Analyst**: Análises estatísticas e forest plots
- **Editor**: Integra relatório final

## Links Úteis

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/reference/)
- [Tavily API Reference](https://docs.tavily.com/documentation/api-reference/endpoint/search)
- [Tavily Extract API](https://docs.tavily.com/documentation/api-reference/endpoint/extract)
