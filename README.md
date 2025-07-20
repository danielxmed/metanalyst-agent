# Metanalyst-Agent

Sistema open source para geração automatizada de meta-análises médicas utilizando LangGraph e modelos LLM. No momento, a única interface disponível é via linha de comando.

## Requisitos

- Python 3.12+
- PostgreSQL em execução e acessível via `DATABASE_URL`
- Chaves de API para OpenAI e Tavily

Instale as dependências e configure o banco antes de rodar o CLI.

## Instalação

1. Clone o repositório e entre na pasta do projeto.
2. Instale as dependências específicas do CLI:
   ```bash
   pip install -r requirements-cli.txt
   ```
3. Exporte as variáveis de ambiente necessárias:
   ```bash
   export OPENAI_API_KEY="sua_openai_key"
   export TAVILY_API_KEY="sua_tavily_key"
   export DATABASE_URL="postgresql://usuario:senha@localhost:5432/metanalysis"
   ```
4. Certifique-se de que o banco PostgreSQL esteja configurado. Há scripts em `scripts/` para facilitar a criação do banco e tabelas.

## Uso

Execute o CLI em modo interativo:
```bash
python3 run_cli.py
```

Para rodar uma meta-análise diretamente pela linha de comando:
```bash
python3 run_cli.py --query "pico ou pergunta" --max-articles 20 --storage postgres
```

O parâmetro `--storage postgres` usa o banco configurado via `DATABASE_URL` para armazenar checkpoints e dados.

Consulte `run_cli.py --help` para todas as opções disponíveis.

## Documentação adicional

Detalhes de arquitetura, proposta do sistema e conceitos de LangGraph estão disponíveis em [`docs/`](docs/).
