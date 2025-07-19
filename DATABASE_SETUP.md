# 🗄️ Configuração do PostgreSQL para Metanalyst-Agent

Este guia fornece instruções completas para configurar o PostgreSQL como backend único para **memória de curto prazo** (checkpointers) e **memória de longo prazo** (stores) do metanalyst-agent.

## 📋 Visão Geral

O sistema usa PostgreSQL para:
- **Checkpointers**: Estado das conversações e execuções (memória de curto prazo)
- **Stores**: Dados permanentes entre sessões (memória de longo prazo)
- **Dados da aplicação**: Meta-análises, artigos, chunks vetorizados, análises estatísticas

## 🚀 Opções de Setup

### Opção 1: Setup Automático (Recomendado)

Execute o script de setup automático:

```bash
# Dar permissão de execução
chmod +x scripts/setup_postgres.sh

# Setup completo (instala PostgreSQL se necessário)
./scripts/setup_postgres.sh --install-pg --auto-config

# Ou apenas configurar banco existente
./scripts/setup_postgres.sh
```

**Parâmetros disponíveis:**
- `--install-pg`: Instala PostgreSQL automaticamente
- `--auto-config`: Configura PostgreSQL automaticamente
- `--skip-deps`: Pula instalação de dependências Python
- `--db-name NAME`: Nome do banco (default: metanalysis)
- `--db-user USER`: Nome do usuário (default: metanalyst)
- `--db-password PWD`: Senha (default: gerada automaticamente)

### Opção 2: Docker Compose (Desenvolvimento)

Para desenvolvimento rápido com Docker:

```bash
# Navegar para pasta de scripts
cd scripts

# Subir PostgreSQL + PgAdmin + Redis
docker-compose up -d

# Verificar se está funcionando
docker-compose ps
```

**Serviços inclusos:**
- **PostgreSQL 16**: `localhost:5432`
- **PgAdmin**: `http://localhost:5050` (admin@metanalyst.com / admin123)
- **Redis**: `localhost:6379` (para cache futuro)

**Credenciais padrão:**
```
Database: metanalysis
User: metanalyst
Password: metanalyst_secure_password_2024
```

### Opção 3: Setup Manual

#### 3.1 Instalar PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-client
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install -y postgresql postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

#### 3.2 Criar Banco e Usuário

```bash
# Conectar como postgres
sudo -u postgres psql

-- Criar usuário
CREATE USER metanalyst WITH PASSWORD 'sua_senha_segura';

-- Criar banco
CREATE DATABASE metanalysis OWNER metanalyst;

-- Conceder privilégios
GRANT ALL PRIVILEGES ON DATABASE metanalysis TO metanalyst;

-- Sair
\q
```

#### 3.3 Executar Script de Setup

```bash
# Executar script SQL
PGPASSWORD='sua_senha_segura' psql -h localhost -U metanalyst -d metanalysis -f scripts/setup_database.sql
```

## 🔧 Configuração das Variáveis de Ambiente

Após o setup, configure seu arquivo `.env`:

```bash
# Copiar exemplo
cp .env.example .env

# Editar com suas credenciais
nano .env
```

**Variáveis principais:**
```bash
# Configuração do banco (gerada automaticamente em .env.db)
DATABASE_URL=postgresql://metanalyst:senha@localhost:5432/metanalysis

# APIs necessárias
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Configurações opcionais
DEFAULT_MAX_ARTICLES=50
DEFAULT_QUALITY_THRESHOLD=0.8
DEFAULT_MAX_ITERATIONS=5
```

## 📊 Estrutura do Banco de Dados

### Tabelas do LangGraph

| Tabela | Descrição |
|--------|-----------|
| `checkpoints` | Estados das conversações (memória curto prazo) |
| `checkpoint_blobs` | Dados grandes dos checkpoints |
| `checkpoint_writes` | Operações de escrita dos checkpoints |
| `store` | Dados permanentes (memória longo prazo) |

### Tabelas da Aplicação

| Tabela | Descrição |
|--------|-----------|
| `meta_analyses` | Meta-análises principais |
| `articles` | Artigos processados |
| `article_chunks` | Chunks vetorizados dos artigos |
| `statistical_analyses` | Análises estatísticas e plots |
| `agent_logs` | Logs de execução dos agentes |
| `embedding_cache` | Cache de embeddings para otimização |

### Views Úteis

| View | Descrição |
|------|-----------|
| `meta_analysis_stats` | Estatísticas das meta-análises |
| `agent_performance` | Performance dos agentes |

## 🧪 Testando a Configuração

### Teste de Conectividade

```python
from metanalyst_agent.database import get_database_manager

# Testar conexão
db = get_database_manager()
health = db.health_check()
print(health)
```

### Teste via CLI

```bash
# Conectar ao banco
psql -h localhost -U metanalyst -d metanalysis

-- Verificar tabelas
\dt

-- Ver estatísticas
SELECT * FROM database_stats();

-- Ver meta-análises
SELECT * FROM meta_analysis_stats;
```

## 📈 Monitoramento e Manutenção

### Comandos Úteis

```sql
-- Verificar tamanho das tabelas
SELECT * FROM database_stats();

-- Limpar dados antigos (30 dias)
SELECT cleanup_old_data(30);

-- Ver performance dos agentes
SELECT * FROM agent_performance;

-- Verificar saúde do banco
SELECT 
    pg_database_size(current_database()) as db_size_bytes,
    count(*) as total_checkpoints 
FROM checkpoints;
```

### Backup e Restore

```bash
# Fazer backup
pg_dump -h localhost -U metanalyst -d metanalysis > backup_$(date +%Y%m%d).sql

# Restaurar backup
psql -h localhost -U metanalyst -d metanalysis < backup_20240101.sql

# Backup automático (crontab)
0 2 * * * /usr/bin/pg_dump -h localhost -U metanalyst -d metanalysis > /backups/metanalysis_$(date +\%Y\%m\%d).sql
```

### Otimização

```sql
-- Atualizar estatísticas
ANALYZE;

-- Reindexar se necessário
REINDEX DATABASE metanalysis;

-- Vacuum para limpeza
VACUUM ANALYZE;
```

## 🔒 Segurança

### Configurações Recomendadas

1. **Senhas fortes**: Use senhas complexas geradas automaticamente
2. **Conexões SSL**: Para produção, habilite SSL
3. **Firewall**: Limite acesso à porta 5432
4. **Usuários específicos**: Não use o usuário postgres em produção

### Configuração de Produção

```bash
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
password_encryption = scram-sha-256

# pg_hba.conf
hostssl all metanalyst 0.0.0.0/0 scram-sha-256
```

## 🐛 Troubleshooting

### Problemas Comuns

**1. Erro de conexão:**
```
FATAL: password authentication failed for user "metanalyst"
```
**Solução:** Verificar senha e configuração do pg_hba.conf

**2. Banco não encontrado:**
```
FATAL: database "metanalysis" does not exist
```
**Solução:** Criar banco com `createdb -O metanalyst metanalysis`

**3. Extensões não instaladas:**
```
ERROR: extension "uuid-ossp" is not available
```
**Solução:** Instalar `postgresql-contrib` e executar como superuser

**4. Permissões insuficientes:**
```
ERROR: permission denied for table checkpoints
```
**Solução:** Verificar privilégios do usuário metanalyst

### Logs Úteis

```bash
# Ver logs do PostgreSQL
sudo tail -f /var/log/postgresql/postgresql-*.log

# Ver logs do Docker
docker-compose logs -f postgres

# Verificar status do serviço
sudo systemctl status postgresql
```

## 📚 Referências

- [LangGraph Checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
- [LangGraph Stores](https://langchain-ai.github.io/langgraph/reference/store/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Psycopg2 Documentation](https://www.psycopg.org/docs/)

## 🆘 Suporte

Se encontrar problemas:

1. Verifique os logs do PostgreSQL
2. Execute `./scripts/setup_postgres.sh --help` para ver opções
3. Teste a conectividade com `psql`
4. Verifique as variáveis de ambiente no `.env`

**Comandos de diagnóstico:**
```bash
# Verificar se PostgreSQL está rodando
sudo systemctl status postgresql

# Testar conectividade
pg_isready -h localhost -p 5432 -U metanalyst

# Ver configuração atual
psql -h localhost -U metanalyst -d metanalysis -c "SHOW config_file;"
```