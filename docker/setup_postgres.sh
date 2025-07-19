#!/bin/bash

# Script para configurar PostgreSQL para o metanalyst-agent
# Compatível com macOS usando Docker

set -e

echo "🐘 Configurando PostgreSQL para Metanalyst-Agent"
echo "================================================"

# Configurações
CONTAINER_NAME="metanalyst-postgres"
DB_NAME="metanalyst"
DB_USER="metanalyst"
DB_PASSWORD="metanalyst123"
DB_PORT="5432"

# Verificar se Docker está rodando
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está rodando. Por favor, inicie o Docker Desktop."
    exit 1
fi

# Parar e remover container existente se houver
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🛑 Parando container existente..."
    docker stop ${CONTAINER_NAME} || true
    echo "🗑️  Removendo container existente..."
    docker rm ${CONTAINER_NAME} || true
fi

# Criar e iniciar novo container PostgreSQL
echo "🚀 Criando novo container PostgreSQL..."
docker run --name ${CONTAINER_NAME} \
    -e POSTGRES_DB=${DB_NAME} \
    -e POSTGRES_USER=${DB_USER} \
    -e POSTGRES_PASSWORD=${DB_PASSWORD} \
    -p ${DB_PORT}:5432 \
    -d postgres:15

echo "⏳ Aguardando PostgreSQL inicializar..."
sleep 10

# Verificar se o container está rodando
if ! docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Falha ao iniciar container PostgreSQL"
    docker logs ${CONTAINER_NAME}
    exit 1
fi

# Testar conexão
echo "🔍 Testando conexão..."
if docker exec ${CONTAINER_NAME} pg_isready -U ${DB_USER} -d ${DB_NAME}; then
    echo "✅ PostgreSQL está rodando e acessível!"
else
    echo "❌ Falha ao conectar com PostgreSQL"
    exit 1
fi

# Criar tabelas necessárias para LangGraph
echo "📋 Criando tabelas do LangGraph..."

# SQL para criar tabelas do checkpointer
CHECKPOINTER_SQL="
-- Tabelas para LangGraph Checkpointer
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS checkpoints_parent_id_idx 
ON checkpoints (thread_id, checkpoint_ns, parent_checkpoint_id);

-- Tabelas para LangGraph Store
CREATE TABLE IF NOT EXISTS store (
    namespace TEXT[] NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (namespace, key)
);

CREATE INDEX IF NOT EXISTS store_namespace_idx ON store USING GIN (namespace);
CREATE INDEX IF NOT EXISTS store_created_at_idx ON store (created_at);
CREATE INDEX IF NOT EXISTS store_updated_at_idx ON store (updated_at);
"

# Executar SQL
docker exec -i ${CONTAINER_NAME} psql -U ${DB_USER} -d ${DB_NAME} << EOF
${CHECKPOINTER_SQL}
EOF

if [ $? -eq 0 ]; then
    echo "✅ Tabelas criadas com sucesso!"
else
    echo "❌ Erro ao criar tabelas"
    exit 1
fi

# Exibir informações de conexão
echo ""
echo "🎉 Setup concluído com sucesso!"
echo "================================"
echo "Container: ${CONTAINER_NAME}"
echo "Database: ${DB_NAME}"
echo "User: ${DB_USER}"
echo "Password: ${DB_PASSWORD}"
echo "Port: ${DB_PORT}"
echo "Connection String: postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${DB_PORT}/${DB_NAME}"
echo ""
echo "Comandos úteis:"
echo "  Parar:     docker stop ${CONTAINER_NAME}"
echo "  Iniciar:   docker start ${CONTAINER_NAME}"
echo "  Conectar:  docker exec -it ${CONTAINER_NAME} psql -U ${DB_USER} -d ${DB_NAME}"
echo "  Logs:      docker logs ${CONTAINER_NAME}"
echo "  Remover:   docker rm ${CONTAINER_NAME}"
echo ""
echo "⚠️  Certifique-se de que as variáveis de ambiente estão configuradas no arquivo .env"