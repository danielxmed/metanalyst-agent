-- =====================================================
-- METANALYST-AGENT DATABASE SETUP SCRIPT
-- =====================================================
-- Este script configura um banco PostgreSQL completo
-- para o sistema metanalyst-agent com:
-- - Checkpointers (memória de curto prazo)
-- - Stores (memória de longo prazo)
-- - Índices otimizados para performance
-- =====================================================

-- Criar database (execute como superuser)
-- CREATE DATABASE metanalysis;
-- \c metanalysis;

-- Criar extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- CHECKPOINTERS - MEMÓRIA DE CURTO PRAZO
-- =====================================================

-- Tabela para checkpoints do LangGraph
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Tabela para checkpoint blobs (dados grandes)
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- Tabela para checkpoint writes (operações de escrita)
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- =====================================================
-- STORES - MEMÓRIA DE LONGO PRAZO
-- =====================================================

-- Tabela principal para stores
CREATE TABLE IF NOT EXISTS store (
    namespace TEXT[] NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (namespace, key)
);

-- =====================================================
-- TABELAS ESPECÍFICAS DO METANALYST-AGENT
-- =====================================================

-- Meta-análises principais
CREATE TABLE IF NOT EXISTS meta_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    pico JSONB NOT NULL, -- {P, I, C, O}
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'paused')),
    quality_score FLOAT DEFAULT 0,
    total_articles INTEGER DEFAULT 0,
    processed_articles INTEGER DEFAULT 0,
    failed_articles INTEGER DEFAULT 0,
    final_report TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Artigos processados
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meta_analysis_id UUID REFERENCES meta_analyses(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    authors TEXT[],
    journal TEXT,
    publication_year INTEGER,
    doi TEXT,
    pmid TEXT,
    abstract TEXT,
    full_content TEXT,
    vancouver_citation TEXT,
    extracted_data JSONB DEFAULT '{}',
    quality_score FLOAT DEFAULT 0,
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    failure_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chunks vetorizados
CREATE TABLE IF NOT EXISTS article_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_vector FLOAT[] NOT NULL,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, chunk_index)
);

-- Análises estatísticas
CREATE TABLE IF NOT EXISTS statistical_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meta_analysis_id UUID REFERENCES meta_analyses(id) ON DELETE CASCADE,
    analysis_type TEXT NOT NULL, -- 'meta_analysis', 'forest_plot', 'funnel_plot', etc.
    results JSONB NOT NULL,
    plots JSONB DEFAULT '{}', -- Caminhos para arquivos de plots
    quality_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Log de execução dos agentes
CREATE TABLE IF NOT EXISTS agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meta_analysis_id UUID REFERENCES meta_analyses(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    input_data JSONB,
    output_data JSONB,
    execution_time_ms INTEGER,
    status TEXT NOT NULL CHECK (status IN ('started', 'completed', 'failed')),
    error_message TEXT,
    iteration_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cache de embeddings para otimização
CREATE TABLE IF NOT EXISTS embedding_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_hash TEXT UNIQUE NOT NULL,
    content_preview TEXT NOT NULL, -- Primeiros 200 chars para debug
    embedding_vector FLOAT[] NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- ÍNDICES PARA PERFORMANCE
-- =====================================================

-- Índices para checkpoints
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_thread_id ON checkpoint_blobs(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id);

-- Índices para stores
CREATE INDEX IF NOT EXISTS idx_store_namespace ON store USING GIN(namespace);
CREATE INDEX IF NOT EXISTS idx_store_namespace_key ON store(namespace, key);
CREATE INDEX IF NOT EXISTS idx_store_value ON store USING GIN(value);
CREATE INDEX IF NOT EXISTS idx_store_created_at ON store(created_at);

-- Índices para meta-análises
CREATE INDEX IF NOT EXISTS idx_meta_analyses_status ON meta_analyses(status);
CREATE INDEX IF NOT EXISTS idx_meta_analyses_created_at ON meta_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_meta_analyses_pico ON meta_analyses USING GIN(pico);

-- Índices para artigos
CREATE INDEX IF NOT EXISTS idx_articles_meta_analysis_id ON articles(meta_analysis_id);
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_doi ON articles(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_articles_pmid ON articles(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(processing_status);
CREATE INDEX IF NOT EXISTS idx_articles_quality ON articles(quality_score);
CREATE INDEX IF NOT EXISTS idx_articles_extracted_data ON articles USING GIN(extracted_data);

-- Índices para chunks (busca vetorial)
CREATE INDEX IF NOT EXISTS idx_article_chunks_article_id ON article_chunks(article_id);
-- Índice para busca por similaridade (usando extensão vector se disponível)
-- CREATE INDEX IF NOT EXISTS idx_article_chunks_embedding ON article_chunks USING ivfflat (embedding_vector vector_cosine_ops);

-- Índices para análises
CREATE INDEX IF NOT EXISTS idx_statistical_analyses_meta_analysis_id ON statistical_analyses(meta_analysis_id);
CREATE INDEX IF NOT EXISTS idx_statistical_analyses_type ON statistical_analyses(analysis_type);

-- Índices para logs
CREATE INDEX IF NOT EXISTS idx_agent_logs_meta_analysis_id ON agent_logs(meta_analysis_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_name ON agent_logs(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_logs_created_at ON agent_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_logs_status ON agent_logs(status);

-- Índice para cache de embeddings
CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON embedding_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model_name);

-- =====================================================
-- TRIGGERS PARA UPDATED_AT
-- =====================================================

-- Função para atualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para tabelas principais
CREATE TRIGGER update_meta_analyses_updated_at 
    BEFORE UPDATE ON meta_analyses 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_articles_updated_at 
    BEFORE UPDATE ON articles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_store_updated_at 
    BEFORE UPDATE ON store 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VIEWS ÚTEIS
-- =====================================================

-- View com estatísticas de meta-análises
CREATE OR REPLACE VIEW meta_analysis_stats AS
SELECT 
    ma.id,
    ma.title,
    ma.status,
    ma.quality_score,
    ma.total_articles,
    ma.processed_articles,
    ma.failed_articles,
    ROUND((ma.processed_articles::FLOAT / NULLIF(ma.total_articles, 0)) * 100, 2) as processing_percentage,
    COUNT(DISTINCT sa.id) as analysis_count,
    ma.created_at,
    ma.updated_at
FROM meta_analyses ma
LEFT JOIN statistical_analyses sa ON ma.id = sa.meta_analysis_id
GROUP BY ma.id, ma.title, ma.status, ma.quality_score, ma.total_articles, 
         ma.processed_articles, ma.failed_articles, ma.created_at, ma.updated_at;

-- View com performance dos agentes
CREATE OR REPLACE VIEW agent_performance AS
SELECT 
    agent_name,
    COUNT(*) as total_executions,
    COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
    ROUND(
        (COUNT(*) FILTER (WHERE status = 'completed')::FLOAT / COUNT(*)) * 100, 2
    ) as success_rate,
    AVG(execution_time_ms) as avg_execution_time_ms,
    MAX(execution_time_ms) as max_execution_time_ms,
    AVG(iteration_count) as avg_iterations
FROM agent_logs 
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY agent_name
ORDER BY success_rate DESC;

-- =====================================================
-- FUNÇÕES ÚTEIS
-- =====================================================

-- Função para limpar dados antigos
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS TEXT AS $$
DECLARE
    deleted_checkpoints INTEGER;
    deleted_logs INTEGER;
BEGIN
    -- Limpar checkpoints antigos
    DELETE FROM checkpoints 
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    GET DIAGNOSTICS deleted_checkpoints = ROW_COUNT;
    
    -- Limpar logs antigos
    DELETE FROM agent_logs 
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    GET DIAGNOSTICS deleted_logs = ROW_COUNT;
    
    RETURN FORMAT('Limpeza concluída: %s checkpoints, %s logs removidos', 
                  deleted_checkpoints, deleted_logs);
END;
$$ LANGUAGE plpgsql;

-- Função para estatísticas de uso do banco
CREATE OR REPLACE FUNCTION database_stats()
RETURNS TABLE(
    table_name TEXT,
    row_count BIGINT,
    size_mb NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        n_tup_ins - n_tup_del as row_count,
        ROUND((pg_total_relation_size(schemaname||'.'||tablename) / 1024.0 / 1024.0)::NUMERIC, 2) as size_mb
    FROM pg_stat_user_tables 
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- DADOS INICIAIS / CONFIGURAÇÃO
-- =====================================================

-- Inserir configurações padrão no store
INSERT INTO store (namespace, key, value) VALUES 
(ARRAY['config', 'system'], 'default_settings', '{
    "max_articles": 50,
    "quality_threshold": 0.8,
    "max_iterations": 5,
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "embedding_model": "text-embedding-3-small"
}'::jsonb)
ON CONFLICT (namespace, key) DO NOTHING;

INSERT INTO store (namespace, key, value) VALUES 
(ARRAY['config', 'prompts'], 'system_prompts', '{
    "orchestrator": "You are the central orchestrator of a meta-analysis system...",
    "researcher": "You are a Research Agent specialized in scientific literature search...",
    "processor": "You are a Processor Agent specialized in article extraction...",
    "analyst": "You are an Analyst Agent specialized in statistical meta-analysis...",
    "writer": "You are a Writer Agent specialized in generating meta-analysis reports...",
    "reviewer": "You are a Reviewer Agent specialized in quality assessment...",
    "editor": "You are an Editor Agent specialized in final report editing..."
}'::jsonb)
ON CONFLICT (namespace, key) DO NOTHING;

-- =====================================================
-- GRANTS E SEGURANÇA
-- =====================================================

-- Criar usuário para a aplicação (opcional)
-- CREATE USER metanalyst_app WITH PASSWORD 'secure_password_here';
-- GRANT CONNECT ON DATABASE metanalysis TO metanalyst_app;
-- GRANT USAGE ON SCHEMA public TO metanalyst_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO metanalyst_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO metanalyst_app;

-- =====================================================
-- COMANDOS ÚTEIS PARA MANUTENÇÃO
-- =====================================================

-- Para verificar tamanho das tabelas:
-- SELECT * FROM database_stats();

-- Para limpar dados antigos:
-- SELECT cleanup_old_data(30);

-- Para ver estatísticas das meta-análises:
-- SELECT * FROM meta_analysis_stats;

-- Para ver performance dos agentes:
-- SELECT * FROM agent_performance;

-- Para fazer backup:
-- pg_dump -h localhost -U postgres -d metanalysis > metanalysis_backup.sql

-- Para restaurar backup:
-- psql -h localhost -U postgres -d metanalysis < metanalysis_backup.sql

COMMIT;