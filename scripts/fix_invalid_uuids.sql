-- Script para corrigir UUIDs inválidos no banco de dados
-- Remove registros com meta_analysis_id que não são UUIDs válidos

-- 1. Verificar registros com UUIDs inválidos
SELECT 
    table_name,
    COUNT(*) as invalid_count
FROM (
    -- Verificar tabela articles
    SELECT 'articles' as table_name, meta_analysis_id
    FROM articles 
    WHERE meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    UNION ALL
    
    -- Verificar tabela meta_analyses se existir
    SELECT 'meta_analyses' as table_name, id::text as meta_analysis_id
    FROM meta_analyses 
    WHERE id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    UNION ALL
    
    -- Verificar tabela statistical_analyses se existir
    SELECT 'statistical_analyses' as table_name, meta_analysis_id::text
    FROM statistical_analyses 
    WHERE meta_analysis_id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    UNION ALL
    
    -- Verificar tabela agent_logs se existir
    SELECT 'agent_logs' as table_name, meta_analysis_id::text
    FROM agent_logs 
    WHERE meta_analysis_id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
) invalid_records
GROUP BY table_name;

-- 2. Mostrar registros específicos que serão removidos
SELECT 'INVALID RECORDS TO BE DELETED:' as notice;

SELECT 
    'articles' as table_name,
    meta_analysis_id,
    url,
    title,
    created_at
FROM articles 
WHERE meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
ORDER BY created_at DESC
LIMIT 10;

-- 3. Fazer backup dos dados inválidos (opcional)
CREATE TABLE IF NOT EXISTS invalid_articles_backup AS
SELECT * FROM articles 
WHERE meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

-- 4. Deletar registros com UUIDs inválidos
BEGIN;

-- Deletar de articles
DELETE FROM articles 
WHERE meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

-- Deletar de outras tabelas se existirem
DELETE FROM statistical_analyses 
WHERE meta_analysis_id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

DELETE FROM agent_logs 
WHERE meta_analysis_id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

DELETE FROM meta_analyses 
WHERE id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

COMMIT;

-- 5. Verificar que a limpeza foi bem-sucedida
SELECT 'CLEANUP VERIFICATION:' as notice;

SELECT 
    'articles' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' THEN 1 END) as invalid_records
FROM articles;

-- 6. Recriar índices se necessário
REINDEX TABLE articles;

SELECT 'UUID cleanup completed successfully!' as result;
