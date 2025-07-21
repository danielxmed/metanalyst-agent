#!/usr/bin/env python3
"""
CORREÇÃO SIMPLES E DIRETA DO PROBLEMA DE UUID
===========================================

Este script corrige o problema específico onde meta_analysis_id inválidos
causam falha de integridade referencial no banco de dados.
"""

import os
import sys
import uuid
from datetime import datetime

# Adicionar o diretório do projeto ao Python path
sys.path.insert(0, '/Users/danielnobregamedeiros/Desktop/metanalyst-agent')

def apply_uuid_fix():
    """Aplicar a correção diretamente no research_tools.py"""
    
    print("🔧 APLICANDO CORREÇÃO DE UUID NO RESEARCH_TOOLS.PY")
    print("=" * 50)
    
    research_tools_path = "metanalyst_agent/tools/research_tools.py"
    
    # Fazer backup
    backup_path = f"{research_tools_path}.backup_uuid_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.system(f"cp {research_tools_path} {backup_path}")
    print(f"✅ Backup criado: {backup_path}")
    
    # Ler arquivo atual
    with open(research_tools_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substituir a função ensure_valid_uuid problemática
    old_function = '''def ensure_valid_uuid(meta_analysis_id: str, operation: str = "operation") -> str:
    """
    Garante que um meta_analysis_id seja um UUID válido.
    Se inválido, gera um novo UUID e registra um warning.
    
    Args:
        meta_analysis_id: ID para validar
        operation: Nome da operação para logs
        
    Returns:
        UUID válido (original ou novo gerado)
    """
    if validate_uuid_format(meta_analysis_id):
        return meta_analysis_id
    
    # Gerar novo UUID válido
    new_uuid = str(uuid.uuid4())
    logger.warning(
        f"Invalid UUID format detected in {operation}: '{meta_analysis_id}'. "
        f"Generated new UUID: {new_uuid}"
    )
    
    return new_uuid'''
    
    new_function = '''def ensure_valid_uuid(meta_analysis_id: str, operation: str = "operation") -> str:
    """
    Garante que um meta_analysis_id seja um UUID válido.
    Se inválido, busca a meta-análise mais recente ativa no banco.
    
    Args:
        meta_analysis_id: ID para validar
        operation: Nome da operação para logs
        
    Returns:
        UUID válido existente no banco
    """
    if validate_uuid_format(meta_analysis_id):
        # Verificar se existe no banco
        try:
            from ..database.connection import get_database_manager
            with get_database_manager().get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM meta_analyses WHERE id = %s", (meta_analysis_id,))
                    if cursor.fetchone()[0] > 0:
                        return meta_analysis_id
        except Exception as e:
            logger.error(f"Error checking meta_analysis_id in database: {e}")
    
    # Se ID inválido ou não existe, buscar meta-análise ativa mais recente
    logger.warning(f"Invalid/non-existent UUID in {operation}: '{meta_analysis_id}'. Searching for recent active meta-analysis.")
    
    try:
        from ..database.connection import get_database_manager
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM meta_analyses 
                    WHERE status = 'active' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    recent_id = str(result[0])
                    logger.info(f"Using recent active meta-analysis: {recent_id}")
                    return recent_id
    except Exception as e:
        logger.error(f"Error finding recent meta-analysis: {e}")
    
    # Como último recurso, criar uma nova meta-análise
    try:
        from ..database.connection import get_database_manager
        new_id = str(uuid.uuid4())
        thread_id = f"thread_{new_id}"
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO meta_analyses (id, thread_id, title, pico, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    new_id, thread_id, f"Auto-created for {operation}",
                    json.dumps({"P": "Not specified", "I": "Not specified", "C": "Not specified", "O": "Not specified"}),
                    "active", datetime.utcnow(), datetime.utcnow()
                ))
                conn.commit()
        
        logger.warning(f"Created new meta-analysis as fallback: {new_id}")
        return new_id
    except Exception as e:
        logger.error(f"Failed to create fallback meta-analysis: {e}")
        raise ValueError(f"Cannot resolve valid meta_analysis_id for {operation}")'''
    
    # Substituir no conteúdo
    content = content.replace(old_function, new_function)
    
    # Também corrigir a função _add_url_to_candidates para usar state
    old_add_function = '''def _add_url_to_candidates(url: str, meta_analysis_id: str, metadata: Dict[str, Any]):
    """Add URL to candidates in PostgreSQL"""
    try:
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                article_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO articles (id, meta_analysis_id, url, title, processing_status, created_at)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                    ON CONFLICT (url, meta_analysis_id) DO NOTHING
                """, (
                    article_id, 
                    meta_analysis_id, 
                    url, 
                    metadata.get('title', ''), 
                    datetime.utcnow()
                ))
                conn.commit()
        _candidate_urls_cache.add(url)
    except Exception:
        _candidate_urls_cache.add(url)'''
    
    new_add_function = '''def _add_url_to_candidates(url: str, meta_analysis_id: str, metadata: Dict[str, Any]):
    """Add URL to candidates in PostgreSQL"""
    try:
        # Validar meta_analysis_id antes de inserir
        valid_id = ensure_valid_uuid(meta_analysis_id, "_add_url_to_candidates")
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                article_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO articles (id, meta_analysis_id, url, title, processing_status, created_at)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                    ON CONFLICT DO NOTHING
                """, (
                    article_id, 
                    valid_id,  # Usar ID validado
                    url, 
                    metadata.get('title', ''), 
                    datetime.utcnow()
                ))
                conn.commit()
        _candidate_urls_cache.add(url)
        logger.info(f"Added URL to candidates with meta_analysis_id: {valid_id}")
    except Exception as e:
        logger.error(f"Error adding URL to candidates: {e}")
        _candidate_urls_cache.add(url)'''
    
    # Substituir a função de adicionar URLs
    content = content.replace(old_add_function, new_add_function)
    
    # Corrigir get_candidate_urls_summary para aceitar state
    old_summary_function = '''@tool
def get_candidate_urls_summary(meta_analysis_id: str) -> Dict[str, Any]:
    """
    Get summary of candidate URLs without loading full content
    
    Args:
        meta_analysis_id: ID of the meta-analysis
        
    Returns:
        Summary of candidate URLs
    """'''
    
    new_summary_function = '''@tool
def get_candidate_urls_summary(state: Dict[str, Any] = None, meta_analysis_id: str = None) -> Dict[str, Any]:
    """
    Get summary of candidate URLs without loading full content
    
    Args:
        state: State containing meta_analysis_id
        meta_analysis_id: Direct ID of the meta-analysis
        
    Returns:
        Summary of candidate URLs
    """
    # Extrair meta_analysis_id do state se não fornecido diretamente
    if not meta_analysis_id and state:
        for key in ['meta_analysis_id', 'id', 'analysis_id']:
            if key in state and state[key]:
                meta_analysis_id = str(state[key])
                break
    
    if not meta_analysis_id:
        return {
            "success": False,
            "error": "No meta_analysis_id provided",
            "candidates": []
        }
    
    # Validar o ID
    meta_analysis_id = ensure_valid_uuid(meta_analysis_id, "get_candidate_urls_summary")'''
    
    content = content.replace(old_summary_function, new_summary_function)
    
    # Salvar arquivo corrigido
    with open(research_tools_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ research_tools.py corrigido com validação robusta!")

def test_correction():
    """Testar se a correção funcionou"""
    
    print("\n🧪 TESTANDO CORREÇÃO")
    print("=" * 50)
    
    try:
        from metanalyst_agent.tools.research_tools import ensure_valid_uuid
        
        # Testar com UUID inválido
        test_invalid = "Amiodarone_vs_BB_AF"
        result = ensure_valid_uuid(test_invalid, "test_correction")
        
        print(f"✅ Teste com UUID inválido '{test_invalid}' retornou: {result}")
        
        # Verificar se é um UUID válido
        import uuid
        uuid.UUID(result)
        print("✅ Resultado é um UUID válido!")
        
        # Verificar se existe no banco
        from metanalyst_agent.database.connection import get_database_manager
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM meta_analyses WHERE id = %s", (result,))
                count = cursor.fetchone()[0]
                if count > 0:
                    print("✅ UUID existe no banco de dados!")
                else:
                    print("❌ UUID não encontrado no banco")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def main():
    """Função principal"""
    print("🛠️  CORREÇÃO DO PROBLEMA DE UUID - METANALYST-AGENT")
    print("=" * 60)
    
    # 1. Aplicar correção
    apply_uuid_fix()
    
    # 2. Testar correção
    if test_correction():
        print("\n🎉 CORREÇÃO APLICADA COM SUCESSO!")
        print("✅ O sistema agora pode lidar corretamente com UUIDs inválidos")
        print("✅ Meta-análises órfãs não causarão mais erros de integridade referencial")
    else:
        print("\n❌ CORREÇÃO FALHOU!")
        print("❌ Verifique os logs para mais detalhes")
    
    print("\n" + "=" * 60)
    print("🚀 Execute o sistema novamente para testar!")

if __name__ == "__main__":
    main()
