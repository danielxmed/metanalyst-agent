#!/usr/bin/env python3
"""
Script para limpar UUIDs inválidos do banco de dados PostgreSQL.
Este script corrige o problema de meta_analysis_id com formato incorreto como "AF_amiodarone_beta".
"""

import os
import sys
import uuid
import re
import psycopg2
from pathlib import Path
from typing import List, Dict, Any

# Adicionar o diretório do projeto ao Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from metanalyst_agent.database.connection import get_database_manager

def validate_uuid(uuid_string: str) -> bool:
    """
    Valida se uma string é um UUID válido.
    
    Args:
        uuid_string: String para validar
        
    Returns:
        True se é um UUID válido, False caso contrário
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def find_invalid_uuids(db_manager) -> Dict[str, List[Dict[str, Any]]]:
    """
    Encontra todos os registros com UUIDs inválidos no banco.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        Dicionário com tabelas e registros inválidos
    """
    invalid_records = {}
    
    # Pattern para UUID válido
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    try:
        with db_manager.get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Verificar tabela articles
                cursor.execute("""
                    SELECT meta_analysis_id, url, title, created_at
                    FROM articles 
                    WHERE meta_analysis_id !~ %s
                    ORDER BY created_at DESC
                """, (uuid_pattern,))
                
                articles = cursor.fetchall()
                if articles:
                    invalid_records['articles'] = [
                        {
                            'meta_analysis_id': row[0],
                            'url': row[1], 
                            'title': row[2],
                            'created_at': row[3]
                        }
                        for row in articles
                    ]
                
                # Verificar outras tabelas que possam existir
                tables_to_check = [
                    ('meta_analyses', 'id'),
                    ('statistical_analyses', 'meta_analysis_id'),
                    ('agent_logs', 'meta_analysis_id')
                ]
                
                for table_name, column_name in tables_to_check:
                    try:
                        cursor.execute(f"""
                            SELECT {column_name}, COUNT(*)
                            FROM {table_name}
                            WHERE {column_name}::text !~ %s
                            GROUP BY {column_name}
                        """, (uuid_pattern,))
                        
                        results = cursor.fetchall()
                        if results:
                            invalid_records[table_name] = [
                                {'id': row[0], 'count': row[1]}
                                for row in results
                            ]
                    except psycopg2.Error:
                        # Tabela não existe, pular
                        continue
                        
    except Exception as e:
        print(f"Erro ao buscar UUIDs inválidos: {e}")
        
    return invalid_records

def backup_invalid_data(db_manager) -> bool:
    """
    Faz backup dos dados inválidos antes de deletá-los.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        True se backup foi bem-sucedido
    """
    try:
        with db_manager.get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Criar tabela de backup
                cursor.execute("""
                    DROP TABLE IF EXISTS invalid_articles_backup;
                    CREATE TABLE invalid_articles_backup AS
                    SELECT * FROM articles 
                    WHERE meta_analysis_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
                """)
                
                # Verificar quantos registros foram salvos
                cursor.execute("SELECT COUNT(*) FROM invalid_articles_backup")
                backup_count = cursor.fetchone()[0]
                
                conn.commit()
                print(f"✅ Backup criado com {backup_count} registros inválidos")
                return True
                
    except Exception as e:
        print(f"❌ Erro ao criar backup: {e}")
        return False

def cleanup_invalid_uuids(db_manager) -> bool:
    """
    Remove registros com UUIDs inválidos do banco.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        True se limpeza foi bem-sucedida
    """
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    try:
        with db_manager.get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Contar registros antes da limpeza
                cursor.execute("SELECT COUNT(*) FROM articles WHERE meta_analysis_id !~ %s", (uuid_pattern,))
                invalid_count = cursor.fetchone()[0]
                
                if invalid_count == 0:
                    print("✅ Nenhum UUID inválido encontrado!")
                    return True
                
                print(f"🔍 Encontrados {invalid_count} registros com UUIDs inválidos")
                
                # Fazer backup primeiro
                if not backup_invalid_data(db_manager):
                    print("❌ Falha no backup, abortando limpeza")
                    return False
                
                # Deletar registros inválidos
                cursor.execute("DELETE FROM articles WHERE meta_analysis_id !~ %s", (uuid_pattern,))
                deleted_count = cursor.rowcount
                
                # Tentar deletar de outras tabelas
                tables_to_clean = [
                    'statistical_analyses',
                    'agent_logs', 
                    'meta_analyses'
                ]
                
                for table_name in tables_to_clean:
                    try:
                        if table_name == 'meta_analyses':
                            cursor.execute(f"DELETE FROM {table_name} WHERE id::text !~ %s", (uuid_pattern,))
                        else:
                            cursor.execute(f"DELETE FROM {table_name} WHERE meta_analysis_id::text !~ %s", (uuid_pattern,))
                    except psycopg2.Error:
                        # Tabela não existe ou erro, continuar
                        continue
                
                conn.commit()
                print(f"✅ Removidos {deleted_count} registros inválidos da tabela articles")
                
                # Verificar se a limpeza foi bem-sucedida
                cursor.execute("SELECT COUNT(*) FROM articles WHERE meta_analysis_id !~ %s", (uuid_pattern,))
                remaining_invalid = cursor.fetchone()[0]
                
                if remaining_invalid == 0:
                    print("✅ Limpeza completa! Todos os UUIDs agora são válidos")
                    return True
                else:
                    print(f"⚠️  Ainda restam {remaining_invalid} UUIDs inválidos")
                    return False
                
    except Exception as e:
        print(f"❌ Erro durante limpeza: {e}")
        return False

def add_uuid_validation(db_manager) -> bool:
    """
    Adiciona constraints para validar UUIDs no futuro.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        True se validação foi adicionada com sucesso
    """
    try:
        with db_manager.get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Adicionar constraint de validação UUID na tabela articles
                cursor.execute("""
                    ALTER TABLE articles 
                    DROP CONSTRAINT IF EXISTS valid_meta_analysis_id_format;
                    
                    ALTER TABLE articles 
                    ADD CONSTRAINT valid_meta_analysis_id_format 
                    CHECK (meta_analysis_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$');
                """)
                
                # Tentar adicionar em outras tabelas também
                other_tables = [
                    ('statistical_analyses', 'meta_analysis_id'),
                    ('agent_logs', 'meta_analysis_id'),
                    ('meta_analyses', 'id')
                ]
                
                for table_name, column_name in other_tables:
                    try:
                        constraint_name = f"valid_{column_name}_format"
                        cursor.execute(f"""
                            ALTER TABLE {table_name} 
                            DROP CONSTRAINT IF EXISTS {constraint_name};
                            
                            ALTER TABLE {table_name} 
                            ADD CONSTRAINT {constraint_name}
                            CHECK ({column_name}::text ~ '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$');
                        """)
                    except psycopg2.Error:
                        # Tabela não existe, pular
                        continue
                
                conn.commit()
                print("✅ Constraints de validação UUID adicionadas")
                return True
                
    except Exception as e:
        print(f"❌ Erro ao adicionar validação: {e}")
        return False

def main():
    """Função principal do script de limpeza."""
    print("🔧 Iniciando limpeza de UUIDs inválidos no banco de dados...")
    
    try:
        # Verificar DATABASE_URL
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL não está configurado!")
            return False
        
        # Inicializar database manager
        db_manager = get_database_manager()
        
        # 1. Encontrar registros inválidos
        print("\n1️⃣ Verificando UUIDs inválidos...")
        invalid_records = find_invalid_uuids(db_manager)
        
        if not invalid_records:
            print("✅ Nenhum UUID inválido encontrado!")
            return True
        
        # 2. Mostrar registros inválidos encontrados
        print("\n📋 Registros inválidos encontrados:")
        for table_name, records in invalid_records.items():
            print(f"   {table_name}: {len(records)} registros")
            if table_name == 'articles':
                for record in records[:3]:  # Mostrar apenas os primeiros 3
                    print(f"     - ID: {record['meta_analysis_id']}")
                    print(f"       URL: {record['url'][:50]}...")
                if len(records) > 3:
                    print(f"     ... e mais {len(records) - 3} registros")
        
        # 3. Confirmar limpeza
        response = input(f"\n❓ Deseja remover {sum(len(records) for records in invalid_records.values())} registros inválidos? (s/N): ")
        if response.lower() not in ['s', 'sim', 'y', 'yes']:
            print("❌ Limpeza cancelada pelo usuário")
            return False
        
        # 4. Executar limpeza
        print("\n2️⃣ Executando limpeza...")
        if cleanup_invalid_uuids(db_manager):
            print("✅ Limpeza concluída com sucesso!")
        else:
            print("❌ Falha na limpeza")
            return False
        
        # 5. Adicionar validação para prevenir problema futuro
        print("\n3️⃣ Adicionando validação de UUID...")
        if add_uuid_validation(db_manager):
            print("✅ Validação adicionada com sucesso!")
        else:
            print("⚠️  Falha ao adicionar validação, mas limpeza foi bem-sucedida")
        
        print("\n🎉 Processo completo! Seu banco agora deve aceitar apenas UUIDs válidos.")
        return True
        
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
