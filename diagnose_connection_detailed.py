#!/usr/bin/env python3
"""
DIAGNÓSTICO DETALHADO DA CONEXÃO COM BANCO DE DADOS
================================================

Este script faz um diagnóstico completo da conexão para identificar
exatamente qual é o problema que está causando o erro "0".
"""

import os
import sys
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor

def test_direct_connection():
    """Testar conexão direta com o banco de dados"""
    print("🔍 TESTANDO CONEXÃO DIRETA COM POSTGRESQL")
    print("=" * 50)
    
    database_url = os.getenv("DATABASE_URL")
    print(f"DATABASE_URL: {database_url}")
    
    if not database_url:
        print("❌ DATABASE_URL não encontrada!")
        return False
    
    try:
        print("📡 Conectando...")
        conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        print("✅ Conexão estabelecida!")
        
        print("🧪 Testando query simples...")
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            print(f"✅ Query executada: {result}")
        
        print("📊 Testando tabela meta_analyses...")
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM meta_analyses")
            count = cursor.fetchone()[0]
            print(f"✅ Meta-analyses encontradas: {count}")
            
            # Buscar algumas meta-analyses
            cursor.execute("SELECT id, title FROM meta_analyses ORDER BY created_at DESC LIMIT 3")
            results = cursor.fetchall()
            print("📋 Meta-analyses recentes:")
            for row in results:
                print(f"   - {row['id']}: {row['title']}")
        
        conn.close()
        print("✅ Conexão fechada com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na conexão direta:")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensagem: {e}")
        print(f"   Traceback completo:")
        traceback.print_exc()
        return False

def test_database_manager():
    """Testar o DatabaseManager"""
    print("\n🔧 TESTANDO DATABASE MANAGER")
    print("=" * 50)
    
    try:
        # Adicionar o caminho do projeto
        sys.path.insert(0, '/Users/danielnobregamedeiros/Desktop/metanalyst-agent')
        
        from metanalyst_agent.database.connection import get_database_manager
        
        print("📦 Importação bem-sucedida")
        
        db = get_database_manager()
        print("🏗️ DatabaseManager instanciado")
        
        # Testar get_db_connection
        print("🔗 Testando get_db_connection...")
        with db.get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                print(f"✅ get_db_connection funcionando: {result}")
        
        # Testar execute_query
        print("🗃️ Testando execute_query...")
        results = db.execute_query("SELECT COUNT(*) FROM meta_analyses")
        print(f"✅ execute_query funcionando: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no DatabaseManager:")
        print(f"   Tipo: {type(e).__name__}")  
        print(f"   Mensagem: {e}")
        print(f"   Traceback completo:")
        traceback.print_exc()
        return False

def test_research_tools():
    """Testar as funções de research_tools"""
    print("\n🧪 TESTANDO RESEARCH TOOLS")
    print("=" * 50)
    
    try:
        sys.path.insert(0, '/Users/danielnobregamedeiros/Desktop/metanalyst-agent')
        
        from metanalyst_agent.tools.research_tools import ensure_valid_uuid, validate_uuid_format
        
        print("📦 Importação bem-sucedida")
        
        # Testar validate_uuid_format
        print("🔍 Testando validate_uuid_format...")
        valid_test = validate_uuid_format("550e8400-e29b-41d4-a716-446655440000")
        invalid_test = validate_uuid_format("Amiodarone_vs_BB_AF")
        print(f"✅ UUID válido: {valid_test}")
        print(f"✅ UUID inválido: {invalid_test}")
        
        # Testar ensure_valid_uuid com mais detalhes
        print("🛠️ Testando ensure_valid_uuid com UUID inválido...")
        try:
            result = ensure_valid_uuid("Amiodarone_vs_BB_AF", "test_detailed")
            print(f"✅ Resultado: {result}")
        except Exception as e:
            print(f"❌ Erro em ensure_valid_uuid:")
            print(f"   Tipo: {type(e).__name__}")
            print(f"   Mensagem: {e}")
            print(f"   Traceback completo:")
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro em research_tools:")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensagem: {e}")
        print(f"   Traceback completo:")
        traceback.print_exc()
        return False

def check_postgresql_status():
    """Verificar status do PostgreSQL"""
    print("\n🐘 VERIFICANDO STATUS DO POSTGRESQL")
    print("=" * 50)
    
    # Verificar se o PostgreSQL está rodando
    import subprocess
    
    try:
        result = subprocess.run(['brew', 'services', 'list'], capture_output=True, text=True)
        if 'postgresql' in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'postgresql' in line:
                    print(f"🗃️ PostgreSQL status: {line}")
        else:
            print("❌ PostgreSQL não encontrado nos services do brew")
    except Exception as e:
        print(f"❌ Erro verificando services: {e}")
    
    # Testar conexão direta com psql
    try:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Extrair parâmetros da URL
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            
            cmd = [
                'psql', 
                f'-h{parsed.hostname}', 
                f'-p{parsed.port}', 
                f'-U{parsed.username}', 
                f'-d{parsed.path[1:]}',  # Remove leading /
                '-c', 'SELECT version();'
            ]
            
            print(f"📡 Testando conexão com psql...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ psql conectou com sucesso!")
                print(f"📊 Versão: {result.stdout.strip()}")
            else:
                print(f"❌ psql falhou:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
    
    except Exception as e:
        print(f"❌ Erro testando psql: {e}")

def main():
    """Função principal"""
    print("🔬 DIAGNÓSTICO DETALHADO DA CONEXÃO COM BANCO")
    print("=" * 60)
    
    # 1. Verificar PostgreSQL
    check_postgresql_status()
    
    # 2. Testar conexão direta
    direct_ok = test_direct_connection()
    
    # 3. Testar DatabaseManager  
    manager_ok = test_database_manager()
    
    # 4. Testar research_tools
    tools_ok = test_research_tools()
    
    print(f"\n📊 RESUMO DO DIAGNÓSTICO")
    print("=" * 60)
    print(f"Conexão direta: {'✅ OK' if direct_ok else '❌ FALHOU'}")
    print(f"DatabaseManager: {'✅ OK' if manager_ok else '❌ FALHOU'}")
    print(f"Research Tools: {'✅ OK' if tools_ok else '❌ FALHOU'}")
    
    if direct_ok and manager_ok and tools_ok:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("O problema deve estar em outro lugar.")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM!")
        print("Verifique os detalhes acima para identificar o problema.")

if __name__ == "__main__":
    main()
