#!/usr/bin/env python3
"""
Teste rápido para verificar se o sistema está configurado corretamente.
"""

import os
import sys


def test_imports():
    """Testa se todas as importações funcionam."""
    print("🔍 Testando importações...")
    
    try:
        import metanalyst_agent
        print("✅ metanalyst_agent importado")
        
        from metanalyst_agent.config.settings import settings
        print("✅ Configurações carregadas")
        
        from metanalyst_agent.agents import supervisor_agent
        print("✅ Supervisor agent importado")
        
        from metanalyst_agent.tools.research_tools import search_pubmed
        print("✅ Ferramentas de pesquisa importadas")
        
        print("✅ Todas as importações funcionaram!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na importação: {e}")
        return False


def test_environment():
    """Testa variáveis de ambiente."""
    print("\n🔍 Testando variáveis de ambiente...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY", 
        "POSTGRES_URL"
    ]
    
    missing = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} configurada")
        else:
            print(f"❌ {var} não encontrada")
            missing.append(var)
    
    if missing:
        print(f"\n❌ Configure as variáveis em falta: {missing}")
        return False
    else:
        print("✅ Todas as variáveis de ambiente configuradas!")
        return True


def test_database_connection():
    """Testa conexão com PostgreSQL."""
    print("\n🔍 Testando conexão PostgreSQL...")
    
    try:
        from metanalyst_agent.config.database import get_checkpointer
        checkpointer = get_checkpointer()
        print("✅ Checkpointer PostgreSQL criado")
        
        from metanalyst_agent.config.database import get_store
        store = get_store()
        print("✅ Store PostgreSQL criado")
        
        print("✅ Conexão PostgreSQL funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na conexão PostgreSQL: {e}")
        print("Verifique se o PostgreSQL está rodando:")
        print("docker run --name metanalyst-postgres -e POSTGRES_DB=metanalysis -e POSTGRES_USER=metanalyst -e POSTGRES_PASSWORD=secure_password -p 5432:5432 -d postgres:15")
        return False


def main():
    """Executa todos os testes."""
    print("🧪 TESTE DE CONFIGURAÇÃO - METANALYST-AGENT")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_database_connection
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("O sistema está pronto para uso.")
        print("\nExecute: python run_example.py")
        return 0
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
        print("Corrija os problemas antes de usar o sistema.")
        return 1


if __name__ == "__main__":
    sys.exit(main())