#!/usr/bin/env python3
"""
Script de exemplo para demonstrar o uso do Metanalyst-Agent.
Execute após configurar as variáveis de ambiente e PostgreSQL.
"""

import os
from metanalyst_agent import MetanalystAgent


def main():
    """Executa exemplo de meta-análise."""
    
    print("🧬 EXEMPLO - METANALYST-AGENT")
    print("=" * 50)
    
    # Verificar variáveis de ambiente
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "POSTGRES_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Variáveis de ambiente em falta:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure as variáveis no arquivo .env")
        return
    
    print("✅ Variáveis de ambiente configuradas")
    print()
    
    # Inicializar sistema
    try:
        agent = MetanalystAgent()
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        print("Verifique se o PostgreSQL está rodando:")
        print("docker run --name metanalyst-postgres -e POSTGRES_DB=metanalysis -e POSTGRES_USER=metanalyst -e POSTGRES_PASSWORD=secure_password -p 5432:5432 -d postgres:15")
        return
    
    # Exemplo 1: Mindfulness vs CBT para ansiedade
    print("🔬 EXEMPLO 1: Mindfulness vs CBT para Ansiedade")
    print("-" * 50)
    
    query1 = """
    Realize uma meta-análise comparando a eficácia da meditação mindfulness 
    versus terapia cognitivo-comportamental (CBT) para redução de sintomas 
    de ansiedade em adultos. Inclua forest plot e análise de heterogeneidade.
    """
    
    results1 = agent.run(query1, thread_id="example_mindfulness_cbt")
    
    if results1["success"]:
        print("✅ Meta-análise 1 concluída!")
        print(f"📄 Relatório: {results1.get('final_report_path', 'N/A')}")
    else:
        print(f"❌ Erro: {results1.get('error', 'Erro desconhecido')}")
    
    print("\n" + "=" * 50)
    
    # Exemplo 2: Exercício vs medicação para depressão
    print("🔬 EXEMPLO 2: Exercício vs Medicação para Depressão")
    print("-" * 50)
    
    query2 = """
    Conduza uma meta-análise sobre a eficácia do exercício físico comparado 
    a antidepressivos para tratamento de depressão maior em adultos. 
    Foque em estudos randomizados controlados.
    """
    
    results2 = agent.run(query2, thread_id="example_exercise_depression")
    
    if results2["success"]:
        print("✅ Meta-análise 2 concluída!")
        print(f"📄 Relatório: {results2.get('final_report_path', 'N/A')}")
    else:
        print(f"❌ Erro: {results2.get('error', 'Erro desconhecido')}")
    
    print("\n" + "=" * 50)
    print("🎉 Exemplos concluídos!")
    print("\nVerifique os relatórios gerados na pasta 'reports/'")
    print("Os gráficos estão na pasta 'plots/'")


if __name__ == "__main__":
    main()