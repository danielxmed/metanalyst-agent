#!/usr/bin/env python3
"""
Script de exemplo para executar o sistema metanalyst-agent.
"""

import os
import sys
from datetime import datetime

# Adicionar o diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metanalyst_agent import run_meta_analysis

def main():
    """Executa exemplo de meta-análise."""
    
    print("🤖 METANALYST-AGENT - Exemplo de Execução")
    print("=" * 60)
    
    # Query de exemplo
    example_query = """
    Realize uma meta-análise sobre a eficácia da meditação mindfulness 
    versus terapia cognitivo-comportamental para tratamento de ansiedade 
    em adultos. Inclua forest plot e análise de heterogeneidade.
    """
    
    print(f"📋 Query: {example_query.strip()}")
    print(f"⏰ Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Iniciando processamento...")
    print("-" * 60)
    
    try:
        # Executar meta-análise
        for i, result in enumerate(run_meta_analysis(
            user_query=example_query,
            use_memory=True  # Usar modo de desenvolvimento
        )):
            
            # Verificar se há erro
            if "error" in result:
                print(f"❌ Erro: {result['error']}")
                return
            
            # Mostrar progresso
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                agent = result.get("current_agent", "system")
                phase = result.get("current_phase", "unknown")
                
                print(f"\n[{i+1}] 🤖 {agent.upper()} ({phase})")
                print(f"💬 {last_message.get('content', '')[:150]}...")
                
                # Mostrar métricas quando disponíveis
                if result.get("candidate_urls"):
                    print(f"🔍 URLs encontradas: {len(result['candidate_urls'])}")
                
                if result.get("processed_articles"):
                    print(f"📚 Artigos processados: {len(result['processed_articles'])}")
                
                if result.get("vector_store_id"):
                    print(f"🎯 Vector store: {result['vector_store_id']}")
                
                if result.get("statistical_analysis"):
                    analysis = result["statistical_analysis"]
                    if isinstance(analysis, dict):
                        effect_size = analysis.get("pooled_effect_size", "N/A")
                        print(f"📊 Effect size: {effect_size}")
                
                if result.get("final_report"):
                    print("✅ Relatório final disponível!")
                    print(f"📄 Tamanho: {len(result['final_report'])} caracteres")
                    
                    # Salvar relatório
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"meta_analysis_report_{timestamp}.html"
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(result["final_report"])
                    
                    print(f"💾 Relatório salvo: {filename}")
                    return  # Finalizar
                    
                print("-" * 40)
        
        print("\n✅ Meta-análise concluída!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Verificar se as variáveis de ambiente estão configuradas
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Variáveis de ambiente faltando:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure as variáveis e tente novamente.")
        print("Exemplo:")
        print("export OPENAI_API_KEY='sua_chave_aqui'")
        print("export TAVILY_API_KEY='sua_chave_aqui'")
        sys.exit(1)
    
    main()