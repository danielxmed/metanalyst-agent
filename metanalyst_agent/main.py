"""
Arquivo principal para execução do sistema metanalyst-agent.
"""

import argparse
import sys
from datetime import datetime
from .graph.multi_agent_graph import run_meta_analysis, visualize_graph
from .models.config import config

def main():
    """Função principal do sistema metanalyst-agent."""
    
    parser = argparse.ArgumentParser(
        description="Metanalyst-Agent: Sistema automatizado de meta-análise médica"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query de meta-análise a ser executada"
    )
    
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Usar persistência em memória (desenvolvimento)"
    )
    
    parser.add_argument(
        "--thread-id",
        type=str,
        help="ID da thread para continuação de sessão"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Gerar visualização do grafo"
    )
    
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Mostrar exemplos de queries"
    )
    
    args = parser.parse_args()
    
    # Mostrar exemplos
    if args.examples:
        show_examples()
        return
    
    # Visualizar grafo
    if args.visualize:
        print("Gerando visualização do grafo...")
        visualize_graph()
        return
    
    # Verificar se query foi fornecida
    if not args.query:
        print("❌ Erro: Query de meta-análise é obrigatória")
        print("Use --help para ver opções disponíveis")
        print("Use --examples para ver exemplos de queries")
        sys.exit(1)
    
    # Executar meta-análise
    print("🚀 Iniciando sistema metanalyst-agent...")
    print(f"📋 Query: {args.query}")
    print(f"💾 Modo: {'Memória' if args.memory else 'PostgreSQL'}")
    print(f"🔗 Thread ID: {args.thread_id or 'Auto-gerado'}")
    print("=" * 80)
    
    try:
        # Executar análise
        for i, result in enumerate(run_meta_analysis(
            user_query=args.query,
            thread_id=args.thread_id,
            use_memory=args.memory
        )):
            
            # Verificar se há erro
            if "error" in result:
                print(f"❌ Erro: {result['error']}")
                break
            
            # Mostrar progresso
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                agent = result.get("current_agent", "system")
                phase = result.get("current_phase", "unknown")
                
                print(f"\n[{i+1}] 🤖 {agent.upper()} ({phase})")
                print(f"💬 {last_message.get('content', '')[:200]}...")
                
                # Mostrar métricas quando disponíveis
                if result.get("processed_articles"):
                    print(f"📚 Artigos processados: {len(result['processed_articles'])}")
                
                if result.get("statistical_analysis"):
                    analysis = result["statistical_analysis"]
                    if isinstance(analysis, dict):
                        effect_size = analysis.get("pooled_effect_size", "N/A")
                        print(f"📊 Effect size: {effect_size}")
                
                if result.get("final_report"):
                    print("✅ Relatório final disponível!")
                    
                print("-" * 40)
        
        print("\n✅ Meta-análise concluída!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

def show_examples():
    """Mostra exemplos de queries válidas."""
    
    examples = [
        {
            "title": "Meta-análise de intervenções farmacológicas",
            "query": """Realize uma meta-análise sobre a eficácia da metformina 
            versus placebo para prevenção de diabetes tipo 2 em adultos com 
            pré-diabetes. Inclua análise de heterogeneidade e forest plot."""
        },
        {
            "title": "Meta-análise de intervenções psicológicas",
            "query": """Conduza uma meta-análise comparando terapia cognitivo-comportamental 
            versus mindfulness para tratamento de depressão em adultos. 
            Avalie qualidade da evidência e heterogeneidade."""
        },
        {
            "title": "Meta-análise de intervenções cirúrgicas",
            "query": """Faça uma meta-análise comparando cirurgia laparoscópica 
            versus cirurgia aberta para apendicectomia. Foque em tempo de 
            recuperação e complicações pós-operatórias."""
        },
        {
            "title": "Meta-análise de diagnóstico",
            "query": """Realize uma meta-análise sobre a acurácia diagnóstica 
            da ressonância magnética versus tomografia computadorizada para 
            detecção de AVC isquêmico agudo."""
        },
        {
            "title": "Meta-análise de fatores de risco",
            "query": """Conduza uma meta-análise sobre a associação entre 
            consumo de álcool e risco de câncer de mama em mulheres. 
            Calcule odds ratio pooled e intervalos de confiança."""
        }
    ]
    
    print("📚 EXEMPLOS DE QUERIES PARA METANALYST-AGENT")
    print("=" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   Query: {example['query'].strip()}")
    
    print("\n💡 DICAS:")
    print("- Seja específico sobre Population, Intervention, Comparison, Outcome (PICO)")
    print("- Mencione tipos de análise desejados (forest plot, heterogeneidade, etc.)")
    print("- Especifique população-alvo quando relevante")
    print("- Indique medidas de efeito de interesse (OR, RR, MD, SMD)")

def run_interactive():
    """Executa o sistema em modo interativo."""
    
    print("🤖 METANALYST-AGENT - Modo Interativo")
    print("Digite 'quit' para sair")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n📋 Digite sua query de meta-análise: ").strip()
            
            if query.lower() in ['quit', 'exit', 'sair']:
                print("👋 Encerrando sistema...")
                break
            
            if not query:
                print("⚠️ Query vazia. Tente novamente.")
                continue
            
            print(f"\n🚀 Executando: {query[:100]}...")
            
            # Executar análise
            for result in run_meta_analysis(query, use_memory=True):
                if "error" in result:
                    print(f"❌ Erro: {result['error']}")
                    break
                
                # Mostrar apenas resultado final
                if result.get("final_report"):
                    print("✅ Meta-análise concluída!")
                    break
        
        except KeyboardInterrupt:
            print("\n👋 Encerrando sistema...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    # Verificar se deve executar modo interativo
    if len(sys.argv) == 1:
        run_interactive()
    else:
        main()