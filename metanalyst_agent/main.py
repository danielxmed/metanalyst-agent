"""
Metanalyst-Agent - Sistema Multi-Agente Autônomo para Meta-análises
Primeiro projeto open-source da Nobrega Medtech
"""

import uuid
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .config.database import sync_init_database
from .graph.multi_agent_graph import build_meta_analysis_graph, create_meta_analysis_config


class MetanalystAgent:
    """
    Interface principal do sistema Metanalyst-Agent.
    
    Sistema multi-agente autônomo que realiza meta-análises completas
    desde a busca de literatura até a geração de relatórios finais.
    """
    
    def __init__(self):
        """Inicializa o sistema multi-agente."""
        print("🚀 Inicializando Metanalyst-Agent...")
        
        # Inicializar banco de dados
        print("📊 Configurando PostgreSQL...")
        sync_init_database()
        
        # Construir grafo multi-agente
        print("🤖 Construindo sistema multi-agente...")
        self.graph = build_meta_analysis_graph()
        
        print("✅ Metanalyst-Agent pronto para uso!")
        print("=" * 60)
    
    def run(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        stream: bool = True
    ) -> Dict[str, Any]:
        """
        Executa uma meta-análise completa baseada na query do usuário.
        
        Args:
            query: Solicitação de meta-análise do usuário
            thread_id: ID único para a sessão (opcional)
            stream: Se deve mostrar progresso em tempo real
        
        Returns:
            Resultados da meta-análise incluindo relatório final
        """
        
        # Gerar thread_id único se não fornecido
        if thread_id is None:
            thread_id = f"metanalysis_{uuid.uuid4().hex[:8]}"
        
        # Configuração da execução
        config = create_meta_analysis_config(thread_id)
        
        # Mensagem inicial do usuário
        initial_message = {
            "role": "user",
            "content": query
        }
        
        print(f"🔬 Iniciando meta-análise...")
        print(f"📋 Query: {query}")
        print(f"🆔 Thread ID: {thread_id}")
        print("=" * 60)
        
        try:
            if stream:
                return self._run_with_stream(initial_message, config)
            else:
                return self._run_without_stream(initial_message, config)
                
        except Exception as e:
            print(f"❌ Erro durante execução: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_with_stream(self, initial_message: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Executa com streaming de progresso."""
        
        final_state = None
        step_count = 0
        
        print("📡 Modo streaming ativado - acompanhe o progresso:\n")
        
        for chunk in self.graph.stream(
            {"messages": [initial_message]},
            config,
            stream_mode="values"
        ):
            step_count += 1
            current_agent = chunk.get("last_agent", "system")
            
            # Mostrar progresso
            if chunk.get("messages"):
                last_message = chunk["messages"][-1]
                message_preview = last_message.content[:150] + "..." if len(last_message.content) > 150 else last_message.content
                
                print(f"🤖 [{current_agent.upper()}] {message_preview}")
                
                # Mostrar informações específicas do estado
                if chunk.get("pico_criteria"):
                    pico = chunk["pico_criteria"]
                    if any(pico.values()):  # Se PICO foi definido
                        print(f"   📋 PICO: P={pico.get('P', '')[:30]}... I={pico.get('I', '')[:30]}...")
                
                if chunk.get("relevant_urls"):
                    print(f"   🔗 URLs coletadas: {len(chunk['relevant_urls'])}")
                
                if chunk.get("processed_articles"):
                    print(f"   ⚙️ Artigos processados: {len(chunk['processed_articles'])}")
                
                if chunk.get("meta_analysis_results"):
                    results = chunk["meta_analysis_results"]
                    if results.get("pooled_effect_size"):
                        print(f"   📊 Effect size: {results['pooled_effect_size']:.3f}")
                
                if chunk.get("final_report_path"):
                    print(f"   📄 Relatório: {chunk['final_report_path']}")
                
                print()  # Linha em branco para separação
            
            final_state = chunk
            
            # Verificar se concluído
            if chunk.get("meta_analysis_complete", False):
                break
        
        print("=" * 60)
        print(f"✅ Meta-análise concluída em {step_count} etapas!")
        
        return self._format_final_results(final_state)
    
    def _run_without_stream(self, initial_message: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Executa sem streaming."""
        
        print("⏳ Executando meta-análise (modo silencioso)...")
        
        final_state = self.graph.invoke(
            {"messages": [initial_message]},
            config
        )
        
        print("✅ Meta-análise concluída!")
        
        return self._format_final_results(final_state)
    
    def _format_final_results(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Formata resultados finais para retorno."""
        
        if not final_state:
            return {"success": False, "error": "Nenhum estado final obtido"}
        
        # Extrair informações principais
        results = {
            "success": final_state.get("meta_analysis_complete", False),
            "thread_id": final_state.get("meta_analysis_id"),
            "timestamp": datetime.now().isoformat(),
            
            # PICO
            "pico_criteria": final_state.get("pico_criteria", {}),
            
            # Estatísticas de execução
            "articles_found": len(final_state.get("candidate_articles", [])),
            "articles_processed": len(final_state.get("processed_articles", [])),
            "studies_in_analysis": final_state.get("meta_analysis_results", {}).get("studies_included", 0),
            
            # Resultados da meta-análise
            "meta_analysis_results": final_state.get("meta_analysis_results", {}),
            
            # Arquivos gerados
            "final_report_path": final_state.get("final_report_path"),
            "forest_plot_path": final_state.get("forest_plot_path"),
            "funnel_plot_path": final_state.get("funnel_plot_path"),
            
            # Qualidade
            "quality_assessment": final_state.get("quality_assessment", {}),
            
            # Mensagens finais
            "messages": [msg.content for msg in final_state.get("messages", [])[-3:]]  # Últimas 3 mensagens
        }
        
        # Mostrar resumo
        if results["success"]:
            print("\n📊 RESUMO DA META-ANÁLISE:")
            print(f"   📚 Artigos encontrados: {results['articles_found']}")
            print(f"   ⚙️ Artigos processados: {results['articles_processed']}")
            print(f"   📈 Estudos na análise: {results['studies_in_analysis']}")
            
            if results["meta_analysis_results"]:
                meta_results = results["meta_analysis_results"]
                print(f"   🎯 Effect size: {meta_results.get('pooled_effect_size', 'N/A')}")
                print(f"   🔍 Heterogeneidade (I²): {meta_results.get('heterogeneity', {}).get('I_squared', 'N/A')}%")
                print(f"   📊 P-value: {meta_results.get('p_value', 'N/A')}")
            
            if results["final_report_path"]:
                print(f"   📄 Relatório final: {results['final_report_path']}")
        
        return results


def main():
    """Função principal para execução direta."""
    
    print("🧬 METANALYST-AGENT - Nobrega Medtech")
    print("Sistema Multi-Agente Autônomo para Meta-análises")
    print("=" * 60)
    
    # Exemplo de uso
    agent = MetanalystAgent()
    
    # Query de exemplo
    example_query = """
    Realize uma meta-análise sobre a eficácia da meditação mindfulness 
    versus terapia cognitivo-comportamental para tratamento de ansiedade 
    em adultos. Inclua forest plot e análise de heterogeneidade.
    """
    
    print("🔬 Executando exemplo de meta-análise...")
    results = agent.run(example_query)
    
    if results["success"]:
        print("\n🎉 Meta-análise de exemplo concluída com sucesso!")
    else:
        print(f"\n❌ Erro na meta-análise: {results.get('error', 'Erro desconhecido')}")


if __name__ == "__main__":
    main()