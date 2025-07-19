"""
Agente Orquestrador Central - Hub da arquitetura Hub-and-Spoke.
Responsável por analisar o estado atual e decidir qual agente especializado invocar.
"""

import logging
from typing import Dict, Any, Literal
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.models.state import MetaAnalysisState, update_state_phase, add_agent_log
from src.utils.config import Config

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Agente Orquestrador Central.
    
    Implementa a lógica de decisão para navegação entre agentes especializados
    na arquitetura hub-and-spoke. Mantém o estado global e controla o fluxo
    da meta-análise.
    """
    
    def __init__(self):
        """Inicializa o orquestrador."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,
            api_key=Config.OPENAI_API_KEY
        )
        self.name = "orchestrator"
    
    def decide_next_action(
        self, 
        state: MetaAnalysisState
    ) -> Command[Literal[
        "researcher", "processor", "retriever", "analyst", 
        "writer", "reviewer", "editor", "__end__"
    ]]:
        """
        Analisa o estado atual e decide qual agente invocar próximo.
        
        Args:
            state: Estado atual da meta-análise
            
        Returns:
            Command com próximo agente e atualizações de estado
        """
        try:
            logger.info(f"Orquestrador analisando fase: {state['current_phase']}")
            
            # Analisar estado atual
            phase = state["current_phase"]
            
            # Lógica de decisão baseada na fase atual
            if phase == "pico_definition":
                return self._handle_pico_definition(state)
            
            elif phase == "search":
                return self._handle_search_phase(state)
            
            elif phase == "extraction":
                return self._handle_extraction_phase(state)
            
            elif phase == "vectorization":
                return self._handle_vectorization_phase(state)
            
            elif phase == "analysis":
                return self._handle_analysis_phase(state)
            
            elif phase == "writing":
                return self._handle_writing_phase(state)
            
            elif phase == "review":
                return self._handle_review_phase(state)
            
            elif phase == "editing":
                return self._handle_editing_phase(state)
            
            else:
                # Fase desconhecida ou concluída
                return self._handle_completion(state)
        
        except Exception as e:
            logger.error(f"Erro no orquestrador: {e}")
            return self._handle_error(state, str(e))
    
    def _handle_pico_definition(self, state: MetaAnalysisState) -> Command:
        """Lida com a definição do PICO."""
        # Se PICO já está definido, prosseguir para busca
        if state.get("pico") and all(state["pico"].values()):
            logger.info("PICO já definido, prosseguindo para busca")
            return Command(
                goto="researcher",
                update=update_state_phase(state, "search", "researcher")
            )
        
        # Se não há PICO, analisar solicitação do usuário para extrair PICO
        user_request = state.get("user_request", "")
        if not user_request:
            return Command(
                goto="__end__",
                update={
                    "messages": state["messages"] + [
                        AIMessage("Erro: Solicitação do usuário não encontrada")
                    ]
                }
            )
        
        # Extrair PICO da solicitação
        pico = self._extract_pico_from_request(user_request)
        
        return Command(
            goto="researcher",
            update={
                **update_state_phase(state, "search", "researcher"),
                "pico": pico,
                "messages": state["messages"] + [
                    AIMessage(f"PICO definido: {pico}")
                ]
            }
        )
    
    def _handle_search_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de busca de literatura."""
        # Verificar se já temos URLs candidatas
        candidate_urls = state.get("candidate_urls", [])
        
        if not candidate_urls:
            # Precisamos buscar literatura
            logger.info("Iniciando busca de literatura")
            return Command(
                goto="researcher",
                update=add_agent_log(state, self.name, "search_needed")
            )
        
        # Se temos URLs, verificar se precisamos de mais
        max_papers = Config.MAX_PAPERS_PER_SEARCH
        if len(candidate_urls) < max_papers:
            # Buscar mais literatura
            return Command(
                goto="researcher", 
                update=add_agent_log(state, self.name, "search_more_literature")
            )
        
        # Temos URLs suficientes, prosseguir para extração
        logger.info(f"Encontradas {len(candidate_urls)} URLs, prosseguindo para extração")
        return Command(
            goto="processor",
            update={
                **update_state_phase(state, "extraction", "processor"),
                "processing_queue": [url["url"] for url in candidate_urls[:max_papers]]
            }
        )
    
    def _handle_extraction_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de extração de dados."""
        processing_queue = state.get("processing_queue", [])
        processed_articles = state.get("processed_articles", [])
        
        if processing_queue:
            # Ainda há artigos para processar
            logger.info(f"Processando {len(processing_queue)} artigos restantes")
            return Command(
                goto="processor",
                update=add_agent_log(state, self.name, "continue_processing")
            )
        
        # Verificar se temos dados suficientes
        if len(processed_articles) < 3:
            logger.warning("Poucos artigos processados, retornando para busca")
            return Command(
                goto="researcher",
                update={
                    **update_state_phase(state, "search", "researcher"),
                    "messages": state["messages"] + [
                        AIMessage("Poucos artigos encontrados, expandindo busca...")
                    ]
                }
            )
        
        # Temos dados suficientes, prosseguir para vetorização
        logger.info(f"Extração concluída com {len(processed_articles)} artigos")
        return Command(
            goto="processor",  # Processor também faz vetorização
            update=update_state_phase(state, "vectorization", "processor")
        )
    
    def _handle_vectorization_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de vetorização."""
        vector_store_id = state.get("vector_store_id")
        chunk_count = state.get("chunk_count", 0)
        
        if not vector_store_id or chunk_count == 0:
            # Vetorização ainda não concluída
            return Command(
                goto="processor",
                update=add_agent_log(state, self.name, "continue_vectorization")
            )
        
        # Vetorização concluída, prosseguir para análise
        logger.info(f"Vetorização concluída com {chunk_count} chunks")
        return Command(
            goto="retriever",
            update=update_state_phase(state, "analysis", "retriever")
        )
    
    def _handle_analysis_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de análise."""
        retrieval_results = state.get("retrieval_results", [])
        statistical_analysis = state.get("statistical_analysis", {})
        
        if not retrieval_results:
            # Precisamos buscar informações relevantes
            return Command(
                goto="retriever",
                update=add_agent_log(state, self.name, "retrieve_information")
            )
        
        if not statistical_analysis:
            # Precisamos fazer análise estatística
            return Command(
                goto="analyst",
                update=add_agent_log(state, self.name, "perform_analysis")
            )
        
        # Análise concluída, prosseguir para escrita
        logger.info("Análise concluída, iniciando escrita do relatório")
        return Command(
            goto="writer",
            update=update_state_phase(state, "writing", "writer")
        )
    
    def _handle_writing_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de escrita do relatório."""
        draft_report = state.get("draft_report")
        
        if not draft_report:
            # Precisamos escrever o relatório
            return Command(
                goto="writer",
                update=add_agent_log(state, self.name, "write_report")
            )
        
        # Relatório escrito, prosseguir para revisão
        logger.info("Relatório escrito, iniciando revisão")
        return Command(
            goto="reviewer",
            update=update_state_phase(state, "review", "reviewer")
        )
    
    def _handle_review_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de revisão."""
        review_feedback = state.get("review_feedback", [])
        
        if not review_feedback:
            # Precisamos revisar o relatório
            return Command(
                goto="reviewer",
                update=add_agent_log(state, self.name, "review_report")
            )
        
        # Verificar se há feedback que requer ação
        needs_revision = any(
            feedback.get("action") == "revise" 
            for feedback in review_feedback
        )
        
        if needs_revision:
            # Retornar para escrita com feedback
            logger.info("Feedback de revisão requer modificações")
            return Command(
                goto="writer",
                update={
                    **update_state_phase(state, "writing", "writer"),
                    "messages": state["messages"] + [
                        AIMessage("Aplicando feedback da revisão...")
                    ]
                }
            )
        
        # Revisão aprovada, prosseguir para edição final
        logger.info("Revisão aprovada, iniciando edição final")
        return Command(
            goto="editor",
            update=update_state_phase(state, "editing", "editor")
        )
    
    def _handle_editing_phase(self, state: MetaAnalysisState) -> Command:
        """Lida com a fase de edição final."""
        final_report = state.get("final_report")
        
        if not final_report:
            # Precisamos fazer edição final
            return Command(
                goto="editor",
                update=add_agent_log(state, self.name, "final_editing")
            )
        
        # Edição concluída, finalizar processo
        return self._handle_completion(state)
    
    def _handle_completion(self, state: MetaAnalysisState) -> Command:
        """Lida com a conclusão da meta-análise."""
        logger.info("Meta-análise concluída com sucesso")
        
        # Calcular estatísticas finais
        total_articles = len(state.get("processed_articles", []))
        total_time = sum(state.get("execution_time", {}).values())
        
        completion_message = f"""
        🎉 META-ANÁLISE CONCLUÍDA COM SUCESSO!
        
        📊 Estatísticas:
        - Artigos processados: {total_articles}
        - Chunks criados: {state.get('chunk_count', 0)}
        - Tempo total: {total_time:.2f}s
        
        📋 Relatório final disponível em: outputs/meta_analysis_report_{state['meta_analysis_id']}.html
        """
        
        return Command(
            goto="__end__",
            update={
                **update_state_phase(state, "completed", self.name),
                "messages": state["messages"] + [AIMessage(completion_message)]
            }
        )
    
    def _handle_error(self, state: MetaAnalysisState, error_msg: str) -> Command:
        """Lida com erros no processo."""
        logger.error(f"Erro no orquestrador: {error_msg}")
        
        return Command(
            goto="__end__",
            update={
                "messages": state["messages"] + [
                    AIMessage(f"Erro na meta-análise: {error_msg}")
                ],
                **add_agent_log(state, self.name, "error", {"error": error_msg}, "error")
            }
        )
    
    def _extract_pico_from_request(self, user_request: str) -> Dict[str, str]:
        """
        Extrai estrutura PICO da solicitação do usuário usando LLM.
        
        Args:
            user_request: Solicitação em linguagem natural
            
        Returns:
            Dicionário com PICO estruturado
        """
        try:
            prompt = f"""
            Analise a seguinte solicitação de meta-análise e extraia a estrutura PICO:
            
            Solicitação: "{user_request}"
            
            Extraia e estruture as seguintes informações:
            - P (População): Qual grupo de pacientes/participantes?
            - I (Intervenção): Qual tratamento/intervenção está sendo estudada?
            - C (Comparação): Qual é o grupo controle ou comparação?
            - O (Outcome/Desfecho): Qual resultado está sendo medido?
            
            Responda em formato JSON:
            {{
                "population": "descrição da população",
                "intervention": "descrição da intervenção",
                "comparison": "descrição da comparação",
                "outcome": "descrição do desfecho"
            }}
            
            Se algum elemento não estiver claro, faça uma inferência razoável baseada no contexto médico.
            """
            
            response = self.llm.invoke(prompt)
            
            # Tentar parsear JSON
            import json
            pico = json.loads(response.content)
            
            # Validar que temos todos os campos
            required_fields = ["population", "intervention", "comparison", "outcome"]
            for field in required_fields:
                if field not in pico or not pico[field]:
                    pico[field] = f"Não especificado em '{user_request}'"
            
            logger.info(f"PICO extraído: {pico}")
            return pico
            
        except Exception as e:
            logger.error(f"Erro ao extrair PICO: {e}")
            # Retornar PICO padrão baseado na solicitação
            return {
                "population": f"Pacientes mencionados em: {user_request[:100]}",
                "intervention": "Intervenção a ser determinada",
                "comparison": "Controle a ser determinado",
                "outcome": "Desfecho a ser determinado"
            }


def orchestrator_node(state: MetaAnalysisState) -> Command:
    """
    Nó do orquestrador para uso no LangGraph.
    
    Args:
        state: Estado atual da meta-análise
        
    Returns:
        Command com próxima ação
    """
    orchestrator = OrchestratorAgent()
    return orchestrator.decide_next_action(state)