"""
Agente Pesquisador - Especializado em busca de literatura científica.
Utiliza Tavily API para encontrar artigos relevantes baseados no PICO.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import time

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.models.state import MetaAnalysisState, add_agent_log
from src.tools.tavily_tools import TavilyTools
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """
    Agente Pesquisador.
    
    Responsável por buscar literatura científica usando Tavily API,
    gerar queries baseadas no PICO e filtrar resultados por relevância.
    """
    
    def __init__(self):
        """Inicializa o agente pesquisador."""
        self.tavily = TavilyTools()
        self.name = "researcher"
        self.search_config = Config.get_search_config()
        self.quality_thresholds = Config.get_quality_thresholds()
    
    def search_literature(
        self, 
        state: MetaAnalysisState,
        config: RunnableConfig = None
    ) -> Dict[str, Any]:
        """
        Busca literatura científica baseada no PICO.
        
        Args:
            state: Estado atual da meta-análise
            config: Configuração do LangGraph
            
        Returns:
            Atualizações do estado
        """
        start_time = time.time()
        
        try:
            logger.info("Iniciando busca de literatura científica")
            
            # Obter PICO do estado
            pico = state.get("pico", {})
            if not pico:
                return self._handle_error(state, "PICO não definido")
            
            # Gerar queries de busca
            search_queries = self._generate_search_queries(pico)
            
            # Buscar literatura para cada query
            all_results = []
            for query in search_queries:
                logger.info(f"Buscando: {query}")
                
                results = self.tavily.search_literature(
                    query=query,
                    max_results=self.search_config["max_papers"]
                )
                
                # Calcular PICO match score para cada resultado
                for result in results:
                    result.pico_match_score = self._calculate_pico_match_score(
                        result, pico
                    )
                
                all_results.extend(results)
                
                # Pequena pausa entre buscas
                time.sleep(1)
            
            # Filtrar e ranquear resultados
            filtered_results = self._filter_and_rank_results(all_results, pico)
            
            # Converter para formato do estado
            candidate_urls = []
            for result in filtered_results:
                candidate_urls.append({
                    "url": result.url,
                    "title": result.title,
                    "authors": result.authors,
                    "abstract": result.abstract,
                    "year": result.year,
                    "journal": result.journal,
                    "relevance_score": result.relevance_score,
                    "pico_match_score": result.pico_match_score,
                    "search_query": result.search_query,
                    "domain": result.search_domain
                })
            
            execution_time = time.time() - start_time
            
            logger.info(f"Busca concluída: {len(candidate_urls)} artigos encontrados")
            
            return {
                "candidate_urls": candidate_urls,
                "search_queries": search_queries,
                "search_domains": self.search_config["domains"],
                "execution_time": {
                    **state.get("execution_time", {}),
                    "search": execution_time
                },
                "messages": state["messages"] + [
                    AIMessage(f"Encontrados {len(candidate_urls)} artigos científicos relevantes")
                ],
                **add_agent_log(
                    state, 
                    self.name, 
                    "literature_search_completed",
                    {
                        "queries_used": len(search_queries),
                        "results_found": len(candidate_urls),
                        "execution_time": execution_time
                    }
                )
            }
            
        except Exception as e:
            return self._handle_error(state, str(e))
    
    def _generate_search_queries(self, pico: Dict[str, str]) -> List[str]:
        """
        Gera queries de busca baseadas no PICO.
        
        Args:
            pico: Estrutura PICO
            
        Returns:
            Lista de queries de busca
        """
        queries = []
        
        # Query principal com todos os elementos PICO
        main_query = f"{pico['population']} {pico['intervention']} {pico['comparison']} {pico['outcome']}"
        queries.append(main_query)
        
        # Query focada na intervenção e desfecho
        intervention_outcome = f"{pico['intervention']} {pico['outcome']} randomized controlled trial"
        queries.append(intervention_outcome)
        
        # Query com população e intervenção
        population_intervention = f"{pico['population']} {pico['intervention']} clinical trial"
        queries.append(population_intervention)
        
        # Query com termos de meta-análise
        meta_analysis_query = f"{pico['intervention']} {pico['outcome']} meta-analysis systematic review"
        queries.append(meta_analysis_query)
        
        # Query com comparação específica
        if pico['comparison'].lower() not in ['placebo', 'controle', 'control']:
            comparison_query = f"{pico['intervention']} vs {pico['comparison']} {pico['outcome']}"
            queries.append(comparison_query)
        
        # Limitar número de queries
        return queries[:4]
    
    def _calculate_pico_match_score(
        self, 
        result, 
        pico: Dict[str, str]
    ) -> float:
        """
        Calcula score de match entre resultado e PICO.
        
        Args:
            result: Resultado da busca
            pico: Estrutura PICO
            
        Returns:
            Score de 0 a 1
        """
        text = (result.title + " " + result.abstract).lower()
        
        matches = 0
        total_elements = 0
        
        for element_name, element_value in pico.items():
            if not element_value or element_value.startswith("Não especificado"):
                continue
                
            total_elements += 1
            
            # Dividir elemento em palavras-chave
            keywords = [word.strip().lower() for word in element_value.split() 
                       if len(word.strip()) > 3]
            
            # Verificar se alguma palavra-chave está presente
            element_match = any(keyword in text for keyword in keywords)
            if element_match:
                matches += 1
        
        return matches / total_elements if total_elements > 0 else 0.0
    
    def _filter_and_rank_results(
        self, 
        results: List, 
        pico: Dict[str, str]
    ) -> List:
        """
        Filtra e ranqueia resultados por qualidade e relevância.
        
        Args:
            results: Lista de resultados
            pico: Estrutura PICO
            
        Returns:
            Lista filtrada e ordenada
        """
        # Remover duplicatas por URL
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Filtrar por thresholds de qualidade
        filtered_results = []
        for result in unique_results:
            if (result.relevance_score >= self.quality_thresholds["min_relevance"] and
                result.pico_match_score >= self.quality_thresholds["min_pico_match"]):
                filtered_results.append(result)
        
        # Se poucos resultados passaram no filtro, relaxar critérios
        if len(filtered_results) < 5:
            logger.warning("Poucos resultados de alta qualidade, relaxando critérios")
            filtered_results = [
                result for result in unique_results
                if result.relevance_score >= 0.5 or result.pico_match_score >= 0.4
            ]
        
        # Ordenar por score combinado
        def combined_score(result):
            return (result.relevance_score * 0.6 + result.pico_match_score * 0.4)
        
        filtered_results.sort(key=combined_score, reverse=True)
        
        # Limitar número de resultados
        max_results = self.search_config["max_papers"]
        return filtered_results[:max_results]
    
    def _handle_error(self, state: MetaAnalysisState, error_msg: str) -> Dict[str, Any]:
        """Lida com erros na busca."""
        logger.error(f"Erro no agente pesquisador: {error_msg}")
        
        return {
            "messages": state["messages"] + [
                AIMessage(f"Erro na busca de literatura: {error_msg}")
            ],
            **add_agent_log(
                state, 
                self.name, 
                "search_error", 
                {"error": error_msg}, 
                "error"
            )
        }
    
    def expand_search(
        self, 
        state: MetaAnalysisState,
        config: RunnableConfig = None
    ) -> Dict[str, Any]:
        """
        Expande a busca com queries alternativas.
        
        Args:
            state: Estado atual
            config: Configuração do LangGraph
            
        Returns:
            Atualizações do estado
        """
        try:
            logger.info("Expandindo busca de literatura")
            
            pico = state.get("pico", {})
            existing_urls = {url["url"] for url in state.get("candidate_urls", [])}
            
            # Gerar queries alternativas
            alternative_queries = self._generate_alternative_queries(pico)
            
            new_results = []
            for query in alternative_queries:
                results = self.tavily.search_literature(
                    query=query,
                    max_results=10
                )
                
                # Filtrar URLs já encontradas
                for result in results:
                    if result.url not in existing_urls:
                        result.pico_match_score = self._calculate_pico_match_score(
                            result, pico
                        )
                        new_results.append(result)
                        existing_urls.add(result.url)
            
            # Converter novos resultados
            new_candidate_urls = []
            for result in new_results:
                new_candidate_urls.append({
                    "url": result.url,
                    "title": result.title,
                    "authors": result.authors,
                    "abstract": result.abstract,
                    "year": result.year,
                    "journal": result.journal,
                    "relevance_score": result.relevance_score,
                    "pico_match_score": result.pico_match_score,
                    "search_query": result.search_query,
                    "domain": result.search_domain
                })
            
            # Combinar com URLs existentes
            all_candidate_urls = state.get("candidate_urls", []) + new_candidate_urls
            
            logger.info(f"Busca expandida: {len(new_candidate_urls)} novos artigos")
            
            return {
                "candidate_urls": all_candidate_urls,
                "search_queries": state.get("search_queries", []) + alternative_queries,
                "messages": state["messages"] + [
                    AIMessage(f"Busca expandida: {len(new_candidate_urls)} novos artigos encontrados")
                ],
                **add_agent_log(
                    state,
                    self.name,
                    "search_expanded",
                    {"new_results": len(new_candidate_urls)}
                )
            }
            
        except Exception as e:
            return self._handle_error(state, str(e))
    
    def _generate_alternative_queries(self, pico: Dict[str, str]) -> List[str]:
        """Gera queries alternativas para expansão da busca."""
        queries = []
        
        # Sinônimos e termos alternativos comuns
        intervention_synonyms = {
            "metformina": ["metformin", "glucophage"],
            "aspirina": ["aspirin", "ácido acetilsalicílico"],
            "estatina": ["statin", "atorvastatina", "simvastatina"],
        }
        
        # Query com sinônimos
        intervention = pico.get("intervention", "").lower()
        for key, synonyms in intervention_synonyms.items():
            if key in intervention:
                for synonym in synonyms:
                    synonym_query = f"{pico['population']} {synonym} {pico['outcome']}"
                    queries.append(synonym_query)
        
        # Queries mais específicas por tipo de estudo
        study_types = [
            "randomized controlled trial",
            "clinical trial",
            "cohort study",
            "case control study"
        ]
        
        for study_type in study_types:
            specific_query = f"{pico['intervention']} {pico['outcome']} {study_type}"
            queries.append(specific_query)
        
        return queries[:3]  # Limitar queries alternativas


def researcher_agent(
    state: MetaAnalysisState,
    config: RunnableConfig = None
) -> Dict[str, Any]:
    """
    Nó do agente pesquisador para uso no LangGraph.
    
    Args:
        state: Estado atual da meta-análise
        config: Configuração do LangGraph
        
    Returns:
        Atualizações do estado
    """
    researcher = ResearcherAgent()
    
    # Verificar se é busca inicial ou expansão
    existing_urls = state.get("candidate_urls", [])
    
    if not existing_urls:
        # Busca inicial
        return researcher.search_literature(state, config)
    else:
        # Expansão da busca
        return researcher.expand_search(state, config)