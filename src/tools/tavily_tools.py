"""
Ferramentas para integração com Tavily API.
Implementa busca de literatura científica e extração de conteúdo.
"""

import logging
from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from langchain_core.tools import tool
from src.utils.config import Config
from src.models.schemas import SearchResult
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class TavilyTools:
    """Classe para ferramentas Tavily."""
    
    def __init__(self):
        """Inicializa cliente Tavily."""
        if not Config.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY não configurada")
        
        self.client = TavilyClient(api_key=Config.TAVILY_API_KEY)
        self.search_config = Config.get_search_config()
    
    def search_literature(
        self, 
        query: str, 
        domains: List[str] = None,
        max_results: int = None
    ) -> List[SearchResult]:
        """
        Busca literatura científica usando Tavily.
        
        Args:
            query: Query de busca
            domains: Domínios específicos para buscar
            max_results: Número máximo de resultados
            
        Returns:
            Lista de resultados de busca
        """
        try:
            max_results = max_results or self.search_config["max_papers"]
            domains = domains or self.search_config["domains"]
            
            # Configurar busca para literatura científica
            search_params = {
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_domains": domains,
                "include_answer": False,
                "include_raw_content": False
            }
            
            logger.info(f"Buscando literatura: '{query}' em {len(domains)} domínios")
            
            response = self.client.search(**search_params)
            results = []
            
            for item in response.get("results", []):
                try:
                    # Extrair informações do resultado
                    title = item.get("title", "")
                    url = item.get("url", "")
                    content = item.get("content", "")
                    
                    # Extrair ano do título ou conteúdo
                    year = self._extract_year(title + " " + content)
                    
                    # Extrair autores (heurística simples)
                    authors = self._extract_authors(title, content)
                    
                    # Calcular scores de relevância
                    relevance_score = self._calculate_relevance_score(
                        query, title, content
                    )
                    
                    result = SearchResult(
                        url=url,
                        title=title,
                        authors=authors,
                        abstract=content[:500] + "..." if len(content) > 500 else content,
                        year=year,
                        journal=self._extract_journal(content),
                        relevance_score=relevance_score,
                        pico_match_score=0.0,  # Será calculado depois
                        search_query=query,
                        search_domain=self._extract_domain(url)
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar resultado: {e}")
                    continue
            
            logger.info(f"Encontrados {len(results)} resultados para '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca Tavily: {e}")
            return []
    
    def extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Extrai conteúdo completo de uma URL usando Tavily Extract.
        
        Args:
            url: URL para extrair
            
        Returns:
            Conteúdo extraído ou None se falhou
        """
        try:
            logger.info(f"Extraindo conteúdo de: {url}")
            
            response = self.client.extract(url)
            
            if not response or "content" not in response:
                logger.warning(f"Nenhum conteúdo extraído de {url}")
                return None
            
            content = response["content"]
            
            # Estruturar conteúdo extraído
            extracted = {
                "url": url,
                "title": response.get("title", ""),
                "content": content,
                "raw_content": response.get("raw_content", ""),
                "metadata": {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "content_length": len(content),
                    "has_raw_content": bool(response.get("raw_content"))
                }
            }
            
            # Tentar extrair seções estruturadas
            sections = self._extract_sections(content)
            if sections:
                extracted["sections"] = sections
            
            logger.info(f"Conteúdo extraído com sucesso: {len(content)} caracteres")
            return extracted
            
        except Exception as e:
            logger.error(f"Erro ao extrair conteúdo de {url}: {e}")
            return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extrai ano do texto usando regex."""
        # Procurar anos entre 1900 e ano atual
        current_year = datetime.now().year
        year_pattern = r'\b(19[0-9]{2}|20[0-9]{2})\b'
        matches = re.findall(year_pattern, text)
        
        if matches:
            # Retornar o ano mais recente encontrado
            years = [int(year) for year in matches if 1900 <= int(year) <= current_year]
            return max(years) if years else None
        
        return None
    
    def _extract_authors(self, title: str, content: str) -> List[str]:
        """Extrai autores do título e conteúdo (heurística simples)."""
        authors = []
        
        # Padrões comuns para autores
        author_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Nome Sobrenome
            r'([A-Z]\. [A-Z][a-z]+)',      # A. Sobrenome
            r'([A-Z][a-z]+ [A-Z]\.)',      # Nome A.
        ]
        
        text = title + " " + content[:200]  # Primeiros 200 chars do conteúdo
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text)
            authors.extend(matches[:3])  # Máximo 3 autores
            if len(authors) >= 3:
                break
        
        return list(set(authors))  # Remover duplicatas
    
    def _extract_journal(self, content: str) -> Optional[str]:
        """Extrai nome do journal do conteúdo."""
        # Padrões comuns para journals
        journal_patterns = [
            r'Published in ([^,.\n]+)',
            r'Journal: ([^,.\n]+)',
            r'([A-Z][a-zA-Z\s]+Journal[a-zA-Z\s]*)',
            r'(New England Journal of Medicine)',
            r'(The Lancet)',
            r'(Nature)',
            r'(Science)',
            r'(BMJ)',
            r'(JAMA)'
        ]
        
        for pattern in journal_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_domain(self, url: str) -> str:
        """Extrai domínio da URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"
    
    def _calculate_relevance_score(
        self, 
        query: str, 
        title: str, 
        content: str
    ) -> float:
        """Calcula score de relevância baseado em palavras-chave."""
        query_words = set(query.lower().split())
        text = (title + " " + content).lower()
        
        # Contar palavras da query no texto
        matches = sum(1 for word in query_words if word in text)
        
        # Normalizar por número de palavras na query
        score = matches / len(query_words) if query_words else 0.0
        
        # Bonus por palavras no título
        title_matches = sum(1 for word in query_words if word in title.lower())
        title_bonus = (title_matches / len(query_words)) * 0.5
        
        return min(score + title_bonus, 1.0)
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extrai seções estruturadas do conteúdo."""
        sections = {}
        
        # Padrões para seções comuns em papers
        section_patterns = {
            "abstract": r"(?i)abstract[:\s]*(.*?)(?=\n\s*(?:introduction|background|methods|keywords|$))",
            "introduction": r"(?i)introduction[:\s]*(.*?)(?=\n\s*(?:methods|materials|background|results|$))",
            "methods": r"(?i)(?:methods|methodology)[:\s]*(.*?)(?=\n\s*(?:results|discussion|conclusion|$))",
            "results": r"(?i)results[:\s]*(.*?)(?=\n\s*(?:discussion|conclusion|references|$))",
            "discussion": r"(?i)discussion[:\s]*(.*?)(?=\n\s*(?:conclusion|references|acknowledgments|$))",
            "conclusion": r"(?i)conclusion[:\s]*(.*?)(?=\n\s*(?:references|acknowledgments|$))"
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                if len(section_text) > 50:  # Só incluir se tiver conteúdo substancial
                    sections[section_name] = section_text[:1000]  # Limitar tamanho
        
        return sections


# Ferramentas como funções para uso no LangGraph
@tool
def search_scientific_literature(
    query: str, 
    max_results: int = 15,
    domains: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Busca literatura científica usando Tavily API.
    
    Args:
        query: Termo de busca
        max_results: Número máximo de resultados
        domains: Lista de domínios específicos
        
    Returns:
        Lista de resultados de busca
    """
    tavily = TavilyTools()
    results = tavily.search_literature(query, domains, max_results)
    return [result.dict() for result in results]


@tool
def extract_article_content(url: str) -> Optional[Dict[str, Any]]:
    """
    Extrai conteúdo completo de um artigo científico.
    
    Args:
        url: URL do artigo
        
    Returns:
        Conteúdo extraído ou None
    """
    tavily = TavilyTools()
    return tavily.extract_content(url)


@tool
def search_with_pico(
    population: str,
    intervention: str,
    comparison: str,
    outcome: str,
    max_results: int = 15
) -> List[Dict[str, Any]]:
    """
    Busca literatura usando estrutura PICO.
    
    Args:
        population: População estudada
        intervention: Intervenção
        comparison: Comparação
        outcome: Desfecho
        max_results: Máximo de resultados
        
    Returns:
        Lista de resultados
    """
    # Construir query a partir do PICO
    pico_query = f"{population} {intervention} {comparison} {outcome}"
    
    # Adicionar termos médicos relevantes
    medical_terms = ["study", "trial", "randomized", "controlled", "meta-analysis"]
    query = f"{pico_query} {' OR '.join(medical_terms)}"
    
    tavily = TavilyTools()
    results = tavily.search_literature(query, max_results=max_results)
    
    # Calcular PICO match score para cada resultado
    for result in results:
        result.pico_match_score = _calculate_pico_match_score(
            result, population, intervention, comparison, outcome
        )
    
    # Ordenar por PICO match score
    results.sort(key=lambda x: x.pico_match_score, reverse=True)
    
    return [result.dict() for result in results]


def _calculate_pico_match_score(
    result: SearchResult,
    population: str,
    intervention: str,
    comparison: str,
    outcome: str
) -> float:
    """Calcula score de match com PICO."""
    pico_terms = [population, intervention, comparison, outcome]
    text = (result.title + " " + result.abstract).lower()
    
    matches = 0
    for term in pico_terms:
        if term.lower() in text:
            matches += 1
    
    return matches / len(pico_terms)