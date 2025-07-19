"""
Ferramentas de pesquisa de literatura médica usando Tavily Search.
"""

from typing import List, Dict, Any, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from tavily import TavilyClient
from ..models.state import MetaAnalysisState
from ..models.config import config
import re

# Cliente Tavily global
tavily_client = TavilyClient(api_key=config.tavily_api_key)

@tool("search_medical_literature")
def search_medical_literature(
    query: Annotated[str, "Query de busca otimizada para literatura médica"],
    max_results: Annotated[int, "Número máximo de resultados (padrão: 20)"] = 20,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Busca literatura médica usando Tavily Search com domínios específicos de alta qualidade.
    
    Domínios incluídos:
    - New England Journal of Medicine (NEJM)
    - JAMA Network
    - The Lancet
    - BMJ
    - PubMed/PMC
    - SciELO
    - Cochrane Library
    """
    
    try:
        # Domínios médicos de alta qualidade
        medical_domains = [
            "nejm.org",
            "jamanetwork.com", 
            "thelancet.com",
            "bmj.com",
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov/pmc",
            "scielo.org",
            "cochranelibrary.com"
        ]
        
        # Realizar busca com domínios específicos
        search_results = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=medical_domains,
            include_answer=False,
            include_raw_content=False
        )
        
        # Processar resultados
        processed_results = []
        for result in search_results.get("results", []):
            processed_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "domain": extract_domain(result.get("url", "")),
                "published_date": extract_date_from_content(result.get("content", "")),
            }
            processed_results.append(processed_result)
        
        # Ordenar por score de relevância
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Filtrar por score mínimo
        min_score = config.search.min_relevance_score
        filtered_results = [r for r in processed_results if r["score"] >= min_score]
        
        return {
            "success": True,
            "query": query,
            "total_found": len(processed_results),
            "filtered_count": len(filtered_results),
            "min_score_applied": min_score,
            "results": filtered_results,
            "domains_searched": medical_domains
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": []
        }

@tool("generate_search_queries") 
def generate_search_queries(
    pico: Annotated[Dict[str, str], "Estrutura PICO (Population, Intervention, Comparison, Outcome)"],
    additional_terms: Annotated[List[str], "Termos adicionais para incluir"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> List[str]:
    """
    Gera queries de busca otimizadas baseadas na estrutura PICO.
    """
    
    queries = []
    additional_terms = additional_terms or []
    
    # Extrair componentes PICO
    population = pico.get("P", "").strip()
    intervention = pico.get("I", "").strip() 
    comparison = pico.get("C", "").strip()
    outcome = pico.get("O", "").strip()
    
    if not any([population, intervention, outcome]):
        return ["Erro: PICO deve conter pelo menos Population, Intervention e Outcome"]
    
    # Query básica PICO
    basic_terms = [t for t in [population, intervention, comparison, outcome] if t]
    if basic_terms:
        queries.append(" AND ".join(basic_terms))
    
    # Query focada em intervenção e outcome
    if intervention and outcome:
        queries.append(f'"{intervention}" AND "{outcome}"')
    
    # Query com população específica
    if population and intervention:
        queries.append(f'"{population}" AND "{intervention}"')
    
    # Queries para meta-análise e revisões sistemáticas
    if intervention:
        queries.append(f'"{intervention}" AND "meta-analysis"')
        queries.append(f'"{intervention}" AND "systematic review"')
        queries.append(f'"{intervention}" AND "randomized controlled trial"')
    
    # Queries com comparação (se disponível)
    if comparison and intervention:
        queries.append(f'"{intervention}" vs "{comparison}"')
        queries.append(f'"{intervention}" versus "{comparison}"')
    
    # Adicionar termos extras
    for term in additional_terms:
        if intervention:
            queries.append(f'"{intervention}" AND "{term}"')
    
    # Queries específicas por tipo de estudo
    base_query = " ".join([t for t in [population, intervention] if t])
    if base_query:
        study_types = [
            "randomized controlled trial",
            "clinical trial", 
            "cohort study",
            "case-control study"
        ]
        for study_type in study_types:
            queries.append(f'{base_query} AND "{study_type}"')
    
    # Remover duplicatas e queries vazias
    unique_queries = list(set([q for q in queries if q.strip()]))
    
    return unique_queries[:10]  # Limitar a 10 queries

@tool("evaluate_article_relevance")
def evaluate_article_relevance(
    article_data: Annotated[Dict[str, Any], "Dados do artigo (título, conteúdo, URL)"],
    pico: Annotated[Dict[str, str], "Critérios PICO para avaliação"],
    inclusion_criteria: Annotated[List[str], "Critérios de inclusão"] = None,
    exclusion_criteria: Annotated[List[str], "Critérios de exclusão"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Avalia a relevância de um artigo baseado nos critérios PICO e inclusão/exclusão.
    """
    
    title = article_data.get("title", "").lower()
    content = article_data.get("content", "").lower()
    url = article_data.get("url", "")
    
    # Extrair componentes PICO
    population = pico.get("P", "").lower()
    intervention = pico.get("I", "").lower()
    comparison = pico.get("C", "").lower()
    outcome = pico.get("O", "").lower()
    
    inclusion_criteria = inclusion_criteria or []
    exclusion_criteria = exclusion_criteria or []
    
    # Score de relevância
    relevance_score = 0.0
    details = []
    
    # Verificar presença dos componentes PICO (peso: 0.6)
    pico_score = 0.0
    pico_components = 0
    
    if population and (population in title or population in content):
        pico_score += 0.25
        details.append(f"População '{population}' encontrada")
    if population:
        pico_components += 1
        
    if intervention and (intervention in title or intervention in content):
        pico_score += 0.25
        details.append(f"Intervenção '{intervention}' encontrada")
    if intervention:
        pico_components += 1
        
    if comparison and (comparison in title or comparison in content):
        pico_score += 0.15
        details.append(f"Comparação '{comparison}' encontrada")
    if comparison:
        pico_components += 1
        
    if outcome and (outcome in title or outcome in content):
        pico_score += 0.35
        details.append(f"Outcome '{outcome}' encontrado")
    if outcome:
        pico_components += 1
    
    # Normalizar score PICO
    if pico_components > 0:
        relevance_score += (pico_score / pico_components) * 0.6
    
    # Verificar critérios de inclusão (peso: 0.2)
    inclusion_score = 0.0
    if inclusion_criteria:
        for criteria in inclusion_criteria:
            if criteria.lower() in title or criteria.lower() in content:
                inclusion_score += 1.0 / len(inclusion_criteria)
                details.append(f"Critério de inclusão '{criteria}' atendido")
    else:
        inclusion_score = 1.0  # Se não há critérios, assume que atende
    
    relevance_score += inclusion_score * 0.2
    
    # Verificar critérios de exclusão (peso: -0.3)
    exclusion_penalty = 0.0
    excluded = False
    if exclusion_criteria:
        for criteria in exclusion_criteria:
            if criteria.lower() in title or criteria.lower() in content:
                exclusion_penalty += 0.3
                details.append(f"❌ Critério de exclusão '{criteria}' encontrado")
                excluded = True
                break
    
    relevance_score -= exclusion_penalty
    
    # Bonus por tipo de estudo (peso: 0.1)
    study_type_bonus = 0.0
    high_quality_terms = [
        "randomized controlled trial", "rct", "meta-analysis", 
        "systematic review", "clinical trial"
    ]
    
    for term in high_quality_terms:
        if term in title or term in content:
            study_type_bonus = 0.1
            details.append(f"Estudo de alta qualidade: '{term}'")
            break
    
    relevance_score += study_type_bonus
    
    # Bonus por domínio confiável (peso: 0.1)
    domain_bonus = 0.0
    high_impact_domains = ["nejm.org", "jamanetwork.com", "thelancet.com", "bmj.com"]
    domain = extract_domain(url)
    
    if any(trusted in domain for trusted in high_impact_domains):
        domain_bonus = 0.1
        details.append(f"Domínio de alto impacto: {domain}")
    
    relevance_score += domain_bonus
    
    # Garantir que o score está entre 0 e 1
    relevance_score = max(0.0, min(1.0, relevance_score))
    
    # Determinar se o artigo deve ser incluído
    should_include = relevance_score >= config.search.min_relevance_score and not excluded
    
    return {
        "url": url,
        "title": article_data.get("title", ""),
        "relevance_score": round(relevance_score, 3),
        "should_include": should_include,
        "excluded": excluded,
        "evaluation_details": details,
        "domain": domain,
        "pico_match_score": round(pico_score, 3),
        "inclusion_score": round(inclusion_score, 3),
        "exclusion_penalty": round(exclusion_penalty, 3)
    }

def extract_domain(url: str) -> str:
    """Extrair domínio de uma URL"""
    import re
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1) if match else ""

def extract_date_from_content(content: str) -> str:
    """Tentar extrair data de publicação do conteúdo"""
    import re
    
    # Padrões comuns de data
    date_patterns = [
        r'\b(20\d{2})\b',  # Ano
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(20\d{2})\b',  # Mês dia, ano
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(20\d{2})\b',  # Dia mês ano
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return ""