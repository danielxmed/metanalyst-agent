"""
Ferramentas de pesquisa científica para busca de literatura médica.
Integração com PubMed, Cochrane Library e geração de queries PICO.
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from tavily import TavilyClient

from ..config.settings import settings


@tool
def generate_search_queries(
    population: str,
    intervention: str, 
    comparison: str,
    outcome: str
) -> List[str]:
    """
    Gera queries de busca otimizadas baseadas no framework PICO.
    
    Args:
        population: População alvo do estudo
        intervention: Intervenção sendo estudada
        comparison: Comparação ou controle
        outcome: Desfecho/resultado medido
    
    Returns:
        Lista de queries otimizadas para diferentes bases de dados
    """
    queries = []
    
    # Query básica PICO
    basic_query = f"({population}) AND ({intervention}) AND ({comparison}) AND ({outcome})"
    queries.append(basic_query)
    
    # Query para revisões sistemáticas
    systematic_review_query = f"({intervention}) AND ({outcome}) AND (systematic review OR meta-analysis)"
    queries.append(systematic_review_query)
    
    # Query para ensaios clínicos randomizados
    rct_query = f"({population}) AND ({intervention}) AND ({comparison}) AND (randomized controlled trial OR RCT)"
    queries.append(rct_query)
    
    # Query simplificada para maior recall
    simple_query = f"({intervention}) AND ({outcome})"
    queries.append(simple_query)
    
    # Query com termos MeSH (Medical Subject Headings)
    mesh_query = f'"{intervention}"[MeSH Terms] AND "{outcome}"[MeSH Terms]'
    queries.append(mesh_query)
    
    return queries


@tool
def search_pubmed(
    query: str,
    max_results: int = 50,
    study_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Busca artigos no PubMed usando a API Entrez.
    
    Args:
        query: Query de busca
        max_results: Número máximo de resultados
        study_types: Tipos de estudo preferidos (e.g., ["randomized controlled trial"])
    
    Returns:
        Lista de artigos com metadados
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Adicionar filtros de tipo de estudo se especificado
    if study_types:
        type_filter = " OR ".join([f'"{stype}"[Publication Type]' for stype in study_types])
        query = f"({query}) AND ({type_filter})"
    
    # Buscar IDs dos artigos
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml",
        "sort": "relevance"
    }
    
    try:
        search_response = requests.get(search_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(search_response.content)
        id_list = root.find(".//IdList")
        
        if id_list is None:
            return []
        
        pmids = [id_elem.text for id_elem in id_list.findall("Id")]
        
        if not pmids:
            return []
        
        # Buscar detalhes dos artigos
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_response.raise_for_status()
        
        # Parse artigos
        articles = []
        fetch_root = ET.fromstring(fetch_response.content)
        
        for article in fetch_root.findall(".//PubmedArticle"):
            try:
                # Extrair informações básicas
                medline_citation = article.find(".//MedlineCitation")
                pmid = medline_citation.find(".//PMID").text
                
                article_element = medline_citation.find(".//Article")
                title_element = article_element.find(".//ArticleTitle")
                title = title_element.text if title_element is not None else "Título não disponível"
                
                abstract_element = article_element.find(".//Abstract/AbstractText")
                abstract = abstract_element.text if abstract_element is not None else "Abstract não disponível"
                
                # Autores
                authors = []
                author_list = article_element.find(".//AuthorList")
                if author_list is not None:
                    for author in author_list.findall(".//Author"):
                        last_name = author.find(".//LastName")
                        first_name = author.find(".//ForeName")
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")
                
                # Journal info
                journal = article_element.find(".//Journal/Title")
                journal_name = journal.text if journal is not None else "Journal não identificado"
                
                # Ano de publicação
                pub_date = article_element.find(".//PubDate/Year")
                year = pub_date.text if pub_date is not None else "Ano não disponível"
                
                # URL do PubMed
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal_name,
                    "year": year,
                    "url": url,
                    "source": "PubMed",
                    "study_type": "Research Article"  # Poderia ser refinado
                })
                
            except Exception as e:
                continue  # Skip artigos com problemas de parsing
        
        return articles
        
    except Exception as e:
        return [{"error": f"Erro na busca PubMed: {str(e)}"}]


@tool
def search_cochrane(
    query: str,
    max_results: int = 25
) -> List[Dict[str, Any]]:
    """
    Busca revisões sistemáticas na Cochrane Library.
    
    Args:
        query: Query de busca
        max_results: Número máximo de resultados
    
    Returns:
        Lista de revisões Cochrane
    """
    # Usar Tavily para buscar na Cochrane (domínio específico)
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        
        # Buscar especificamente no domínio Cochrane
        response = client.search(
            query=f"{query} site:cochranelibrary.com",
            search_depth="advanced",
            max_results=max_results,
            include_domains=["cochranelibrary.com"],
            include_answer=False
        )
        
        cochrane_reviews = []
        for result in response.get("results", []):
            cochrane_reviews.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0),
                "source": "Cochrane Library",
                "study_type": "Systematic Review"
            })
        
        return cochrane_reviews
        
    except Exception as e:
        return [{"error": f"Erro na busca Cochrane: {str(e)}"}]


@tool 
def search_clinical_trials(
    condition: str,
    intervention: str,
    max_results: int = 30
) -> List[Dict[str, Any]]:
    """
    Busca ensaios clínicos em ClinicalTrials.gov.
    
    Args:
        condition: Condição médica
        intervention: Intervenção estudada
        max_results: Número máximo de resultados
    
    Returns:
        Lista de ensaios clínicos
    """
    try:
        # API do ClinicalTrials.gov
        base_url = "https://clinicaltrials.gov/api/query/study_fields"
        
        params = {
            "expr": f"{condition} AND {intervention}",
            "fields": "NCTId,BriefTitle,Condition,InterventionName,Phase,StudyType,PrimaryCompletionDate,OverallStatus",
            "min_rnk": 1,
            "max_rnk": max_results,
            "fmt": "json"
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        studies = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
        
        clinical_trials = []
        for study in studies:
            fields = study.get("Field", [])
            
            # Extrair campos específicos
            nct_id = next((f.get("FieldValue", [""])[0] for f in fields if f.get("FieldName") == "NCTId"), "")
            title = next((f.get("FieldValue", [""])[0] for f in fields if f.get("FieldName") == "BriefTitle"), "")
            condition_list = next((f.get("FieldValue", []) for f in fields if f.get("FieldName") == "Condition"), [])
            intervention_list = next((f.get("FieldValue", []) for f in fields if f.get("FieldName") == "InterventionName"), [])
            phase = next((f.get("FieldValue", [""])[0] for f in fields if f.get("FieldName") == "Phase"), "")
            status = next((f.get("FieldValue", [""])[0] for f in fields if f.get("FieldName") == "OverallStatus"), "")
            
            if nct_id:
                clinical_trials.append({
                    "nct_id": nct_id,
                    "title": title,
                    "conditions": condition_list,
                    "interventions": intervention_list,
                    "phase": phase,
                    "status": status,
                    "url": f"https://clinicaltrials.gov/ct2/show/{nct_id}",
                    "source": "ClinicalTrials.gov",
                    "study_type": "Clinical Trial"
                })
        
        return clinical_trials
        
    except Exception as e:
        return [{"error": f"Erro na busca Clinical Trials: {str(e)}"}]


@tool
def evaluate_article_relevance(
    article: Dict[str, Any],
    pico_criteria: Dict[str, str]
) -> Dict[str, Any]:
    """
    Avalia a relevância de um artigo baseado nos critérios PICO.
    
    Args:
        article: Dados do artigo
        pico_criteria: Critérios PICO {P, I, C, O}
    
    Returns:
        Avaliação de relevância com score e justificativa
    """
    try:
        title = article.get("title", "").lower()
        abstract = article.get("abstract", "").lower()
        content = f"{title} {abstract}"
        
        score = 0
        matches = {}
        
        # Verificar cada componente PICO
        for component, criteria in pico_criteria.items():
            if criteria.lower() in content:
                score += 25  # 25 pontos por componente PICO encontrado
                matches[component] = True
            else:
                matches[component] = False
        
        # Bonus para tipos de estudo de alta qualidade
        study_type = article.get("study_type", "").lower()
        if "systematic review" in study_type or "meta-analysis" in study_type:
            score += 20
        elif "randomized controlled trial" in study_type or "rct" in content:
            score += 15
        elif "clinical trial" in study_type:
            score += 10
        
        # Penalizar se for muito antigo (>10 anos)
        year = article.get("year", "")
        if year and year.isdigit():
            current_year = 2024
            if current_year - int(year) > 10:
                score -= 10
        
        # Normalizar score (0-100)
        score = max(0, min(100, score))
        
        return {
            "article_id": article.get("pmid", article.get("nct_id", article.get("url", ""))),
            "relevance_score": score,
            "pico_matches": matches,
            "recommendation": "include" if score >= 60 else "review" if score >= 40 else "exclude",
            "justification": f"Score: {score}/100. PICO matches: {sum(matches.values())}/4"
        }
        
    except Exception as e:
        return {
            "error": f"Erro na avaliação de relevância: {str(e)}",
            "relevance_score": 0,
            "recommendation": "review"
        }