"""
Ferramentas de processamento de artigos para extração, vetorização e estruturação de dados.
Integração com Tavily Extract, OpenAI Embeddings e processamento de texto.
"""

import uuid
import json
import re
from typing import List, Dict, Any, Optional, Annotated
from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup

from ..config.settings import settings


@tool
def extract_article_content(url: str) -> Dict[str, Any]:
    """
    Extrai conteúdo completo de artigo usando Tavily Extract API.
    
    Args:
        url: URL do artigo para extração
    
    Returns:
        Conteúdo extraído com metadados
    """
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        
        # Usar Tavily Extract para obter conteúdo limpo
        response = client.extract(url=url)
        
        # Fallback para requests + BeautifulSoup se Tavily falhar
        if not response or not response.get("content"):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            page_response = requests.get(url, headers=headers, timeout=30)
            page_response.raise_for_status()
            
            soup = BeautifulSoup(page_response.content, 'html.parser')
            
            # Remover scripts, styles, etc.
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extrair texto principal
            content = soup.get_text()
            content = re.sub(r'\s+', ' ', content).strip()
            
            response = {
                "content": content,
                "title": soup.title.string if soup.title else "",
                "url": url
            }
        
        return {
            "url": url,
            "title": response.get("title", ""),
            "content": response.get("content", ""),
            "raw_content": response.get("raw_content", ""),
            "extracted_at": datetime.now().isoformat(),
            "extraction_method": "tavily" if response.get("content") else "fallback",
            "success": True
        }
        
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "extracted_at": datetime.now().isoformat(),
            "success": False
        }


@tool
def extract_statistical_data(
    content: str,
    title: str,
    pico_criteria: Dict[str, str]
) -> Dict[str, Any]:
    """
    Extrai dados estatísticos relevantes ao PICO do conteúdo do artigo.
    
    Args:
        content: Conteúdo do artigo
        title: Título do artigo
        pico_criteria: Critérios PICO para guiar a extração
    
    Returns:
        Dados estatísticos estruturados
    """
    try:
        # Patterns para identificar dados estatísticos
        patterns = {
            "sample_size": [
                r"n\s*=\s*(\d+)",
                r"(\d+)\s*participants?",
                r"(\d+)\s*patients?",
                r"(\d+)\s*subjects?"
            ],
            "p_value": [
                r"p\s*[=<>]\s*([\d.]+)",
                r"P\s*[=<>]\s*([\d.]+)",
                r"p-value\s*[=<>]\s*([\d.]+)"
            ],
            "confidence_interval": [
                r"95%\s*CI[:\s]*([\d.-]+)\s*[-–]\s*([\d.-]+)",
                r"CI\s*95%[:\s]*([\d.-]+)\s*[-–]\s*([\d.-]+)",
                r"\[?([\d.-]+)\s*[-–]\s*([\d.-]+)\]?\s*95%"
            ],
            "odds_ratio": [
                r"OR\s*[=:]\s*([\d.]+)",
                r"odds ratio\s*[=:]\s*([\d.]+)"
            ],
            "risk_ratio": [
                r"RR\s*[=:]\s*([\d.]+)",
                r"risk ratio\s*[=:]\s*([\d.]+)",
                r"relative risk\s*[=:]\s*([\d.]+)"
            ],
            "mean_difference": [
                r"MD\s*[=:]\s*([-\d.]+)",
                r"mean difference\s*[=:]\s*([-\d.]+)"
            ],
            "standard_deviation": [
                r"SD\s*[=:]\s*([\d.]+)",
                r"std\s*[=:]\s*([\d.]+)",
                r"standard deviation\s*[=:]\s*([\d.]+)"
            ],
            "effect_size": [
                r"effect size\s*[=:]\s*([\d.]+)",
                r"Cohen's d\s*[=:]\s*([\d.]+)",
                r"d\s*=\s*([\d.]+)"
            ]
        }
        
        extracted_data = {}
        
        # Extrair cada tipo de dado estatístico
        for data_type, pattern_list in patterns.items():
            values = []
            for pattern in pattern_list:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if data_type == "confidence_interval":
                        # CI tem dois valores
                        for match in matches:
                            if isinstance(match, tuple) and len(match) == 2:
                                try:
                                    lower = float(match[0])
                                    upper = float(match[1])
                                    values.append([lower, upper])
                                except ValueError:
                                    continue
                    else:
                        # Outros dados têm um valor
                        for match in matches:
                            try:
                                value = float(match) if isinstance(match, str) else float(match[0])
                                values.append(value)
                            except (ValueError, IndexError):
                                continue
            
            if values:
                extracted_data[data_type] = values
        
        # Extrair informações sobre grupos/arms do estudo
        groups_info = extract_study_groups(content)
        
        # Extrair desfechos (outcomes)
        outcomes = extract_outcomes(content, pico_criteria.get("O", ""))
        
        # Determinar tipo de estudo
        study_type = determine_study_type(content, title)
        
        return {
            "statistical_data": extracted_data,
            "study_groups": groups_info,
            "outcomes": outcomes,
            "study_type": study_type,
            "extraction_confidence": calculate_extraction_confidence(extracted_data),
            "extracted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na extração de dados estatísticos: {str(e)}",
            "statistical_data": {},
            "extracted_at": datetime.now().isoformat()
        }


def extract_study_groups(content: str) -> List[Dict[str, Any]]:
    """Extrai informações sobre grupos de estudo."""
    groups = []
    
    # Patterns para identificar grupos
    group_patterns = [
        r"intervention group[:\s]*([^.]+)",
        r"control group[:\s]*([^.]+)",
        r"treatment group[:\s]*([^.]+)",
        r"placebo group[:\s]*([^.]+)",
        r"experimental group[:\s]*([^.]+)"
    ]
    
    for pattern in group_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            groups.append({
                "description": match.strip(),
                "type": "intervention" if "intervention" in pattern or "treatment" in pattern or "experimental" in pattern else "control"
            })
    
    return groups


def extract_outcomes(content: str, primary_outcome: str) -> List[Dict[str, Any]]:
    """Extrai informações sobre desfechos do estudo."""
    outcomes = []
    
    # Buscar pelo desfecho primário especificado no PICO
    if primary_outcome:
        outcome_mentions = re.findall(
            rf"{re.escape(primary_outcome.lower())}[^.]*",
            content.lower()
        )
        for mention in outcome_mentions:
            outcomes.append({
                "outcome": primary_outcome,
                "type": "primary",
                "description": mention.strip()
            })
    
    # Buscar por outros desfechos comuns
    outcome_patterns = [
        r"primary outcome[:\s]*([^.]+)",
        r"secondary outcome[:\s]*([^.]+)",
        r"endpoint[:\s]*([^.]+)"
    ]
    
    for pattern in outcome_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            outcomes.append({
                "outcome": match.strip(),
                "type": "primary" if "primary" in pattern else "secondary",
                "description": match.strip()
            })
    
    return outcomes


def determine_study_type(content: str, title: str) -> str:
    """Determina o tipo de estudo baseado no conteúdo."""
    text = f"{title} {content}".lower()
    
    if any(term in text for term in ["systematic review", "meta-analysis"]):
        return "systematic_review"
    elif any(term in text for term in ["randomized controlled trial", "rct", "randomized"]):
        return "randomized_controlled_trial"
    elif any(term in text for term in ["cohort study", "prospective study"]):
        return "cohort_study"
    elif any(term in text for term in ["case-control", "case control"]):
        return "case_control_study"
    elif any(term in text for term in ["cross-sectional", "survey"]):
        return "cross_sectional_study"
    else:
        return "observational_study"


def calculate_extraction_confidence(extracted_data: Dict[str, Any]) -> float:
    """Calcula confiança na extração baseada na quantidade de dados encontrados."""
    total_fields = len(extracted_data)
    if total_fields == 0:
        return 0.0
    
    # Campos importantes para meta-análise
    important_fields = ["sample_size", "p_value", "confidence_interval", "effect_size"]
    important_found = sum(1 for field in important_fields if field in extracted_data)
    
    confidence = (important_found / len(important_fields)) * 0.7 + (total_fields / 10) * 0.3
    return min(1.0, confidence)


@tool
def generate_vancouver_citation(
    title: str,
    authors: List[str],
    journal: str,
    year: str,
    volume: Optional[str] = None,
    issue: Optional[str] = None,
    pages: Optional[str] = None,
    doi: Optional[str] = None,
    pmid: Optional[str] = None
) -> str:
    """
    Gera citação no formato Vancouver.
    
    Args:
        title: Título do artigo
        authors: Lista de autores
        journal: Nome do periódico
        year: Ano de publicação
        volume: Volume (opcional)
        issue: Número/issue (opcional)
        pages: Páginas (opcional)
        doi: DOI (opcional)
        pmid: PMID (opcional)
    
    Returns:
        Citação formatada no estilo Vancouver
    """
    try:
        # Formatar autores (máximo 6, depois et al.)
        if len(authors) <= 6:
            author_str = ", ".join(authors)
        else:
            author_str = ", ".join(authors[:6]) + ", et al."
        
        # Construir citação básica
        citation = f"{author_str}. {title}. {journal}."
        
        # Adicionar ano
        if year:
            citation += f" {year}"
        
        # Adicionar volume e issue
        if volume:
            citation += f";{volume}"
            if issue:
                citation += f"({issue})"
        
        # Adicionar páginas
        if pages:
            citation += f":{pages}"
        
        # Adicionar ponto final se não terminar com um
        if not citation.endswith('.'):
            citation += "."
        
        # Adicionar DOI se disponível
        if doi:
            citation += f" doi: {doi}"
        
        # Adicionar PMID se disponível
        if pmid:
            citation += f" PMID: {pmid}"
        
        return citation
        
    except Exception as e:
        return f"Erro na geração da citação: {str(e)}"


@tool
def chunk_and_vectorize(
    content: str,
    url: str,
    metadata: Dict[str, Any],
    store: Annotated[BaseStore, InjectedStore]
) -> Dict[str, Any]:
    """
    Cria chunks do conteúdo e gera embeddings para armazenamento no vector store.
    
    Args:
        content: Conteúdo do artigo
        url: URL do artigo
        metadata: Metadados do artigo
        store: Store para persistência
    
    Returns:
        Informações sobre o processo de chunking e vetorização
    """
    try:
        # Configurar text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Criar chunks
        chunks = text_splitter.split_text(content)
        
        # Configurar embeddings
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Gerar embeddings para cada chunk
        chunk_embeddings = embeddings.embed_documents(chunks)
        
        # Namespace para organização no store
        namespace = ("metanalysis", "articles", metadata.get("study_type", "general"))
        
        # Armazenar cada chunk no store
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            chunk_data = {
                "content": chunk,
                "embedding": embedding,
                "metadata": {
                    **metadata,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_id": chunk_id,
                    "vectorized_at": datetime.now().isoformat()
                },
                "summary": chunk[:200] + "..." if len(chunk) > 200 else chunk
            }
            
            store.put(namespace, chunk_id, chunk_data)
        
        return {
            "success": True,
            "total_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "namespace": namespace,
            "embedding_model": settings.embedding_model,
            "vectorized_at": datetime.now().isoformat(),
            "url": url
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro na vetorização: {str(e)}",
            "url": url,
            "vectorized_at": datetime.now().isoformat()
        }


@tool
def process_article_batch(
    urls: List[str],
    pico_criteria: Dict[str, str],
    store: Annotated[BaseStore, InjectedStore]
) -> Dict[str, Any]:
    """
    Processa um lote de artigos em paralelo para maior eficiência.
    
    Args:
        urls: Lista de URLs para processar
        pico_criteria: Critérios PICO para extração
        store: Store para persistência
    
    Returns:
        Resultados do processamento em lote
    """
    results = {
        "processed": [],
        "failed": [],
        "total_urls": len(urls),
        "processing_started": datetime.now().isoformat()
    }
    
    for url in urls:
        try:
            # Extrair conteúdo
            extraction_result = extract_article_content(url)
            
            if not extraction_result.get("success"):
                results["failed"].append({
                    "url": url,
                    "error": extraction_result.get("error", "Falha na extração")
                })
                continue
            
            # Extrair dados estatísticos
            statistical_data = extract_statistical_data(
                extraction_result["content"],
                extraction_result["title"],
                pico_criteria
            )
            
            # Vetorizar conteúdo
            vectorization_result = chunk_and_vectorize(
                extraction_result["content"],
                url,
                {
                    "title": extraction_result["title"],
                    "statistical_data": statistical_data,
                    "pico_criteria": pico_criteria
                },
                store
            )
            
            results["processed"].append({
                "url": url,
                "title": extraction_result["title"],
                "chunks_created": vectorization_result.get("total_chunks", 0),
                "statistical_data": statistical_data,
                "extraction_confidence": statistical_data.get("extraction_confidence", 0)
            })
            
        except Exception as e:
            results["failed"].append({
                "url": url,
                "error": str(e)
            })
    
    results["processing_completed"] = datetime.now().isoformat()
    results["success_rate"] = len(results["processed"]) / len(urls) if urls else 0
    
    return results