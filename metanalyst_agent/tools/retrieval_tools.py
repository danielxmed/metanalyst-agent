"""
Ferramentas de busca semântica no vector store para recuperação de informações relevantes.
"""

from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
import numpy as np
from datetime import datetime

from ..config.settings import settings


@tool
def search_vector_store(
    query: str,
    namespace_filter: Optional[str] = None,
    limit: int = 10,
    store: Annotated[BaseStore, InjectedStore]
) -> List[Dict[str, Any]]:
    """
    Busca semântica no vector store usando embeddings.
    
    Args:
        query: Query de busca
        namespace_filter: Filtro por namespace (opcional)
        limit: Número máximo de resultados
        store: Store para busca
    
    Returns:
        Lista de chunks relevantes com metadados
    """
    try:
        # Configurar namespace
        if namespace_filter:
            namespace = ("metanalysis", namespace_filter)
        else:
            namespace = ("metanalysis",)
        
        # Buscar usando store com embeddings
        results = store.search(
            namespace,
            query=query,
            limit=limit
        )
        
        # Processar resultados
        relevant_chunks = []
        for result in results:
            chunk_data = result.value
            relevant_chunks.append({
                "content": chunk_data.get("content", ""),
                "metadata": chunk_data.get("metadata", {}),
                "summary": chunk_data.get("summary", ""),
                "score": result.score if hasattr(result, 'score') else 0,
                "chunk_id": result.key,
                "namespace": result.namespace
            })
        
        return relevant_chunks
        
    except Exception as e:
        return [{"error": f"Erro na busca semântica: {str(e)}"}]


@tool
def search_by_pico(
    pico_criteria: Dict[str, str],
    study_types: Optional[List[str]] = None,
    limit: int = 15,
    store: Annotated[BaseStore, InjectedStore]
) -> Dict[str, Any]:
    """
    Busca informações baseadas nos critérios PICO.
    
    Args:
        pico_criteria: Critérios PICO {P, I, C, O}
        study_types: Tipos de estudo para filtrar
        limit: Número máximo de resultados
        store: Store para busca
    
    Returns:
        Resultados organizados por componente PICO
    """
    try:
        results = {
            "population": [],
            "intervention": [],
            "comparison": [],
            "outcome": [],
            "combined": []
        }
        
        # Buscar por cada componente PICO
        for component, criteria in pico_criteria.items():
            if criteria:
                component_results = search_vector_store(
                    query=criteria,
                    limit=limit // 4,
                    store=store
                )
                
                component_name = {
                    "P": "population",
                    "I": "intervention", 
                    "C": "comparison",
                    "O": "outcome"
                }.get(component, component.lower())
                
                results[component_name] = component_results
        
        # Busca combinada
        combined_query = " AND ".join([
            f"{criteria}" for criteria in pico_criteria.values() if criteria
        ])
        
        if combined_query:
            combined_results = search_vector_store(
                query=combined_query,
                limit=limit,
                store=store
            )
            results["combined"] = combined_results
        
        # Filtrar por tipo de estudo se especificado
        if study_types:
            for key in results:
                results[key] = [
                    chunk for chunk in results[key]
                    if any(study_type.lower() in chunk.get("metadata", {}).get("study_type", "").lower()
                          for study_type in study_types)
                ]
        
        return {
            "pico_results": results,
            "total_chunks": sum(len(chunks) for chunks in results.values()),
            "search_criteria": pico_criteria,
            "study_type_filter": study_types,
            "searched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na busca PICO: {str(e)}",
            "searched_at": datetime.now().isoformat()
        }


@tool
def get_relevant_chunks(
    topic: str,
    context: str,
    min_score: float = 0.7,
    store: Annotated[BaseStore, InjectedStore]
) -> List[Dict[str, Any]]:
    """
    Recupera chunks relevantes para um tópico específico com contexto.
    
    Args:
        topic: Tópico principal de interesse
        context: Contexto adicional para refinar a busca
        min_score: Score mínimo de similaridade
        store: Store para busca
    
    Returns:
        Lista de chunks filtrados por relevância
    """
    try:
        # Combinar tópico e contexto
        enhanced_query = f"{topic} {context}"
        
        # Buscar chunks
        all_results = search_vector_store(
            query=enhanced_query,
            limit=30,
            store=store
        )
        
        # Filtrar por score mínimo
        relevant_chunks = [
            chunk for chunk in all_results
            if chunk.get("score", 0) >= min_score
        ]
        
        # Ordenar por relevância
        relevant_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return relevant_chunks[:15]  # Top 15 mais relevantes
        
    except Exception as e:
        return [{"error": f"Erro na recuperação de chunks: {str(e)}"}]