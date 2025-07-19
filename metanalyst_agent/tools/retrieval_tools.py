"""
Retrieval tools for semantic search in the vector store.
AI-first approach using embeddings and FAISS for intelligent information retrieval.
"""

import json
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
import numpy as np
import faiss

from ..config.settings import settings


# Initialize embeddings
embeddings = OpenAIEmbeddings(**settings.get_embedding_config())


@tool
def search_vector_store(
    query: str,
    store_id: str,
    top_k: int = 10,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Search the vector store for relevant chunks based on semantic similarity.
    
    Args:
        query: Search query
        store_id: ID of the vector store to search
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score
    
    Returns:
        List of relevant chunks with metadata
    """
    
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Load FAISS index
        store_path = os.path.join(settings.faiss_index_path, store_id)
        index_file = os.path.join(store_path, "index.faiss")
        metadata_file = os.path.join(store_path, "metadata.json")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            return [{"error": f"Vector store {store_id} not found"}]
        
        # Load index and metadata
        index = faiss.read_index(index_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Perform search
        similarities, indices = index.search(query_vector, top_k)
        
        # Process results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= similarity_threshold and idx < len(metadata):
                chunk_metadata = metadata[idx]
                result = {
                    "chunk_id": chunk_metadata.get("chunk_id"),
                    "content": chunk_metadata.get("content", ""),
                    "similarity_score": float(similarity),
                    "rank": i + 1,
                    "metadata": {
                        "article_url": chunk_metadata.get("article_url"),
                        "article_title": chunk_metadata.get("article_title"),
                        "article_author": chunk_metadata.get("article_author"),
                        "chunk_length": chunk_metadata.get("chunk_length"),
                        "chunk_position": chunk_metadata.get("chunk_position"),
                    }
                }
                results.append(result)
        
        return results
        
    except Exception as e:
        return [{"error": f"Vector search failed: {str(e)}"}]


@tool
def retrieve_relevant_chunks(
    pico_query: Dict[str, str],
    store_id: str,
    chunks_per_element: int = 5
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for each PICO element.
    
    Args:
        pico_query: PICO framework with search terms
        store_id: ID of the vector store
        chunks_per_element: Number of chunks to retrieve per PICO element
    
    Returns:
        Organized chunks by PICO element
    """
    
    try:
        results = {
            "population_chunks": [],
            "intervention_chunks": [],
            "comparison_chunks": [],
            "outcome_chunks": [],
            "combined_chunks": [],
            "retrieval_summary": {}
        }
        
        # Search for each PICO element
        pico_elements = {
            "population": pico_query.get("P", ""),
            "intervention": pico_query.get("I", ""),
            "comparison": pico_query.get("C", ""),
            "outcome": pico_query.get("O", "")
        }
        
        total_chunks_found = 0
        
        for element_name, query_text in pico_elements.items():
            if query_text:
                chunks = search_vector_store(
                    query=query_text,
                    store_id=store_id,
                    top_k=chunks_per_element,
                    similarity_threshold=0.6
                )
                
                # Filter out error results
                valid_chunks = [c for c in chunks if "error" not in c]
                results[f"{element_name}_chunks"] = valid_chunks
                total_chunks_found += len(valid_chunks)
        
        # Combined search with full PICO
        combined_query = " AND ".join([
            f"{k}: {v}" for k, v in pico_elements.items() if v
        ])
        
        if combined_query:
            combined_chunks = search_vector_store(
                query=combined_query,
                store_id=store_id,
                top_k=chunks_per_element * 2,
                similarity_threshold=0.7
            )
            results["combined_chunks"] = [c for c in combined_chunks if "error" not in c]
            total_chunks_found += len(results["combined_chunks"])
        
        # Summary
        results["retrieval_summary"] = {
            "total_chunks_retrieved": total_chunks_found,
            "population_count": len(results["population_chunks"]),
            "intervention_count": len(results["intervention_chunks"]),
            "comparison_count": len(results["comparison_chunks"]),
            "outcome_count": len(results["outcome_chunks"]),
            "combined_count": len(results["combined_chunks"]),
            "retrieval_quality": calculate_retrieval_quality(results)
        }
        
        return results
        
    except Exception as e:
        return {
            "error": f"PICO retrieval failed: {str(e)}",
            "retrieval_summary": {"total_chunks_retrieved": 0}
        }


@tool
def search_for_statistical_data(
    statistical_terms: List[str],
    store_id: str,
    top_k: int = 15
) -> List[Dict[str, Any]]:
    """
    Search for chunks containing statistical data and results.
    
    Args:
        statistical_terms: Terms related to statistics (e.g., "effect size", "p-value")
        store_id: ID of the vector store
        top_k: Number of results to return
    
    Returns:
        List of chunks containing statistical information
    """
    
    try:
        all_results = []
        
        # Common statistical terms if none provided
        if not statistical_terms:
            statistical_terms = [
                "effect size",
                "confidence interval",
                "p-value",
                "odds ratio",
                "mean difference",
                "standard deviation",
                "sample size",
                "statistical significance"
            ]
        
        # Search for each statistical term
        for term in statistical_terms:
            results = search_vector_store(
                query=term,
                store_id=store_id,
                top_k=top_k // len(statistical_terms),
                similarity_threshold=0.6
            )
            
            # Add search term to results
            for result in results:
                if "error" not in result:
                    result["search_term"] = term
                    all_results.append(result)
        
        # Remove duplicates based on chunk_id
        unique_results = []
        seen_chunks = set()
        
        for result in all_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return unique_results[:top_k]
        
    except Exception as e:
        return [{"error": f"Statistical data search failed: {str(e)}"}]


@tool
def find_study_methodologies(
    methodology_terms: List[str],
    store_id: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for chunks describing study methodologies and designs.
    
    Args:
        methodology_terms: Terms related to study methodology
        store_id: ID of the vector store
        top_k: Number of results to return
    
    Returns:
        List of chunks with methodology information
    """
    
    try:
        if not methodology_terms:
            methodology_terms = [
                "randomized controlled trial",
                "double blind",
                "placebo controlled",
                "inclusion criteria",
                "exclusion criteria",
                "randomization",
                "blinding",
                "methodology",
                "study design"
            ]
        
        all_results = []
        
        for term in methodology_terms:
            results = search_vector_store(
                query=term,
                store_id=store_id,
                top_k=top_k // len(methodology_terms),
                similarity_threshold=0.65
            )
            
            for result in results:
                if "error" not in result:
                    result["methodology_term"] = term
                    all_results.append(result)
        
        # Remove duplicates and sort
        unique_results = []
        seen_chunks = set()
        
        for result in all_results:
            chunk_id = result.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return unique_results[:top_k]
        
    except Exception as e:
        return [{"error": f"Methodology search failed: {str(e)}"}]


@tool
def get_vector_store_stats(store_id: str) -> Dict[str, Any]:
    """
    Get statistics about the vector store.
    
    Args:
        store_id: ID of the vector store
    
    Returns:
        Statistics about the vector store
    """
    
    try:
        store_path = os.path.join(settings.faiss_index_path, store_id)
        index_file = os.path.join(store_path, "index.faiss")
        metadata_file = os.path.join(store_path, "metadata.json")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            return {"error": f"Vector store {store_id} not found"}
        
        # Load index and metadata
        index = faiss.read_index(index_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Calculate statistics
        total_chunks = len(metadata)
        total_vectors = index.ntotal
        vector_dimension = index.d
        
        # Analyze articles
        articles = set()
        chunk_lengths = []
        
        for chunk_meta in metadata:
            if chunk_meta.get("article_url"):
                articles.add(chunk_meta["article_url"])
            if chunk_meta.get("chunk_length"):
                chunk_lengths.append(chunk_meta["chunk_length"])
        
        avg_chunk_length = np.mean(chunk_lengths) if chunk_lengths else 0
        
        return {
            "store_id": store_id,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
            "vector_dimension": vector_dimension,
            "unique_articles": len(articles),
            "average_chunk_length": float(avg_chunk_length),
            "chunks_per_article": total_chunks / len(articles) if articles else 0,
            "store_size_mb": (os.path.getsize(index_file) + os.path.getsize(metadata_file)) / (1024 * 1024)
        }
        
    except Exception as e:
        return {"error": f"Failed to get store stats: {str(e)}"}


def calculate_retrieval_quality(results: Dict[str, Any]) -> float:
    """Calculate quality score for retrieval results."""
    
    try:
        total_chunks = results["retrieval_summary"]["total_chunks_retrieved"]
        
        if total_chunks == 0:
            return 0.0
        
        # Check coverage of PICO elements
        pico_coverage = 0
        for element in ["population", "intervention", "comparison", "outcome"]:
            if results[f"{element}_chunks"]:
                pico_coverage += 0.25
        
        # Check similarity scores
        all_chunks = []
        for key in results:
            if key.endswith("_chunks") and isinstance(results[key], list):
                all_chunks.extend(results[key])
        
        if all_chunks:
            avg_similarity = np.mean([
                chunk.get("similarity_score", 0) for chunk in all_chunks
                if "similarity_score" in chunk
            ])
        else:
            avg_similarity = 0
        
        # Combined quality score
        quality_score = (pico_coverage * 0.6) + (avg_similarity * 0.4)
        
        return min(quality_score, 1.0)
        
    except Exception:
        return 0.5  # Default moderate quality