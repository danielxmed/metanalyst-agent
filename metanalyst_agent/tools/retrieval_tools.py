"""
Ferramentas de retrieval semântico usando FAISS e LLMs.
"""

from typing import Dict, Any, List, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import faiss
import numpy as np
import pickle
import os
import json
from datetime import datetime
from ..models.state import MetaAnalysisState
from ..models.config import config

# Inicializar modelos
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=config.openai_api_key
)

retrieval_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.1,
    api_key=config.openai_api_key
)

@tool("search_vector_store")
def search_vector_store(
    query: Annotated[str, "Query de busca para encontrar chunks relevantes"],
    vector_store_path: Annotated[str, "Caminho do vector store para buscar"],
    top_k: Annotated[int, "Número de chunks mais similares para retornar"] = 10,
    similarity_threshold: Annotated[float, "Threshold mínimo de similaridade (0-1)"] = 0.7,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Busca chunks similares no vector store usando similaridade cosine.
    """
    
    try:
        # Verificar se o vector store existe
        index_path = os.path.join(vector_store_path, "index.faiss")
        metadata_path = os.path.join(vector_store_path, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return {
                "success": False,
                "error": "Vector store não encontrado ou incompleto",
                "results": []
            }
        
        # Carregar índice FAISS
        index = faiss.read_index(index_path)
        
        # Carregar metadados
        with open(metadata_path, "rb") as f:
            metadata_list = pickle.load(f)
        
        # Gerar embedding da query
        query_embedding = embeddings_model.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalizar para cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Buscar chunks similares
        similarities, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        # Preparar resultados
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity >= similarity_threshold:
                result = {
                    "rank": i + 1,
                    "similarity_score": float(similarity),
                    "content": metadata_list[idx]["content"],
                    "metadata": metadata_list[idx]["metadata"],
                    "chunk_id": metadata_list[idx]["metadata"]["chunk_id"]
                }
                results.append(result)
        
        return {
            "success": True,
            "query": query,
            "total_found": len(results),
            "top_k_requested": top_k,
            "similarity_threshold": similarity_threshold,
            "results": results,
            "search_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }

@tool("retrieve_relevant_chunks")
def retrieve_relevant_chunks(
    pico_query: Annotated[str, "Query baseada no PICO para busca contextual"],
    vector_store_path: Annotated[str, "Caminho do vector store"],
    focus_areas: Annotated[List[str], "Áreas específicas de foco (ex: 'statistical_data', 'methodology')"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Recupera chunks relevantes usando múltiplas estratégias de busca baseadas em PICO.
    AI-first approach para busca inteligente e contextual.
    """
    
    try:
        focus_areas = focus_areas or ["statistical_data", "methodology", "outcomes", "population"]
        
        # Gerar queries específicas para cada área de foco
        query_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em recuperação de informações para meta-análises.

TAREFA: Gerar queries de busca específicas baseadas no PICO fornecido e áreas de foco.

PICO ORIGINAL: {pico_query}

ÁREAS DE FOCO: {focus_areas}

Para cada área de foco, gere uma query otimizada que capture informações específicas:
- statistical_data: buscar resultados numéricos, estatísticas, medidas de efeito
- methodology: buscar métodos, design do estudo, randomização
- outcomes: buscar desfechos, resultados clínicos
- population: buscar características dos participantes, critérios de inclusão

FORMATO DE SAÍDA: JSON com:
{{
  "queries": [
    {{"area": "statistical_data", "query": "texto da query"}},
    {{"area": "methodology", "query": "texto da query"}},
    ...
  ]
}}"""),
            ("human", "Gere as queries específicas.")
        ])
        
        # Gerar queries específicas
        chain = query_generation_prompt | retrieval_llm
        response = chain.invoke({
            "pico_query": pico_query,
            "focus_areas": focus_areas
        })
        
        # Parsear queries geradas
        try:
            queries_data = json.loads(response.content)
            specific_queries = queries_data.get("queries", [])
        except:
            # Fallback: usar query original para todas as áreas
            specific_queries = [{"area": area, "query": pico_query} for area in focus_areas]
        
        # Buscar para cada query específica
        all_results = {}
        unique_chunks = {}  # Para evitar duplicatas
        
        for query_info in specific_queries:
            area = query_info["area"]
            query = query_info["query"]
            
            # Buscar chunks para esta query
            search_result = search_vector_store(
                query=query,
                vector_store_path=vector_store_path,
                top_k=5,
                similarity_threshold=0.6,
                state=state
            )
            
            if search_result["success"]:
                all_results[area] = search_result["results"]
                
                # Adicionar chunks únicos
                for result in search_result["results"]:
                    chunk_id = result["chunk_id"]
                    if chunk_id not in unique_chunks or result["similarity_score"] > unique_chunks[chunk_id]["similarity_score"]:
                        unique_chunks[chunk_id] = result
        
        # Ordenar chunks únicos por relevância
        sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "success": True,
            "pico_query": pico_query,
            "focus_areas": focus_areas,
            "queries_generated": specific_queries,
            "results_by_area": all_results,
            "unique_chunks": sorted_chunks[:15],  # Top 15 chunks únicos
            "total_unique_chunks": len(sorted_chunks),
            "retrieval_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results_by_area": {},
            "unique_chunks": []
        }

@tool("rank_by_relevance")
def rank_by_relevance(
    chunks: Annotated[List[Dict[str, Any]], "Lista de chunks para re-ranking"],
    pico: Annotated[Dict[str, str], "Estrutura PICO para contexto de relevância"],
    ranking_criteria: Annotated[str, "Critérios específicos para ranking"] = "statistical_relevance",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Re-ranking inteligente de chunks usando LLM para avaliar relevância contextual.
    """
    
    ranking_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em avaliação de relevância para meta-análises médicas.

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

CRITÉRIO DE RANKING: {ranking_criteria}

TAREFA: Avaliar e rankear os chunks fornecidos por relevância para esta meta-análise.

CRITÉRIOS DE AVALIAÇÃO:
1. Relevância direta ao PICO (0-10)
2. Qualidade da informação (0-10)
3. Especificidade dos dados (0-10)
4. Utilidade para análise estatística (0-10)

Para cada chunk, forneça:
- Score total (0-40)
- Justificativa breve
- Categoria de informação (statistical, methodological, clinical, etc.)

FORMATO DE SAÍDA: JSON com array de objetos:
[
  {{
    "chunk_id": "id_do_chunk",
    "relevance_score": score_total,
    "category": "categoria",
    "justification": "justificativa",
    "key_information": "informação_chave_extraída"
  }}
]"""),
        ("human", "Chunks para avaliação:\n\n{chunks_summary}")
    ])
    
    try:
        # Preparar resumo dos chunks para o LLM
        chunks_summary = []
        for i, chunk in enumerate(chunks[:10]):  # Limitar a 10 chunks para eficiência
            summary = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "content_preview": chunk.get("content", "")[:500],  # Primeiros 500 chars
                "similarity_score": chunk.get("similarity_score", 0),
                "article_title": chunk.get("metadata", {}).get("article_title", "")
            }
            chunks_summary.append(summary)
        
        # Processar com LLM
        chain = ranking_prompt | retrieval_llm
        response = chain.invoke({
            "population": pico.get("P", ""),
            "intervention": pico.get("I", ""),
            "comparison": pico.get("C", ""),
            "outcome": pico.get("O", ""),
            "ranking_criteria": ranking_criteria,
            "chunks_summary": json.dumps(chunks_summary, indent=2)
        })
        
        # Parsear resposta
        try:
            rankings = json.loads(response.content)
        except:
            # Fallback: ordenar por similarity_score
            rankings = []
            for chunk in chunks:
                rankings.append({
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "relevance_score": chunk.get("similarity_score", 0) * 40,
                    "category": "unknown",
                    "justification": "Avaliação automática via similarity score",
                    "key_information": chunk.get("content", "")[:100]
                })
        
        # Combinar rankings com chunks originais
        ranked_chunks = []
        ranking_dict = {r["chunk_id"]: r for r in rankings}
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            if chunk_id in ranking_dict:
                enhanced_chunk = chunk.copy()
                enhanced_chunk["llm_ranking"] = ranking_dict[chunk_id]
                ranked_chunks.append(enhanced_chunk)
        
        # Ordenar por relevance_score
        ranked_chunks.sort(
            key=lambda x: x.get("llm_ranking", {}).get("relevance_score", 0), 
            reverse=True
        )
        
        return {
            "success": True,
            "ranking_criteria": ranking_criteria,
            "total_chunks_ranked": len(ranked_chunks),
            "ranked_chunks": ranked_chunks,
            "ranking_method": "llm_evaluation",
            "ranking_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ranked_chunks": chunks  # Retornar chunks originais em caso de erro
        }

@tool("extract_key_information")
def extract_key_information(
    relevant_chunks: Annotated[List[Dict[str, Any]], "Chunks relevantes para extração"],
    information_type: Annotated[str, "Tipo de informação para extrair"] = "statistical_data",
    pico: Annotated[Dict[str, str], "Contexto PICO para extração focada"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Extrai informações-chave dos chunks relevantes usando LLM especializado.
    """
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em extração de informações para meta-análises.

TIPO DE INFORMAÇÃO: {information_type}
CONTEXTO PICO: {pico_context}

TAREFA: Extrair e estruturar informações específicas dos chunks fornecidos.

TIPOS DE EXTRAÇÃO:
- statistical_data: números, estatísticas, medidas de efeito, IC, p-valores
- methodology: design, randomização, cegamento, critérios
- population: características, n, idade, gênero, critérios
- outcomes: desfechos primários/secundários, instrumentos de medida
- clinical_context: contexto clínico, relevância prática

FORMATO DE SAÍDA: JSON estruturado com informações extraídas e organizadas.

IMPORTANTE: Seja preciso e cite as fontes (chunk_id) para cada informação extraída."""),
        ("human", "Chunks para extração:\n\n{chunks_content}")
    ])
    
    try:
        # Preparar conteúdo dos chunks
        chunks_content = []
        for chunk in relevant_chunks[:8]:  # Limitar para eficiência
            content = {
                "chunk_id": chunk.get("chunk_id", "unknown"),
                "content": chunk.get("content", ""),
                "article_title": chunk.get("metadata", {}).get("article_title", ""),
                "similarity_score": chunk.get("similarity_score", 0)
            }
            chunks_content.append(content)
        
        pico_context = ""
        if pico:
            pico_context = f"P: {pico.get('P', '')}, I: {pico.get('I', '')}, C: {pico.get('C', '')}, O: {pico.get('O', '')}"
        
        # Processar com LLM
        chain = extraction_prompt | retrieval_llm
        response = chain.invoke({
            "information_type": information_type,
            "pico_context": pico_context,
            "chunks_content": json.dumps(chunks_content, indent=2)[:12000]  # Limitar tamanho
        })
        
        # Parsear resposta
        try:
            extracted_info = json.loads(response.content)
        except:
            # Fallback: estrutura básica
            extracted_info = {
                "extraction_type": information_type,
                "extracted_data": [],
                "summary": "Falha na extração estruturada",
                "source_chunks": [c["chunk_id"] for c in chunks_content]
            }
        
        return {
            "success": True,
            "information_type": information_type,
            "chunks_processed": len(chunks_content),
            "extracted_information": extracted_info,
            "extraction_method": "llm_processing",
            "extraction_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "extracted_information": {}
        }