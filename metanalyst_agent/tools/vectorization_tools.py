"""
Ferramentas de vetorização e armazenamento usando OpenAI embeddings e FAISS.
"""

from typing import Dict, Any, List, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle
import uuid
import os
from datetime import datetime
from ..models.state import MetaAnalysisState
from ..models.config import config

# Inicializar embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=config.openai_api_key
)

# Splitter para chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

@tool("create_text_chunks")
def create_text_chunks(
    content: Annotated[str, "Conteúdo completo do artigo para dividir em chunks"],
    article_metadata: Annotated[Dict[str, Any], "Metadados do artigo para contexto"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Cria chunks inteligentes do conteúdo do artigo mantendo contexto.
    """
    
    try:
        # Dividir texto em chunks
        chunks = text_splitter.split_text(content)
        
        # Adicionar metadados aos chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "article_url": article_metadata.get("url", ""),
                "article_title": article_metadata.get("title", ""),
                "article_authors": article_metadata.get("authors", []),
                "article_doi": article_metadata.get("doi", ""),
                "chunk_length": len(chunk),
                "created_at": datetime.now().isoformat()
            }
            
            enhanced_chunks.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return {
            "success": True,
            "total_chunks": len(enhanced_chunks),
            "chunks": enhanced_chunks,
            "average_chunk_size": sum(len(c["content"]) for c in enhanced_chunks) / len(enhanced_chunks),
            "processing_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_chunks": 0,
            "chunks": []
        }

@tool("generate_embeddings")
def generate_embeddings(
    chunks: Annotated[List[Dict[str, Any]], "Lista de chunks com metadados para gerar embeddings"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Gera embeddings para os chunks usando OpenAI text-embedding-3-small.
    """
    
    try:
        if not chunks:
            return {
                "success": False,
                "error": "Lista de chunks vazia",
                "embeddings": []
            }
        
        # Extrair textos dos chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Gerar embeddings em batch
        embeddings = embeddings_model.embed_documents(texts)
        
        # Combinar embeddings com chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding
            chunk_with_embedding["embedding_model"] = "text-embedding-3-small"
            chunk_with_embedding["embedding_dims"] = len(embedding)
            chunks_with_embeddings.append(chunk_with_embedding)
        
        return {
            "success": True,
            "total_embeddings": len(chunks_with_embeddings),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "chunks_with_embeddings": chunks_with_embeddings,
            "model_used": "text-embedding-3-small",
            "generation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "embeddings": []
        }

@tool("store_in_vector_db")
def store_in_vector_db(
    chunks_with_embeddings: Annotated[List[Dict[str, Any]], "Chunks com embeddings para armazenar"],
    vector_store_path: Annotated[str, "Caminho para salvar o vector store"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Armazena embeddings em FAISS vector database local.
    """
    
    try:
        if not chunks_with_embeddings:
            return {
                "success": False,
                "error": "Lista de chunks com embeddings vazia"
            }
        
        # Preparar dados
        embeddings_array = np.array([chunk["embedding"] for chunk in chunks_with_embeddings])
        dimensions = embeddings_array.shape[1]
        
        # Criar índice FAISS
        index = faiss.IndexFlatIP(dimensions)  # Inner Product (cosine similarity)
        
        # Normalizar embeddings para cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Adicionar embeddings ao índice
        index.add(embeddings_array.astype('float32'))
        
        # Preparar metadados (sem embeddings para economizar espaço)
        metadata_list = []
        for chunk in chunks_with_embeddings:
            metadata = chunk.copy()
            del metadata["embedding"]  # Remover embedding dos metadados
            metadata_list.append(metadata)
        
        # Definir caminho de armazenamento
        if not vector_store_path:
            store_id = state.get("meta_analysis_id", str(uuid.uuid4()))
            vector_store_path = f"/tmp/vectorstore_{store_id}"
        
        # Criar diretório se não existir
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Salvar índice FAISS
        index_path = os.path.join(vector_store_path, "index.faiss")
        faiss.write_index(index, index_path)
        
        # Salvar metadados
        metadata_path = os.path.join(vector_store_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_list, f)
        
        # Salvar informações do store
        store_info = {
            "store_id": os.path.basename(vector_store_path),
            "total_vectors": len(chunks_with_embeddings),
            "dimensions": dimensions,
            "index_type": "IndexFlatIP",
            "created_at": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-small"
        }
        
        info_path = os.path.join(vector_store_path, "store_info.json")
        import json
        with open(info_path, "w") as f:
            json.dump(store_info, f, indent=2)
        
        return {
            "success": True,
            "vector_store_path": vector_store_path,
            "store_id": store_info["store_id"],
            "total_vectors": store_info["total_vectors"],
            "dimensions": dimensions,
            "index_path": index_path,
            "metadata_path": metadata_path,
            "store_info": store_info
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool("setup_vector_store")
def setup_vector_store(
    meta_analysis_id: Annotated[str, "ID da meta-análise para criar store"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Configura um novo vector store para a meta-análise.
    """
    
    try:
        # Criar diretório para o vector store
        store_path = f"/tmp/vectorstore_{meta_analysis_id}"
        os.makedirs(store_path, exist_ok=True)
        
        # Inicializar informações do store
        store_info = {
            "store_id": meta_analysis_id,
            "total_vectors": 0,
            "dimensions": 1536,  # text-embedding-3-small
            "index_type": "IndexFlatIP",
            "created_at": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-small",
            "status": "initialized"
        }
        
        # Salvar informações
        info_path = os.path.join(store_path, "store_info.json")
        import json
        with open(info_path, "w") as f:
            json.dump(store_info, f, indent=2)
        
        return {
            "success": True,
            "vector_store_path": store_path,
            "store_id": meta_analysis_id,
            "status": "initialized",
            "ready_for_vectors": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool("load_vector_store")
def load_vector_store(
    vector_store_path: Annotated[str, "Caminho do vector store para carregar"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Carrega um vector store existente do disco.
    """
    
    try:
        # Verificar se o store existe
        if not os.path.exists(vector_store_path):
            return {
                "success": False,
                "error": f"Vector store não encontrado: {vector_store_path}"
            }
        
        # Carregar informações do store
        info_path = os.path.join(vector_store_path, "store_info.json")
        if os.path.exists(info_path):
            import json
            with open(info_path, "r") as f:
                store_info = json.load(f)
        else:
            store_info = {"status": "unknown"}
        
        # Verificar se o índice existe
        index_path = os.path.join(vector_store_path, "index.faiss")
        metadata_path = os.path.join(vector_store_path, "metadata.pkl")
        
        index_exists = os.path.exists(index_path)
        metadata_exists = os.path.exists(metadata_path)
        
        return {
            "success": True,
            "vector_store_path": vector_store_path,
            "store_info": store_info,
            "index_exists": index_exists,
            "metadata_exists": metadata_exists,
            "ready_for_search": index_exists and metadata_exists,
            "total_vectors": store_info.get("total_vectors", 0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }