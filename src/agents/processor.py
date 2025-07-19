"""
Agente Processador - Combina extração de dados e vetorização.
Responsável por extrair conteúdo, processar dados estruturados e criar embeddings.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store import BaseStore

from src.models.state import MetaAnalysisState, add_agent_log
from src.tools.tavily_tools import TavilyTools
from src.tools.processing_tools import ProcessingTools
from src.utils.config import Config

logger = logging.getLogger(__name__)


class ProcessorAgent:
    """
    Agente Processador.
    
    Combina extração de conteúdo usando Tavily Extract, processamento 
    de dados estruturados com LLM e criação de embeddings vetoriais.
    """
    
    def __init__(self):
        """Inicializa o agente processador."""
        self.tavily = TavilyTools()
        self.processor = ProcessingTools()
        self.name = "processor"
        self.processing_config = Config.get_processing_config()
    
    def process_articles(
        self,
        state: MetaAnalysisState,
        config: RunnableConfig = None,
        *,
        store: BaseStore
    ) -> Dict[str, Any]:
        """
        Processa artigos da fila de processamento.
        
        Args:
            state: Estado atual da meta-análise
            config: Configuração do LangGraph
            store: Store para persistência de longo prazo
            
        Returns:
            Atualizações do estado
        """
        start_time = time.time()
        
        try:
            processing_queue = state.get("processing_queue", [])
            
            if not processing_queue:
                logger.info("Nenhum artigo na fila de processamento")
                return self._handle_no_articles(state)
            
            # Processar próximo artigo da fila
            url = processing_queue[0]
            logger.info(f"Processando artigo: {url}")
            
            # Extrair conteúdo
            content = self.tavily.extract_content(url)
            if not content:
                return self._handle_extraction_failure(state, url)
            
            # Extrair dados estruturados
            pico = state.get("pico", {})
            extracted_study = self.processor.extract_structured_data(content, pico)
            if not extracted_study:
                return self._handle_extraction_failure(state, url)
            
            # Criar chunks vetorizados
            chunks = self.processor.create_vector_chunks(extracted_study, content)
            
            # Armazenar no store de longo prazo
            article_id = self._store_article_data(
                store, state["meta_analysis_id"], url, content, 
                extracted_study, chunks
            )
            
            # Atualizar estado
            processed_articles = state.get("processed_articles", [])
            processed_articles.append({
                "id": article_id,
                "url": url,
                "title": extracted_study.characteristics.title,
                "authors": extracted_study.characteristics.authors,
                "year": extracted_study.characteristics.year,
                "study_type": extracted_study.characteristics.study_type.value,
                "sample_size": extracted_study.characteristics.sample_size,
                "quality_score": extracted_study.quality_assessment.quality_score,
                "confidence_score": extracted_study.confidence_score,
                "outcomes_count": len(extracted_study.outcomes),
                "chunks_count": len(chunks)
            })
            
            # Remover da fila de processamento
            remaining_queue = processing_queue[1:]
            
            execution_time = time.time() - start_time
            
            logger.info(f"Artigo processado com sucesso: {len(chunks)} chunks criados")
            
            return {
                "processing_queue": remaining_queue,
                "processed_articles": processed_articles,
                "chunk_count": state.get("chunk_count", 0) + len(chunks),
                "total_articles_processed": len(processed_articles),
                "execution_time": {
                    **state.get("execution_time", {}),
                    "processing": state.get("execution_time", {}).get("processing", 0) + execution_time
                },
                "messages": state["messages"] + [
                    AIMessage(f"Artigo processado: {extracted_study.characteristics.title[:100]}...")
                ],
                **add_agent_log(
                    state,
                    self.name,
                    "article_processed",
                    {
                        "url": url,
                        "chunks_created": len(chunks),
                        "quality_score": extracted_study.quality_assessment.quality_score,
                        "execution_time": execution_time
                    }
                )
            }
            
        except Exception as e:
            return self._handle_error(state, str(e))
    
    def create_vector_store(
        self,
        state: MetaAnalysisState,
        config: RunnableConfig = None,
        *,
        store: BaseStore
    ) -> Dict[str, Any]:
        """
        Cria vector store com todos os chunks processados.
        
        Args:
            state: Estado atual
            config: Configuração do LangGraph
            store: Store para persistência
            
        Returns:
            Atualizações do estado
        """
        try:
            logger.info("Criando vector store consolidado")
            
            # Buscar todos os chunks armazenados
            namespace = ("metanalysis", state["meta_analysis_id"], "chunks")
            all_chunks = []
            
            # Iterar sobre artigos processados para buscar chunks
            for article in state.get("processed_articles", []):
                article_namespace = ("metanalysis", state["meta_analysis_id"], "articles", article["id"])
                article_data = store.get(article_namespace, "data")
                
                if article_data and "chunks" in article_data.value:
                    all_chunks.extend(article_data.value["chunks"])
            
            if not all_chunks:
                logger.warning("Nenhum chunk encontrado para criar vector store")
                return {
                    "messages": state["messages"] + [
                        AIMessage("Erro: Nenhum chunk encontrado para vetorização")
                    ]
                }
            
            # Criar vector store usando FAISS local
            vector_store_id = self._create_faiss_vector_store(
                all_chunks, state["meta_analysis_id"]
            )
            
            # Armazenar metadados do vector store
            vector_metadata = {
                "vector_store_id": vector_store_id,
                "total_chunks": len(all_chunks),
                "created_at": datetime.now().isoformat(),
                "embedding_model": Config.EMBEDDING_MODEL
            }
            
            store.put(
                ("metanalysis", state["meta_analysis_id"], "vector_store"),
                "metadata",
                vector_metadata
            )
            
            logger.info(f"Vector store criado com {len(all_chunks)} chunks")
            
            return {
                "vector_store_id": vector_store_id,
                "vector_store_status": {
                    "created": True,
                    "total_chunks": len(all_chunks),
                    "created_at": datetime.now().isoformat()
                },
                "chunk_count": len(all_chunks),
                "messages": state["messages"] + [
                    AIMessage(f"Vector store criado com {len(all_chunks)} chunks")
                ],
                **add_agent_log(
                    state,
                    self.name,
                    "vector_store_created",
                    {"total_chunks": len(all_chunks)}
                )
            }
            
        except Exception as e:
            return self._handle_error(state, str(e))
    
    def _store_article_data(
        self,
        store: BaseStore,
        meta_analysis_id: str,
        url: str,
        content: Dict[str, Any],
        extracted_study,
        chunks: List
    ) -> str:
        """Armazena dados do artigo no store de longo prazo."""
        article_id = str(uuid.uuid4())
        
        # Namespace hierárquico para organização
        namespace = ("metanalysis", meta_analysis_id, "articles", article_id)
        
        # Dados completos do artigo
        article_data = {
            "url": url,
            "content": content,
            "extracted_study": extracted_study.dict(),
            "chunks": [chunk.dict() for chunk in chunks],
            "processed_at": datetime.now().isoformat(),
            "processor_version": "1.0"
        }
        
        # Armazenar no store
        store.put(namespace, "data", article_data)
        
        # Armazenar metadados para busca rápida
        metadata = {
            "title": extracted_study.characteristics.title,
            "authors": extracted_study.characteristics.authors,
            "year": extracted_study.characteristics.year,
            "quality_score": extracted_study.quality_assessment.quality_score,
            "confidence_score": extracted_study.confidence_score,
            "chunks_count": len(chunks)
        }
        
        store.put(namespace, "metadata", metadata)
        
        logger.info(f"Artigo armazenado com ID: {article_id}")
        return article_id
    
    def _create_faiss_vector_store(
        self, 
        chunks: List[Dict[str, Any]], 
        meta_analysis_id: str
    ) -> str:
        """Cria vector store FAISS local."""
        try:
            import faiss
            import numpy as np
            import pickle
            import os
            
            # Extrair embeddings
            embeddings = []
            chunk_metadata = []
            
            for chunk in chunks:
                if "embedding" in chunk and chunk["embedding"]:
                    embeddings.append(chunk["embedding"])
                    chunk_metadata.append({
                        "chunk_id": chunk["chunk_id"],
                        "study_id": chunk["study_id"],
                        "content": chunk["content"],
                        "section": chunk["section"],
                        "study_title": chunk["study_title"],
                        "study_authors": chunk["study_authors"],
                        "study_year": chunk["study_year"]
                    })
            
            if not embeddings:
                raise ValueError("Nenhum embedding encontrado")
            
            # Converter para numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Criar índice FAISS
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # Normalizar embeddings para cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Adicionar embeddings ao índice
            index.add(embeddings_array)
            
            # Salvar índice e metadados
            vector_store_id = f"faiss_{meta_analysis_id}"
            os.makedirs("data/vector_stores", exist_ok=True)
            
            # Salvar índice FAISS
            faiss.write_index(index, f"data/vector_stores/{vector_store_id}.index")
            
            # Salvar metadados dos chunks
            with open(f"data/vector_stores/{vector_store_id}_metadata.pkl", "wb") as f:
                pickle.dump(chunk_metadata, f)
            
            logger.info(f"Vector store FAISS criado: {vector_store_id}")
            return vector_store_id
            
        except Exception as e:
            logger.error(f"Erro ao criar vector store FAISS: {e}")
            raise
    
    def _handle_no_articles(self, state: MetaAnalysisState) -> Dict[str, Any]:
        """Lida com caso de não haver artigos para processar."""
        # Verificar se já temos vector store criado
        if not state.get("vector_store_id") and state.get("processed_articles"):
            # Criar vector store com artigos já processados
            logger.info("Criando vector store com artigos já processados")
            return {
                "current_phase": "vectorization",
                "messages": state["messages"] + [
                    AIMessage("Iniciando criação do vector store...")
                ]
            }
        
        return {
            "messages": state["messages"] + [
                AIMessage("Processamento de artigos concluído")
            ]
        }
    
    def _handle_extraction_failure(
        self, 
        state: MetaAnalysisState, 
        url: str
    ) -> Dict[str, Any]:
        """Lida com falha na extração de um artigo."""
        logger.warning(f"Falha ao processar artigo: {url}")
        
        # Adicionar à lista de URLs com falha
        failed_urls = state.get("failed_urls", [])
        failed_urls.append({
            "url": url,
            "error": "Falha na extração de dados",
            "timestamp": datetime.now().isoformat()
        })
        
        # Remover da fila de processamento
        processing_queue = state.get("processing_queue", [])
        remaining_queue = processing_queue[1:] if processing_queue else []
        
        return {
            "processing_queue": remaining_queue,
            "failed_urls": failed_urls,
            "messages": state["messages"] + [
                AIMessage(f"Falha ao processar artigo: {url}")
            ],
            **add_agent_log(
                state,
                self.name,
                "extraction_failed",
                {"url": url},
                "warning"
            )
        }
    
    def _handle_error(self, state: MetaAnalysisState, error_msg: str) -> Dict[str, Any]:
        """Lida com erros no processamento."""
        logger.error(f"Erro no agente processador: {error_msg}")
        
        return {
            "messages": state["messages"] + [
                AIMessage(f"Erro no processamento: {error_msg}")
            ],
            **add_agent_log(
                state,
                self.name,
                "processing_error",
                {"error": error_msg},
                "error"
            )
        }


def processor_agent(
    state: MetaAnalysisState,
    config: RunnableConfig = None,
    *,
    store: BaseStore
) -> Dict[str, Any]:
    """
    Nó do agente processador para uso no LangGraph.
    
    Args:
        state: Estado atual da meta-análise
        config: Configuração do LangGraph
        store: Store para persistência
        
    Returns:
        Atualizações do estado
    """
    processor = ProcessorAgent()
    
    # Verificar fase atual
    current_phase = state.get("current_phase")
    
    if current_phase == "extraction":
        # Processar artigos
        return processor.process_articles(state, config, store=store)
    
    elif current_phase == "vectorization":
        # Criar vector store
        return processor.create_vector_store(state, config, store=store)
    
    else:
        # Fase desconhecida, retornar erro
        return {
            "messages": state["messages"] + [
                AIMessage(f"Fase desconhecida para processador: {current_phase}")
            ]
        }