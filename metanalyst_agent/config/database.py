"""
Configuração de banco de dados PostgreSQL para persistência de estado.
"""

import asyncio
from typing import Optional
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain.embeddings import init_embeddings

from .settings import settings


def get_checkpointer() -> PostgresSaver:
    """
    Retorna instância configurada do PostgresSaver para persistência de estado.
    """
    return PostgresSaver.from_conn_string(
        conn_string=settings.postgres_url,
        pool_config={
            "max_size": 20,
            "min_size": 5,
            "max_idle": 300,
            "max_lifetime": 3600,
        }
    )


def get_store() -> PostgresStore:
    """
    Retorna instância configurada do PostgresStore para memória de longo prazo.
    Inclui busca semântica com embeddings.
    """
    return PostgresStore.from_conn_string(
        conn_string=settings.postgres_url,
        index={
            "embed": init_embeddings(f"openai:{settings.embedding_model}"),
            "dims": settings.vector_store_dimensions,
            "fields": ["content", "summary", "metadata", "extracted_data"]
        },
        pool_config={
            "max_size": 20,
            "min_size": 5,
            "max_idle": 300,
            "max_lifetime": 3600,
        }
    )


async def init_database() -> None:
    """
    Inicializa as tabelas necessárias no PostgreSQL.
    """
    # Criar tabelas para checkpointer
    checkpointer = get_checkpointer()
    await checkpointer.setup()
    
    # Criar tabelas para store
    store = get_store()
    await store.setup()
    
    print("✅ Banco de dados PostgreSQL inicializado com sucesso!")


def sync_init_database() -> None:
    """
    Versão síncrona da inicialização do banco de dados.
    """
    asyncio.run(init_database())