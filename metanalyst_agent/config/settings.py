"""
Configurações centralizadas do sistema usando Pydantic e variáveis de ambiente.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configurações centralizadas do Metanalyst-Agent"""
    
    # APIs Externas
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    
    # Banco de Dados PostgreSQL
    postgres_url: str = Field(..., env="POSTGRES_URL")
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("metanalysis", env="POSTGRES_DB")
    postgres_user: str = Field("metanalyst", env="POSTGRES_USER")
    postgres_password: str = Field("secure_password", env="POSTGRES_PASSWORD")
    
    # Configurações do Sistema
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    max_articles_per_search: int = Field(50, env="MAX_ARTICLES_PER_SEARCH")
    vector_store_dimensions: int = Field(1536, env="VECTOR_STORE_DIMENSIONS")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")
    
    # Configurações de Meta-análise
    min_studies_for_analysis: int = Field(3, env="MIN_STUDIES_FOR_ANALYSIS")
    confidence_level: float = Field(0.95, env="CONFIDENCE_LEVEL")
    heterogeneity_threshold: float = Field(0.5, env="HETEROGENEITY_THRESHOLD")
    
    # Diretórios de Trabalho
    temp_dir: str = Field("/tmp/metanalyst", env="TEMP_DIR")
    reports_dir: str = Field("./reports", env="REPORTS_DIR")
    plots_dir: str = Field("./plots", env="PLOTS_DIR")
    
    # Modelos LLM
    primary_model: str = Field("openai:gpt-4-turbo", env="PRIMARY_MODEL")
    secondary_model: str = Field("openai:gpt-4o-mini", env="SECONDARY_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Criar diretórios se não existirem
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)


# Instância global das configurações
settings = Settings()