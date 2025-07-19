"""
Configuração do sistema MetAnalyst Agent.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class LLMConfig(BaseModel):
    """Configuração dos modelos LLM"""
    provider: str = Field(default="openai", description="Provedor LLM (openai ou anthropic)")
    openai_model: str = Field(default="gpt-4-1106-preview", description="Modelo OpenAI")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", description="Modelo Anthropic")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Temperatura do modelo")
    max_tokens: int = Field(default=4000, gt=0, description="Máximo de tokens")

class SearchConfig(BaseModel):
    """Configuração de busca"""
    max_results: int = Field(default=50, gt=0, description="Máximo de resultados por busca")
    min_relevance_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Score mínimo de relevância")
    domains: list[str] = Field(
        default=[
            "nejm.org",
            "jamanetwork.com", 
            "thelancet.com",
            "bmj.com",
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov/pmc",
            "scielo.org",
            "cochranelibrary.com"
        ],
        description="Domínios para busca médica"
    )

class VectorConfig(BaseModel):
    """Configuração do vector store"""
    embedding_model: str = Field(default="text-embedding-3-small", description="Modelo de embeddings")
    chunk_size: int = Field(default=1000, gt=0, description="Tamanho dos chunks")
    chunk_overlap: int = Field(default=100, ge=0, description="Sobreposição dos chunks")
    store_path: str = Field(default="./vector_stores/", description="Caminho para armazenar vector stores")

class AnalysisConfig(BaseModel):
    """Configuração de análise estatística"""
    min_studies: int = Field(default=3, gt=0, description="Mínimo de estudos para análise")
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0, description="Nível de confiança")
    heterogeneity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold para heterogeneidade")

class DatabaseConfig(BaseModel):
    """Configuração do banco de dados"""
    url: str = Field(description="URL de conexão PostgreSQL")
    pool_size: int = Field(default=10, gt=0, description="Tamanho do pool de conexões")
    max_overflow: int = Field(default=20, ge=0, description="Máximo overflow do pool")

class SystemConfig(BaseModel):
    """Configuração completa do sistema"""
    
    # APIs obrigatórias
    openai_api_key: Optional[str] = Field(default=None, description="Chave API OpenAI")
    anthropic_api_key: Optional[str] = Field(default=None, description="Chave API Anthropic")
    tavily_api_key: str = Field(description="Chave API Tavily (obrigatória)")
    
    # Configurações dos componentes
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    database: DatabaseConfig = Field(description="Configuração do banco")
    
    # Configurações gerais
    debug_mode: bool = Field(default=False, description="Modo debug")
    log_level: str = Field(default="INFO", description="Nível de log")
    max_retries: int = Field(default=3, gt=0, description="Máximo de tentativas")
    timeout_seconds: int = Field(default=30, gt=0, description="Timeout em segundos")
    
    def model_post_init(self, __context: Any) -> None:
        """Validações pós-inicialização"""
        # Verificar se pelo menos uma API de LLM está configurada
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("Pelo menos uma chave de API (OpenAI ou Anthropic) deve ser fornecida")
        
        # Ajustar provedor baseado nas chaves disponíveis
        if self.llm.provider == "openai" and not self.openai_api_key:
            if self.anthropic_api_key:
                self.llm.provider = "anthropic"
            else:
                raise ValueError("Chave OpenAI necessária para provedor 'openai'")
        
        if self.llm.provider == "anthropic" and not self.anthropic_api_key:
            if self.openai_api_key:
                self.llm.provider = "openai"
            else:
                raise ValueError("Chave Anthropic necessária para provedor 'anthropic'")

def load_config() -> SystemConfig:
    """Carregar configuração a partir das variáveis de ambiente"""
    
    # Configuração do banco
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL é obrigatória")
    
    database_config = DatabaseConfig(url=database_url)
    
    # Chaves de API
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") 
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY é obrigatória")
    
    # Configuração LLM
    llm_config = LLMConfig(
        provider=os.getenv("PREFERRED_LLM_PROVIDER", "openai"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4-1106-preview"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    )
    
    # Configuração de busca
    search_config = SearchConfig(
        max_results=int(os.getenv("MAX_SEARCH_RESULTS", "50")),
        min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.7"))
    )
    
    # Configuração de vector
    vector_config = VectorConfig(
        embedding_model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"),
        chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        store_path=os.getenv("VECTOR_STORE_PATH", "./vector_stores/")
    )
    
    # Configuração de análise
    analysis_config = AnalysisConfig(
        min_studies=int(os.getenv("MIN_STUDIES_FOR_ANALYSIS", "3")),
        confidence_level=float(os.getenv("CONFIDENCE_LEVEL", "0.95"))
    )
    
    return SystemConfig(
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        database=database_config,
        llm=llm_config,
        search=search_config,
        vector=vector_config,
        analysis=analysis_config,
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

# Instância global da configuração
config = load_config()