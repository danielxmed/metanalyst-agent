"""
Configuração centralizada do sistema metanalyst-agent.
Gerencia variáveis de ambiente, configurações de banco e parâmetros do sistema.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Carregar variáveis de ambiente
load_dotenv()


class Config:
    """Configuração centralizada do sistema."""
    
    # === APIs ===
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # === BANCO DE DADOS ===
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://metanalyst:metanalyst123@localhost:5432/metanalyst"
    )
    
    # === MODELOS ===
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # === PARÂMETROS DE BUSCA ===
    MAX_PAPERS_PER_SEARCH: int = int(os.getenv("MAX_PAPERS_PER_SEARCH", "15"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # === PERFORMANCE ===
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
    
    # === LOGGING ===
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "metanalyst.log")
    
    # === DOMÍNIOS DE BUSCA ===
    SEARCH_DOMAINS = [
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
        "clinicaltrials.gov",
        "cochranelibrary.com",
        "embase.com",
        "sciencedirect.com",
        "springer.com",
        "nature.com",
        "bmj.com",
        "nejm.org",
        "thelancet.com",
        "jamanetwork.com"
    ]
    
    # === CRITÉRIOS DE QUALIDADE ===
    MIN_STUDY_QUALITY_SCORE = 6.0
    MIN_RELEVANCE_SCORE = 0.7
    MIN_PICO_MATCH_SCORE = 0.6
    
    @classmethod
    def validate_required_keys(cls) -> bool:
        """Valida se as chaves de API necessárias estão configuradas."""
        required_keys = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("TAVILY_API_KEY", cls.TAVILY_API_KEY)
        ]
        
        missing_keys = []
        for key_name, key_value in required_keys:
            if not key_value or key_value.startswith("sk-your-") or key_value.startswith("tvly-your-"):
                missing_keys.append(key_name)
        
        if missing_keys:
            logging.error(f"Chaves de API não configuradas: {', '.join(missing_keys)}")
            return False
        
        return True
    
    @classmethod
    def get_database_config(cls) -> Dict[str, str]:
        """Retorna configuração do banco de dados."""
        return {
            "database_url": cls.DATABASE_URL,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "pool_recycle": 3600
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Retorna configuração dos modelos de linguagem."""
        return {
            "model": cls.LLM_MODEL,
            "temperature": 0.1,
            "max_tokens": 4000,
            "timeout": cls.TIMEOUT_SECONDS
        }
    
    @classmethod
    def get_embedding_config(cls) -> Dict[str, Any]:
        """Retorna configuração dos embeddings."""
        return {
            "model": cls.EMBEDDING_MODEL,
            "dimensions": 1536,
            "batch_size": cls.BATCH_SIZE,
            "timeout": cls.TIMEOUT_SECONDS
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Retorna configuração de busca."""
        return {
            "max_papers": cls.MAX_PAPERS_PER_SEARCH,
            "domains": cls.SEARCH_DOMAINS,
            "timeout": cls.TIMEOUT_SECONDS,
            "retries": cls.MAX_RETRIES
        }
    
    @classmethod
    def get_processing_config(cls) -> Dict[str, Any]:
        """Retorna configuração de processamento."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "batch_size": cls.BATCH_SIZE,
            "timeout": cls.TIMEOUT_SECONDS
        }
    
    @classmethod
    def get_quality_thresholds(cls) -> Dict[str, float]:
        """Retorna thresholds de qualidade."""
        return {
            "min_study_quality": cls.MIN_STUDY_QUALITY_SCORE,
            "min_relevance": cls.MIN_RELEVANCE_SCORE,
            "min_pico_match": cls.MIN_PICO_MATCH_SCORE
        }


def setup_logging():
    """Configura o sistema de logging."""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )


def get_postgres_connection_string() -> str:
    """
    Retorna string de conexão PostgreSQL formatada.
    
    Returns:
        String de conexão para PostgreSQL
    """
    return Config.DATABASE_URL


def validate_environment() -> bool:
    """
    Valida se o ambiente está configurado corretamente.
    
    Returns:
        True se válido, False caso contrário
    """
    # Verificar chaves de API
    if not Config.validate_required_keys():
        return False
    
    # Verificar se consegue conectar ao banco (opcional para setup inicial)
    try:
        import psycopg2
        conn = psycopg2.connect(Config.DATABASE_URL)
        conn.close()
        logging.info("Conexão com PostgreSQL validada com sucesso")
    except Exception as e:
        logging.warning(f"Não foi possível conectar ao PostgreSQL: {e}")
        logging.warning("Certifique-se de que o PostgreSQL está rodando")
    
    return True


# Configurações específicas para diferentes ambientes
class DevelopmentConfig(Config):
    """Configurações para desenvolvimento."""
    LOG_LEVEL = "DEBUG"
    MAX_PAPERS_PER_SEARCH = 5  # Menos papers para testes rápidos


class ProductionConfig(Config):
    """Configurações para produção."""
    LOG_LEVEL = "WARNING"
    MAX_PAPERS_PER_SEARCH = 25  # Mais papers para análises completas


def get_config(environment: str = "development") -> Config:
    """
    Retorna configuração baseada no ambiente.
    
    Args:
        environment: Ambiente (development, production)
        
    Returns:
        Instância de configuração apropriada
    """
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig
    }
    
    return configs.get(environment, Config)