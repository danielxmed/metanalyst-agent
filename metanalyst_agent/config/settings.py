"""
Configuration settings for Metanalyst-Agent using Pydantic for validation
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Main configuration class for Metanalyst-Agent"""
    
    # API Configuration
    openai_api_key: str = Field("", env="OPENAI_API_KEY", description="OpenAI API key - required for operation")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    tavily_api_key: str = Field("", env="TAVILY_API_KEY", description="Tavily API key - required for operation")
    
    # Database Configuration
    database_url: str = Field(
        "postgresql://metanalyst:password@localhost:5432/metanalysis",
        env="DATABASE_URL"
    )
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    mongodb_url: str = Field("mongodb://localhost:27017/metanalysis", env="MONGODB_URL")
    
    # Vector Store Configuration
    faiss_index_path: Path = Field(Path("./data/vector_store/"), env="FAISS_INDEX_PATH")
    vector_dimension: int = Field(1536, env="VECTOR_DIMENSION")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")
    
    # Meta-Analysis Configuration
    default_max_articles: int = Field(50, env="DEFAULT_MAX_ARTICLES")
    default_quality_threshold: float = Field(0.8, env="DEFAULT_QUALITY_THRESHOLD")
    default_max_iterations: int = Field(5, env="DEFAULT_MAX_ITERATIONS")
    default_recursion_limit: int = Field(100, env="DEFAULT_RECURSION_LIMIT")
    
    # Agent Configuration
    agent_limits: Dict[str, int] = {
        "researcher": 5,
        "processor": 10,
        "retriever": 3,
        "analyst": 7,
        "writer": 3,
        "reviewer": 3,
        "editor": 2
    }
    
    quality_thresholds: Dict[str, float] = {
        "researcher": 0.7,
        "processor": 0.8,
        "retriever": 0.75,
        "analyst": 0.85,
        "writer": 0.8,
        "reviewer": 0.9,
        "editor": 0.9
    }
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Path = Field(Path("./logs/metanalyst.log"), env="LOG_FILE")
    
    # Development Configuration
    debug: bool = Field(False, env="DEBUG")
    development_mode: bool = Field(False, env="DEVELOPMENT_MODE")
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory from log_file path"""
        return self.log_file.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory"""
        return Path("./data")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @field_validator("faiss_index_path", "log_file", mode="before")
    def convert_to_path(cls, v):
        """Convert string paths to Path objects"""
        if isinstance(v, str):
            return Path(v)
        return v
        
    @field_validator("default_quality_threshold")
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        return v
        
    @field_validator("vector_dimension")
    def validate_vector_dimension(cls, v):
        """Validate vector dimension is positive"""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.faiss_index_path,
            self.log_file.parent,
            Path("./data/checkpoints/"),
            Path("./data/temp/")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration dictionary"""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "temperature": 0.1
        }
        
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get OpenAI embeddings configuration dictionary"""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_embedding_model
        }
        
    def get_tavily_config(self) -> Dict[str, Any]:
        """Get Tavily configuration dictionary"""
        return {
            "api_key": self.tavily_api_key
        }
        
    def get_database_configs(self) -> Dict[str, str]:
        """Get all database configuration URLs"""
        return {
            "postgresql": self.database_url,
            "redis": self.redis_url,
            "mongodb": self.mongodb_url
        }


# Global settings instance (lazy instantiation)
_settings = None

def get_settings() -> Settings:
    """Get or create the global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# For backward compatibility
settings = get_settings()
