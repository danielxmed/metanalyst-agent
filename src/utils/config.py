"""
Configuration management for the Metanalyst Agent system.

This module handles environment variables, API keys, and system settings
required for the multi-agent meta-analysis workflow.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import dotenv

dotenv.load_dotenv()

@dataclass
class APIConfig:
    """Configuration for external API services."""
    tavily_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment variables."""
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class SearchConfig:
    """Configuration for literature search."""
    max_papers_per_search: int = 15
    search_depth: str = "basic"  # "basic" or "advanced"
    include_domains: list = field(default_factory=lambda: [
        "pubmed.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov/pmc", 
        "www.cochranelibrary.com",
        "lilacs.bvsalud.org",
        "scielo.org",
        "www.embase.com",
        "www.webofscience.com",
        "www.scopus.com",
        "www.epistemonikos.org",
        "www.ebscohost.com",
        "www.tripdatabase.com",
        "pedro.org.au",
        "doaj.org",
        "scholar.google.com",
        "clinicaltrials.gov",
        "apps.who.int/trialsearch",
        "www.clinicaltrialsregister.eu",
        "www.isrctn.com",
        "www.thelancet.com",
        "www.nejm.org",
        "jamanetwork.com",
        "www.bmj.com",
        "www.nature.com/nm",
        "www.acpjournals.org/journal/aim",
        "journals.plos.org/plosmedicine",
        "www.jclinepi.com",
        "systematicreviewsjournal.biomedcentral.com",
        "ascopubs.org/journal/jco",
        "www.ahajournals.org/journal/circ",
        "www.gastrojournal.org",
        "academic.oup.com/eurheartj",
        "www.archives-pmr.org",
        "www.jacc.org",
        "www.scielo.br",
        "nejm.org",
        "thelancet.com",
        "bmj.com",
        "cacancerjournal.com",
        "nature.com/nm",
        "cell.com/cell-metabolism/home",
        "thelancet.com/journals/langlo/home",
        "cochranelibrary.com",
        "memorias.ioc.fiocruz.br",
        "scielo.br/j/csp/",
        "cadernos.ensp.fiocruz.br",
        "scielo.br/j/rsp/",
        "scielo.org/journal/rpsp/",
        "journal.paho.org",
        "rbmt.org.br",
        "revistas.usp.br/rmrp",
        "ncbi.nlm.nih.gov/pmc",
        "scopus.com",
        "webofscience.com",
        "bvsalud.org",
        "jbi.global",
        "tripdatabase.com",
        "gov.br",
        "droracle.ai",
        "wolterskluwer.com",
        "semanticscholar.org",
        "globalindexmedicus.net",
        "sciencedirect.com",
        "openevidence.com"
    ])
    topic: str = "general"  # "general" or "news"


@dataclass
class VectorConfig:
    """Configuration for vector embeddings and storage."""
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    vector_store_path: str = "data/vector_store"
    similarity_threshold: float = 0.7
    max_chunks_per_query: int = 100000


@dataclass
class LLMConfig:
    """Configuration for language models."""
    primary_model: str = "claude-sonnet-4-20250514"
    fallback_model: str = "gpt-4.1"
    fast_processing_model: str = "gpt-4.1-mini"  # Fast model for content processing
    temperature: float = 0.8
    max_tokens: int = 16000
    timeout: int = 60


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    output_dir: str = "outputs"
    report_template_dir: str = "templates/html"
    css_template_dir: str = "templates/css"
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"
    export_formats: list = field(default_factory=lambda: ["html", "pdf"])


@dataclass
class QualityConfig:
    """Configuration for quality control."""
    min_papers_for_meta_analysis: int = 3
    max_papers_per_meta_analysis: int = 50
    quality_score_threshold: float = 0.6
    reviewer_approval_threshold: float = 7.0  # 0-10 scale
    enable_human_review: bool = False


@dataclass
class SystemConfig:
    """Main system configuration combining all sub-configurations."""
    api: APIConfig = field(default_factory=APIConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # System-level settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Initialize configuration and create necessary directories."""
        self._create_directories()
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output.output_dir,
            self.output.figures_dir,
            self.output.tables_dir,
            self.vector.vector_store_path,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration settings."""
        if not self.api.tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        
        if not (self.api.openai_api_key or self.api.anthropic_api_key):
            raise ValueError("At least one LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) is required")
        
        if self.search.max_papers_per_search < 1:
            raise ValueError("max_papers_per_search must be at least 1")
        
        if self.vector.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        
        if not 0 <= self.vector.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "api": {
                "tavily_api_key": "***" if self.api.tavily_api_key else None,
                "openai_api_key": "***" if self.api.openai_api_key else None,
                "anthropic_api_key": "***" if self.api.anthropic_api_key else None,
            },
            "search": {
                "max_papers_per_search": self.search.max_papers_per_search,
                "search_depth": self.search.search_depth,
                "domains_count": len(self.search.include_domains),
                "topic": self.search.topic
            },
            "vector": {
                "embedding_model": self.vector.embedding_model,
                "chunk_size": self.vector.chunk_size,
                "chunk_overlap": self.vector.chunk_overlap,
                "vector_store_path": self.vector.vector_store_path,
                "similarity_threshold": self.vector.similarity_threshold
            },
            "llm": {
                "primary_model": self.llm.primary_model,
                "fallback_model": self.llm.fallback_model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens
            },
            "output": {
                "output_dir": self.output.output_dir,
                "export_formats": self.output.export_formats
            },
            "quality": {
                "min_papers_for_meta_analysis": self.quality.min_papers_for_meta_analysis,
                "quality_score_threshold": self.quality.quality_score_threshold,
                "reviewer_approval_threshold": self.quality.reviewer_approval_threshold
            },
            "system": {
                "debug_mode": self.debug_mode,
                "log_level": self.log_level,
                "max_retries": self.max_retries
            }
        }


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = SystemConfig()
    return _config


def reload_config() -> SystemConfig:
    """Reload configuration from environment variables."""
    global _config
    _config = SystemConfig()
    return _config


def update_config(**kwargs) -> SystemConfig:
    """Update specific configuration values."""
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to update nested configs
            for nested_config in [config.api, config.search, config.vector, 
                                config.llm, config.output, config.quality]:
                if hasattr(nested_config, key):
                    setattr(nested_config, key, value)
                    break
    
    return config


# Environment-specific configurations
def get_development_config() -> SystemConfig:
    """Get configuration optimized for development."""
    config = SystemConfig()
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.search.max_papers_per_search = 5  # Smaller for testing
    config.llm.temperature = 0.2  # More deterministic for testing
    return config


def get_production_config() -> SystemConfig:
    """Get configuration optimized for production."""
    config = SystemConfig()
    config.debug_mode = False
    config.log_level = "INFO"
    config.search.max_papers_per_search = 15
    config.llm.temperature = 0.1
    config.quality.enable_human_review = True
    return config


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    try:
        get_config()
        return True
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"📊 Config summary: {len(config.search.include_domains)} domains, "
              f"{config.search.max_papers_per_search} max papers")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
