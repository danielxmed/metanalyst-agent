"""
Data models for metanalyst-agent database tables.

Provides Pydantic models that correspond to PostgreSQL tables
for type safety and data validation.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of article processing."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


class MetaAnalysisStatus(str, Enum):
    """Status of meta-analysis."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class AgentStatus(str, Enum):
    """Status of agent execution."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class PICO(BaseModel):
    """PICO framework for research questions."""
    P: str = Field(..., description="Population/Patient")
    I: str = Field(..., description="Intervention")
    C: str = Field(..., description="Comparison/Control")
    O: str = Field(..., description="Outcome")
    
    @validator('P', 'I', 'C', 'O')
    def validate_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("PICO components cannot be empty")
        return v.strip()


class MetaAnalysis(BaseModel):
    """Meta-analysis model."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    thread_id: str = Field(..., description="LangGraph thread ID")
    title: str = Field(..., min_length=1, max_length=500)
    pico: PICO = Field(..., description="PICO framework")
    status: MetaAnalysisStatus = Field(default=MetaAnalysisStatus.ACTIVE)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_articles: int = Field(default=0, ge=0)
    processed_articles: int = Field(default=0, ge=0)
    failed_articles: int = Field(default=0, ge=0)
    final_report: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('processed_articles', 'failed_articles')
    def validate_article_counts(cls, v, values):
        total = values.get('total_articles', 0)
        if v > total:
            raise ValueError("Processed/failed articles cannot exceed total articles")
        return v
    
    @property
    def processing_percentage(self) -> float:
        """Calculate processing percentage."""
        if self.total_articles == 0:
            return 0.0
        return (self.processed_articles / self.total_articles) * 100
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class Article(BaseModel):
    """Article model."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    meta_analysis_id: uuid.UUID = Field(..., description="Parent meta-analysis ID")
    url: str = Field(..., description="Article URL")
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    publication_year: Optional[int] = Field(None, ge=1900, le=2030)
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    full_content: Optional[str] = None
    vancouver_citation: Optional[str] = None
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    failure_reason: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @validator('doi')
    def validate_doi(cls, v):
        if v and not v.startswith('10.'):
            raise ValueError("DOI must start with '10.'")
        return v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class ArticleChunk(BaseModel):
    """Article chunk model for vector storage."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    article_id: uuid.UUID = Field(..., description="Parent article ID")
    chunk_index: int = Field(..., ge=0, description="Chunk order index")
    content: str = Field(..., min_length=1, description="Chunk content")
    embedding_vector: List[float] = Field(..., description="Embedding vector")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('embedding_vector')
    def validate_embedding_dimension(cls, v):
        if len(v) != 1536:  # OpenAI text-embedding-3-small dimension
            raise ValueError("Embedding vector must have 1536 dimensions")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class StatisticalAnalysis(BaseModel):
    """Statistical analysis model."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    meta_analysis_id: uuid.UUID = Field(..., description="Parent meta-analysis ID")
    analysis_type: str = Field(..., description="Type of analysis")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    plots: Dict[str, str] = Field(default_factory=dict, description="Plot file paths")
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed_types = {
            'meta_analysis', 'forest_plot', 'funnel_plot', 
            'heterogeneity_analysis', 'sensitivity_analysis',
            'publication_bias', 'subgroup_analysis'
        }
        if v not in allowed_types:
            raise ValueError(f"Analysis type must be one of: {allowed_types}")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class AgentLog(BaseModel):
    """Agent execution log model."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    meta_analysis_id: uuid.UUID = Field(..., description="Parent meta-analysis ID")
    agent_name: str = Field(..., description="Name of the agent")
    action: str = Field(..., description="Action performed")
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = Field(None, ge=0)
    status: AgentStatus = Field(..., description="Execution status")
    error_message: Optional[str] = None
    iteration_count: int = Field(default=1, ge=1)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('agent_name')
    def validate_agent_name(cls, v):
        allowed_agents = {
            'orchestrator', 'researcher', 'processor', 'retriever',
            'analyst', 'writer', 'reviewer', 'editor'
        }
        if v not in allowed_agents:
            raise ValueError(f"Agent name must be one of: {allowed_agents}")
        return v
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class EmbeddingCache(BaseModel):
    """Embedding cache model for optimization."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    content_hash: str = Field(..., description="SHA256 hash of content")
    content_preview: str = Field(..., max_length=200, description="First 200 chars")
    embedding_vector: List[float] = Field(..., description="Cached embedding")
    model_name: str = Field(default="text-embedding-3-small")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('content_hash')
    def validate_hash_format(cls, v):
        if len(v) != 64:  # SHA256 length
            raise ValueError("Content hash must be 64 characters (SHA256)")
        return v
    
    @validator('embedding_vector')
    def validate_embedding_dimension(cls, v):
        if len(v) != 1536:
            raise ValueError("Embedding vector must have 1536 dimensions")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


# Database operations helpers
class DatabaseOperations:
    """Helper class for common database operations."""
    
    @staticmethod
    def to_db_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert Pydantic model to database-compatible dictionary."""
        data = model.dict(exclude_none=exclude_none)
        
        # Convert UUID objects to strings
        for key, value in data.items():
            if isinstance(value, uuid.UUID):
                data[key] = str(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, dict) and key in ['pico', 'extracted_data', 'metadata', 'results', 'quality_metrics', 'plots', 'chunk_metadata', 'input_data', 'output_data']:
                # Keep as dict for JSONB columns
                pass
        
        return data
    
    @staticmethod
    def from_db_row(model_class: type, row: Dict[str, Any]) -> BaseModel:
        """Create Pydantic model from database row."""
        # Convert string UUIDs back to UUID objects if needed
        data = dict(row)
        
        # Handle UUID fields
        uuid_fields = ['id', 'meta_analysis_id', 'article_id']
        for field in uuid_fields:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = uuid.UUID(data[field])
        
        # Handle datetime fields
        datetime_fields = ['created_at', 'updated_at']
        for field in datetime_fields:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
        
        return model_class(**data)


# Export all models
__all__ = [
    'ProcessingStatus',
    'MetaAnalysisStatus', 
    'AgentStatus',
    'PICO',
    'MetaAnalysis',
    'Article',
    'ArticleChunk',
    'StatisticalAnalysis',
    'AgentLog',
    'EmbeddingCache',
    'DatabaseOperations'
]