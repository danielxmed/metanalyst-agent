# Metanalyst-Agent: Technical Documentation

## Overview

The Metanalyst-Agent is an AI-first multi-agent system for automated meta-analysis generation. It follows a hub-and-spoke architecture where specialized agents collaborate to perform comprehensive systematic reviews and meta-analyses from literature search to final report generation.

## Architecture

### Hub-and-Spoke Design

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── RETRIEVER
                │   ORCHESTRATOR  │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── WRITER
                         │
                         │
                     (PROCESSOR combines
                      extraction + vectorization)

    ● = Central Orchestrator (Hub)
    │ = Direct Connections (Agents-as-a-Tool)
```

### Key Principles

1. **AI-First Approach**: LLMs handle complex tasks (extraction, analysis, writing)
2. **Autonomous Agents**: Each agent makes independent decisions using ReAct pattern
3. **Centralized Orchestration**: Hub coordinates all specialized agents
4. **Persistent Memory**: Long-term and short-term memory using LangGraph stores
5. **Quality Control**: Iterative refinement with quality thresholds

## System Components

### 1. Orchestrator Agent (Hub)

**Role**: Central conductor that coordinates the entire meta-analysis process

**Responsibilities**:
- Analyze current state and decide next agent to invoke
- Generate PICO framework from user queries
- Monitor quality scores and iteration counts
- Handle errors and edge cases
- Manage workflow phases

**Tools**:
- All handoff tools for agent coordination
- Emergency stop and human intervention requests

**Decision Logic**:
```python
def orchestrator_decision_logic(state):
    # Phase-based routing
    if phase == "pico_definition":
        return "transfer_to_researcher"
    elif phase == "search" and articles_found >= 10:
        return "transfer_to_processor"
    elif phase == "extraction" and processing_rate >= 0.7:
        return "transfer_to_analyst"
    # ... more decision logic
```

### 2. Research Agent

**Role**: Scientific literature search and relevance assessment

**Responsibilities**:
- Generate optimized search queries from PICO
- Search multiple databases (PubMed, Cochrane, Clinical Trials)
- Assess article relevance using AI
- Filter results based on inclusion/exclusion criteria

**Tools**:
- `search_scientific_literature`: Tavily-powered database search
- `generate_search_queries_with_llm`: AI-generated query optimization
- `assess_article_relevance`: LLM-based relevance scoring
- `filter_articles_by_relevance`: Batch filtering with criteria

**Quality Standards**:
- Target: 10-15 relevant articles minimum
- Relevance threshold: >70%
- Prioritize RCTs and systematic reviews

### 3. Processor Agent

**Role**: Article extraction, data processing, and vectorization

**Responsibilities**:
- Extract full content using Tavily Extract API
- Process content to extract statistical data with LLMs
- Generate Vancouver citations
- Create text chunks and vector embeddings
- Build searchable vector stores

**Tools**:
- `extract_article_content_with_tavily`: Full content extraction
- `extract_statistical_data_with_llm`: Statistical data extraction
- `generate_vancouver_citation_with_llm`: Citation formatting
- `chunk_and_vectorize_content`: Text processing for search
- `create_vector_store`: FAISS index creation

**Processing Pipeline**:
1. Extract content from URLs
2. Extract statistical data relevant to PICO
3. Generate proper citations
4. Create optimized text chunks
5. Generate vector embeddings
6. Build searchable index

### 4. Retriever Agent

**Role**: Semantic search and information retrieval

**Responsibilities**:
- Search vector store for relevant information
- Retrieve chunks by PICO elements
- Find statistical data across studies
- Support evidence gathering for analysis

**Tools**:
- `search_vector_store`: Semantic similarity search
- `retrieve_relevant_chunks`: PICO-based retrieval
- `search_for_statistical_data`: Statistical information search
- `find_study_methodologies`: Methodology information search

### 5. Analyst Agent

**Role**: Statistical meta-analysis and visualization

**Responsibilities**:
- Perform comprehensive meta-analysis calculations
- Create forest plots and visualizations
- Assess study quality using established frameworks
- Calculate heterogeneity statistics

**Tools**:
- `perform_meta_analysis_with_llm`: AI-assisted statistical analysis
- `create_forest_plot`: Interactive Plotly visualizations
- `assess_study_quality_with_llm`: Quality assessment with frameworks
- `calculate_heterogeneity`: I², tau², Q-statistics

**Statistical Methods**:
- Random-effects and fixed-effects models
- Heterogeneity assessment (I², Tau², Q-statistic)
- Publication bias evaluation
- Subgroup analysis recommendations

### 6. Writer Agent

**Role**: Report generation and documentation

**Responsibilities**:
- Generate structured reports following PRISMA guidelines
- Create section-specific content (abstract, methods, results, discussion)
- Format citations properly
- Integrate statistical results and visualizations

**Tools**:
- `generate_report_section_with_llm`: Section-specific content generation
- `format_citations_with_llm`: Citation style formatting
- `create_complete_report`: Full report integration

**Report Structure**:
- Abstract (structured, PRISMA-compliant)
- Introduction (background, objectives, PICO)
- Methods (search strategy, criteria, analysis)
- Results (study selection, characteristics, analysis)
- Discussion (interpretation, limitations, implications)
- Conclusion (key findings, recommendations)

### 7. Reviewer Agent

**Role**: Quality assessment and feedback

**Responsibilities**:
- Review report quality and completeness
- Assess compliance with medical standards
- Generate improvement suggestions
- Validate statistical analyses

### 8. Editor Agent

**Role**: Final integration and formatting

**Responsibilities**:
- Integrate all components into final report
- Apply consistent formatting and styling
- Generate publication-ready HTML output
- Handle final quality checks

## State Management

### Shared State Structure

```python
class MetaAnalysisState(TypedDict):
    # Identification
    meta_analysis_id: str
    thread_id: str
    current_phase: Literal["pico_definition", "search", "extraction", ...]
    
    # PICO Framework
    pico: Dict[str, str]  # {P, I, C, O}
    research_question: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    
    # Processing Pipeline
    candidate_urls: List[Dict[str, Any]]
    processed_articles: List[Dict[str, Any]]
    extracted_data: List[Dict[str, Any]]
    
    # Analysis Results
    statistical_analysis: Dict[str, Any]
    forest_plots: List[Dict[str, Any]]
    quality_assessments: Dict[str, float]
    
    # Quality Control
    quality_scores: Dict[str, float]
    agent_iterations: Dict[str, int]
    improvement_rates: Dict[str, float]
    
    # Communication
    messages: List[BaseMessage]
    agent_logs: List[Dict[str, Any]]
```

### Memory Management

**Short-term Memory (Checkpointers)**:
- PostgreSQL/Redis/Memory backends
- Thread-level persistence
- Conversation state management
- Recovery from failures

**Long-term Memory (Stores)**:
- Cross-thread information sharing
- Article metadata storage
- Analysis history
- Semantic search capabilities

## Quality Control

### Iteration Control

**Global Limits**:
- Maximum global iterations: 10
- Per-agent iteration limits: 3-7 (varies by agent)
- Quality improvement thresholds: 5% minimum

**Quality Thresholds**:
- Researcher: 70% relevance rate
- Processor: 80% successful extraction
- Analyst: 85% analysis quality
- Writer: 85% report quality
- Reviewer: 90% review satisfaction

**Circuit Breaker Pattern**:
- Failure threshold: 5 consecutive failures
- Recovery timeout: 60 seconds
- Half-open state for gradual recovery

### Error Handling

**Retry Strategies**:
- Exponential backoff for network issues
- Maximum 3 retries per operation
- Graceful degradation for partial failures

**Emergency Conditions**:
- Force stop for critical failures
- Human intervention requests for complex issues
- Timeout handling with partial result recovery

## Technical Implementation

### Dependencies

**Core Framework**:
- LangGraph: Multi-agent orchestration
- LangChain: LLM integration and tooling
- Pydantic: Configuration and data validation

**AI Services**:
- OpenAI GPT-4: Primary reasoning and analysis
- OpenAI text-embedding-3-small: Vector embeddings
- Tavily API: Web search and content extraction

**Data Processing**:
- FAISS: Vector similarity search
- NumPy/SciPy: Statistical calculations
- Matplotlib/Plotly: Visualization generation
- Pandas: Data manipulation

**Storage**:
- PostgreSQL: Persistent checkpoints and stores
- Redis: High-performance caching
- In-memory: Development and testing

### Configuration

**Environment Variables**:
```bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
DATABASE_URL=postgresql://user:pass@localhost/db
```

**Settings Management**:
```python
from metanalyst_agent.config import settings

# Access configuration
model = settings.openai_model
threshold = settings.default_quality_threshold
```

### Usage Patterns

**Basic Usage**:
```python
from metanalyst_agent import run_meta_analysis

results = run_meta_analysis(
    query="Meta-analysis of mindfulness vs CBT for anxiety",
    max_articles=30,
    quality_threshold=0.8
)
```

**Advanced Usage**:
```python
from metanalyst_agent import MetanalystAgent

with MetanalystAgent(use_persistent_storage=True) as agent:
    results = agent.run(
        query="Research question here",
        max_articles=50,
        quality_threshold=0.85,
        max_time_minutes=45
    )
```

**Streaming Updates**:
```python
async for progress in agent.stream(query):
    print(f"Phase: {progress['phase']}")
    print(f"Articles: {progress['articles_processed']}")
```

## Performance Considerations

### Scalability

**Concurrent Processing**:
- Parallel article processing
- Batch operations where possible
- Async/await for I/O operations

**Resource Management**:
- Connection pooling for databases
- Caching for repeated operations
- Memory-efficient text processing

**Optimization Strategies**:
- Vector store indexing for fast retrieval
- Chunking strategies for large documents
- Quality-based early termination

### Monitoring

**Metrics Tracking**:
- Processing success rates
- Quality scores by component
- Execution time per phase
- Resource utilization

**Logging**:
- Structured logging with timestamps
- Agent-specific log separation
- Error tracking and alerting

## Extension Points

### Custom Agents

Add new specialized agents by:
1. Creating agent class with ReAct pattern
2. Defining specialized tools
3. Adding to orchestrator routing logic
4. Updating state schema if needed

### Custom Tools

Extend functionality by:
1. Implementing `@tool` decorated functions
2. Adding to appropriate agent tool lists
3. Following AI-first principles
4. Including proper error handling

### Custom Analysis Methods

Add new statistical methods by:
1. Creating analysis tools
2. Updating analyst agent capabilities
3. Extending result formatting
4. Adding visualization support

## Security Considerations

**API Key Management**:
- Environment variable storage
- No hardcoded credentials
- Secure configuration validation

**Data Privacy**:
- No persistent storage of article content without consent
- Configurable data retention policies
- Secure database connections

**Input Validation**:
- Pydantic schema validation
- SQL injection prevention
- XSS protection in HTML output

## Troubleshooting

### Common Issues

**API Rate Limits**:
- Implement exponential backoff
- Use rate limiting middleware
- Monitor usage patterns

**Memory Issues**:
- Optimize chunk sizes
- Clear temporary data
- Use streaming for large operations

**Quality Issues**:
- Adjust quality thresholds
- Review search strategies
- Validate input criteria

### Debug Mode

Enable detailed logging:
```python
agent = MetanalystAgent(debug=True)
```

Access agent logs:
```python
results = agent.run(query)
logs = results.get("agent_logs", [])
```

## Contributing

### Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Run tests: `pytest`

### Code Style

- Follow PEP 8
- Use type hints
- Document all functions
- Write comprehensive tests

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit PR with clear description

## Future Roadmap

### Planned Features

- Additional statistical methods
- More visualization types
- Custom report templates
- API endpoint exposure
- Web interface development

### Research Directions

- Improved quality assessment
- Automated PICO extraction
- Multi-language support
- Real-time collaboration
- Integration with reference managers