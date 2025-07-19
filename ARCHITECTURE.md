# Metanalyst-Agent Architecture

## Overview

Metanalyst-Agent implements a **Hub-and-Spoke Multi-Agent Architecture** using LangGraph, where specialized agents collaborate to perform automated meta-analysis from research questions to final reports with statistical analysis.

## Architecture Pattern: Hub-and-Spoke with Agents-as-Tools

```
                    RESEARCHER
                         │
                         │
            EDITOR ──────┼────── PROCESSOR
                │        │        │
                │        │        │
    ANALYST ───┼────────●────────┼─── RETRIEVER
                │   SUPERVISOR   │
                │        │        │
                │        │        │
           REVIEWER ──────┼────── WRITER
                         │
                         │

    ● = Central Supervisor (Hub)
    │ = Direct Connections (Agents-as-Tools)
```

### Key Principles

1. **Central Orchestration**: Supervisor agent maintains global state and makes all routing decisions
2. **Specialized Agents**: Each agent is an expert in a specific domain (research, processing, analysis, etc.)
3. **Autonomous Decision Making**: Agents use ReAct pattern to decide which tools to use
4. **Shared State**: All agents operate on the same comprehensive state structure
5. **AI-First Approach**: LLMs handle complex tasks instead of manual heuristics

## System Components

### 1. Core Agents

#### Supervisor Agent (`supervisor_agent.py`)
- **Role**: Central orchestrator and decision maker
- **Responsibilities**:
  - Analyze research questions and define PICO framework
  - Route tasks to appropriate specialized agents
  - Monitor progress and ensure quality standards
  - Handle error recovery and intervention requests
- **Tools**: Handoff tools to all other agents
- **Decision Logic**: Analyzes current state to determine next agent

#### Researcher Agent (`researcher_agent.py`)
- **Role**: Scientific literature search specialist
- **Responsibilities**:
  - Generate PICO-based search queries
  - Search medical databases (PubMed, Cochrane, etc.)
  - Assess article relevance against inclusion/exclusion criteria
  - Ensure comprehensive literature coverage
- **Tools**: `search_literature`, `generate_search_queries`, `assess_article_relevance`
- **Quality Thresholds**: ≥10 relevant articles minimum, ≥20 target

#### Processor Agent (`processor_agent.py`)
- **Role**: Content extraction and data processing specialist
- **Responsibilities**:
  - Extract full article content using Tavily Extract API
  - Extract statistical data relevant to PICO using LLM analysis
  - Generate Vancouver-style citations
  - Create vector embeddings for semantic search
- **Tools**: `extract_article_content`, `extract_statistical_data`, `chunk_and_vectorize`
- **Quality Standards**: >80% successful extraction target, >60% minimum

#### Analyst Agent (`analyst_agent.py`)
- **Role**: Statistical analysis and meta-analysis specialist
- **Responsibilities**:
  - Perform statistical meta-analysis calculations
  - Create forest plots and visualizations
  - Assess heterogeneity between studies
  - Conduct sensitivity analyses
- **Tools**: `calculate_meta_analysis`, `create_forest_plot`, `assess_heterogeneity`
- **Requirements**: ≥3 studies minimum, ≥5 studies for robust analysis

### 2. State Management

#### Comprehensive State Structure (`meta_analysis_state.py`)
```python
class IterationState(MetaAnalysisState):
    # Core identification and control
    meta_analysis_id: str
    current_phase: Literal[...]
    current_agent: Optional[str]
    
    # Research framework (PICO)
    pico: Dict[str, str]
    research_question: str
    
    # Literature and processing
    candidate_urls: List[Dict[str, Any]]
    processed_articles: List[Dict[str, Any]]
    statistical_data: List[Dict[str, Any]]
    
    # Analysis results
    meta_analysis_results: Dict[str, Any]
    forest_plots: List[Dict[str, Any]]
    
    # Iteration control
    global_iterations: int
    agent_iterations: Dict[str, int]
    quality_scores: Dict[str, float]
    
    # Communication
    messages: List[BaseMessage]
```

#### State Reducers (`reducers.py`)
- **add_messages**: Manages message history between agents
- **merge_dicts**: Handles complex nested dictionary updates
- **append_list**: Manages list-based data accumulation
- **increment_counter**: Tracks iteration counts and metrics

### 3. Tool System

#### Research Tools (`research_tools.py`)
- **search_literature**: Uses Tavily API with medical domain focus
- **generate_search_queries**: LLM-powered PICO-based query generation
- **assess_article_relevance**: LLM-based relevance scoring against criteria

#### Processing Tools (`processing_tools.py`)
- **extract_article_content**: Tavily Extract API integration
- **extract_statistical_data**: LLM-powered statistical data extraction
- **generate_vancouver_citation**: Academic citation generation
- **chunk_and_vectorize**: Text chunking with OpenAI embeddings

#### Analysis Tools (`analysis_tools.py`)
- **calculate_meta_analysis**: LLM-guided statistical meta-analysis
- **create_forest_plot**: Publication-quality forest plot generation
- **assess_heterogeneity**: Comprehensive heterogeneity interpretation
- **perform_sensitivity_analysis**: Robustness testing

#### Handoff Tools (`handoff_tools.py`)
- **transfer_to_[agent]**: Structured agent-to-agent communication
- **signal_completion**: Task completion signaling
- **request_supervisor_intervention**: Error escalation

### 4. Graph Architecture

#### Multi-Agent Graph (`meta_analysis_graph.py`)
```python
class MetaAnalysisGraph:
    def __init__(self, settings: Settings):
        self.graph = StateGraph(IterationState)
        self.checkpointer = MemorySaver()  # Persistence
        self.store = InMemoryStore()       # Long-term memory
        
    def _build_graph(self):
        # Add all agents as nodes
        builder.add_node("supervisor", supervisor_agent)
        builder.add_node("researcher", researcher_agent)
        builder.add_node("processor", processor_agent)
        builder.add_node("analyst", analyst_agent)
        
        # Hub-and-spoke edges
        builder.add_edge(START, "supervisor")
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("processor", "supervisor")
        builder.add_edge("analyst", "supervisor")
```

## Execution Flow

### 1. Initialization Phase
1. User provides research question
2. Supervisor analyzes question and creates initial PICO framework
3. System state initialized with configuration parameters

### 2. Literature Search Phase
1. Supervisor transfers to Researcher Agent
2. Researcher generates optimized search queries
3. Searches multiple medical databases
4. Assesses article relevance
5. Transfers URLs to Processor or returns to Supervisor

### 3. Content Processing Phase
1. Supervisor transfers to Processor Agent
2. Processor extracts full article content
3. Extracts statistical data using LLM analysis
4. Generates citations and creates vector embeddings
5. Transfers to Analyst or returns to Supervisor

### 4. Statistical Analysis Phase
1. Supervisor transfers to Analyst Agent
2. Analyst performs meta-analysis calculations
3. Creates forest plots and assesses heterogeneity
4. Conducts sensitivity analyses
5. Transfers to Writer or returns to Supervisor

### 5. Report Generation Phase
1. Supervisor transfers to Writer Agent (when implemented)
2. Writer synthesizes findings into structured report
3. Transfers to Reviewer for quality assessment
4. Final editing and formatting

## Quality Control

### Multi-Level Quality Assurance
1. **Agent-Level**: Each agent has specific quality thresholds
2. **Component-Level**: Quality scoring for each system component
3. **Overall-Level**: Global quality assessment for completion
4. **Iterative Refinement**: Agents can iterate to improve quality

### Quality Thresholds
- **Researcher**: ≥70% article relevance, ≥10 articles minimum
- **Processor**: ≥80% extraction success, ≥70% statistical data extraction
- **Analyst**: Statistical significance, appropriate model selection
- **Overall**: ≥80% quality score for completion

## Error Handling and Recovery

### Retry Patterns
- **Exponential Backoff**: For network and API failures
- **Circuit Breaker**: For systematic failures
- **Graceful Degradation**: Partial results when possible

### Intervention Mechanisms
- **Supervisor Intervention**: Complex decision escalation
- **Quality Check Requests**: Methodology validation
- **Force Stop**: Emergency termination with partial results

## Memory and Persistence

### Short-Term Memory (Checkpointers)
- **MemorySaver**: In-memory for development
- **PostgresSaver**: Production persistence
- **RedisSaver**: High-performance caching

### Long-Term Memory (Stores)
- **InMemoryStore**: Development and testing
- **PostgresStore**: Production knowledge base
- **Semantic Search**: Vector-based similarity search

## Configuration and Deployment

### Environment Configuration
- **API Keys**: OpenAI and Tavily API integration
- **Model Selection**: Configurable LLM models
- **Quality Thresholds**: Adjustable quality parameters
- **Timeout Controls**: Execution time limits

### Production Considerations
- **Database Backends**: PostgreSQL/Redis for persistence
- **Monitoring**: Comprehensive logging and metrics
- **Scalability**: Horizontal scaling capabilities
- **Security**: API key management and access control

## AI-First Design Philosophy

### LLM Integration Points
1. **Query Generation**: PICO-based search strategy
2. **Relevance Assessment**: Article screening and selection
3. **Data Extraction**: Statistical data identification
4. **Analysis Interpretation**: Meta-analysis methodology
5. **Report Generation**: Synthesis and documentation

### Advantages of AI-First Approach
- **Adaptability**: Handles diverse research questions
- **Quality**: Expert-level analysis capabilities
- **Efficiency**: Automated complex reasoning tasks
- **Scalability**: No manual rule maintenance required

## Future Extensions

### Additional Agents (Planned)
- **Retriever Agent**: Vector store search and synthesis
- **Writer Agent**: Report generation and formatting
- **Reviewer Agent**: Quality assessment and validation
- **Editor Agent**: Final document preparation

### Enhanced Capabilities
- **Subgroup Analysis**: Population-specific meta-analyses
- **Network Meta-Analysis**: Multiple intervention comparisons
- **Individual Patient Data**: Advanced statistical methods
- **Real-Time Updates**: Continuous literature monitoring

This architecture provides a robust, scalable, and maintainable foundation for automated meta-analysis generation while maintaining the flexibility to handle diverse research questions and methodological requirements.