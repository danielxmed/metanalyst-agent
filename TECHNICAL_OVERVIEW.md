# Metanalyst-Agent: Technical Overview

## Architecture

The Metanalyst-Agent implements a **Hub-and-Spoke Architecture** with **Agents-as-a-Tool**, where a central orchestrator agent invokes specialized agents as tools based on the current state.

### System Components

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
                     (All agents are
                      autonomous ReAct agents)

    ● = Central Orchestrator (Hub)
    │ = Direct Connections (Agents-as-a-Tool)
```

### Agent Responsibilities

1. **Orchestrator Agent**: Central hub that conducts the meta-analysis symphony
2. **Researcher Agent**: Searches scientific literature using Tavily and specialized domains
3. **Processor Agent**: Extracts content and vectorizes articles using Tavily Extract + OpenAI
4. **Retriever Agent**: Performs semantic search on the vector store
5. **Analyst Agent**: Performs statistical analyses and creates visualizations
6. **Writer Agent**: Generates structured HTML reports with citations
7. **Reviewer Agent**: Reviews report quality and suggests improvements
8. **Editor Agent**: Integrates final report with analyses

## AI-First Approach

Every complex task is handled by LLMs rather than heuristics:

- **Literature Search**: LLM generates optimized search queries based on PICO
- **Relevance Assessment**: LLM evaluates article relevance against criteria
- **Data Extraction**: LLM extracts statistical data from article content
- **Citation Generation**: LLM creates Vancouver-style citations
- **Quality Assessment**: LLM evaluates study quality and meta-analysis completeness
- **Report Writing**: LLM generates comprehensive meta-analysis reports

## Technical Stack

### Core Technologies
- **LangGraph**: Multi-agent orchestration and state management
- **LangChain**: LLM integration and tool management
- **OpenAI GPT-4**: Primary LLM for complex reasoning tasks
- **Tavily API**: Web search and content extraction
- **FAISS**: Vector storage for semantic search
- **PostgreSQL**: Persistent checkpoints and long-term memory
- **Redis**: High-performance caching (optional)

### Scientific Computing
- **SciPy**: Statistical calculations for meta-analysis
- **NumPy**: Numerical computations
- **Matplotlib/Plotly**: Forest plots and visualizations
- **Pandas**: Data manipulation and analysis

## State Management

The system uses a comprehensive state structure (`MetaAnalysisState`) that tracks:

### Core Information
- Meta-analysis ID and thread ID
- Current phase and active agent
- PICO framework and research question
- Inclusion/exclusion criteria

### Data Processing
- Candidate URLs and processing queue
- Processed articles with statistical data
- Vector store status and embeddings
- Retrieved chunks and analysis results

### Quality Control
- Quality scores per agent and component
- Iteration counts and performance metrics
- Error tracking and recovery information

### Persistence
- Checkpoints for recovery
- Agent logs and execution timeline
- Final reports and citations

## Multi-Agent Coordination

### Hub-and-Spoke Pattern
- **Central Control**: Orchestrator maintains global state and decision logic
- **Specialized Agents**: Each agent is an autonomous ReAct agent with specific tools
- **Tool-based Communication**: Agents transfer control using handoff tools
- **Shared State**: Single point of truth maintained by orchestrator

### Decision Logic
The orchestrator uses intelligent routing based on:
- Current phase of meta-analysis
- Quality scores and completion criteria
- Available data and processing status
- Agent iteration limits and performance

### Iteration Control
- **Recursion Limits**: Configurable limits prevent infinite loops
- **Quality Thresholds**: Agents iterate until quality criteria are met
- **Circuit Breakers**: Automatic failure detection and recovery
- **Graceful Termination**: Saves partial results when limits are reached

## Memory Architecture

### Short-term Memory (Checkpoints)
- **Thread-level persistence** during execution
- **State snapshots** at regular intervals
- **Recovery capability** from interruptions
- **PostgreSQL/Redis backends** for reliability

### Long-term Memory (Store)
- **Cross-thread information** sharing
- **Processed article storage** with metadata
- **Vector embeddings** for semantic search
- **Performance metrics** for learning

## Quality Assurance

### Multi-level Quality Control
1. **Agent-level**: Each agent self-assesses quality
2. **Component-level**: Specific quality metrics per task
3. **System-level**: Overall meta-analysis quality score
4. **Human-level**: Optional human intervention points

### Quality Metrics
- **Researcher**: Article relevance rate (>70%)
- **Processor**: Extraction success rate (>80%)
- **Analyst**: Statistical confidence (>85%)
- **Writer**: Report completeness (>80%)
- **Reviewer**: Scientific rigor (>90%)

### Iteration and Refinement
- Agents can iterate on their work until quality thresholds are met
- Automatic retry logic with exponential backoff
- Sensitivity analysis and robustness checks
- Human intervention requests for complex issues

## API Integration

### OpenAI Integration
- **GPT-4** for complex reasoning and analysis
- **Text-embedding-3-small** for vector embeddings
- **Structured outputs** for reliable data extraction
- **Temperature control** for consistent results

### Tavily Integration
- **Search API** for literature discovery
- **Extract API** for full-text content extraction
- **Domain restrictions** for scientific databases
- **Rate limiting** and error handling

## Scalability Features

### Horizontal Scaling
- **Stateless agents** can run on multiple machines
- **Shared persistence** through PostgreSQL/Redis
- **Load balancing** across agent instances
- **Distributed processing** for large meta-analyses

### Vertical Scaling
- **Configurable recursion limits** based on resources
- **Batch processing** for multiple articles
- **Memory management** with cleanup routines
- **Resource monitoring** and optimization

## Security and Privacy

### Data Protection
- **No persistent storage** of article content by default
- **API key management** through environment variables
- **Database encryption** support
- **Audit logging** for compliance

### Access Control
- **Thread-based isolation** between analyses
- **User identification** in configurations
- **Resource quotas** and rate limiting
- **Emergency stop** mechanisms

## Development and Testing

### Code Organization
```
metanalyst_agent/
├── agents/           # Agent implementations
├── tools/           # Specialized tool functions
├── state/           # State management
├── graph/           # Multi-agent graph construction
├── config/          # Configuration management
└── utils/           # Utility functions
```

### Testing Strategy
- **Unit tests** for individual tools
- **Integration tests** for agent interactions
- **End-to-end tests** with mock data
- **Performance benchmarks** for optimization

### Development Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables: `cp .env.example .env`
3. Run examples: `python example_usage.py`
4. Run tests: `pytest tests/`

## Deployment Options

### Local Development
- **Memory backends** for testing
- **SQLite** for simple persistence
- **Docker Compose** for full stack

### Production Deployment
- **PostgreSQL** for robust persistence
- **Redis** for high-performance caching
- **Load balancers** for multiple instances
- **Monitoring** and alerting systems

### Cloud Deployment
- **Kubernetes** for orchestration
- **Managed databases** (AWS RDS, Google Cloud SQL)
- **Auto-scaling** based on demand
- **CI/CD pipelines** for updates

## Performance Characteristics

### Typical Performance
- **Processing Time**: 15-45 minutes for 20-50 articles
- **Memory Usage**: 512MB-2GB depending on article count
- **API Calls**: 100-500 calls per meta-analysis
- **Database Storage**: 10-100MB per analysis

### Optimization Strategies
- **Parallel processing** of articles
- **Caching** of embeddings and extractions
- **Batch operations** where possible
- **Resource pooling** for connections

## Monitoring and Observability

### Logging
- **Structured logging** with timestamps
- **Agent-specific** log streams
- **Performance metrics** tracking
- **Error reporting** and alerting

### Metrics
- **Quality scores** per agent and overall
- **Processing times** and bottlenecks
- **Success rates** and error frequencies
- **Resource utilization** monitoring

## Future Enhancements

### Planned Features
- **Network meta-analysis** support
- **Real-time collaboration** capabilities
- **Advanced visualizations** (interactive plots)
- **Integration** with reference managers

### Research Directions
- **Multi-modal analysis** (images, tables)
- **Automated quality assessment** improvements
- **Domain-specific** agent specializations
- **Federated learning** across institutions

## Contributing

The system is designed for extensibility:
- **New agents** can be added easily
- **Custom tools** integrate seamlessly
- **Configuration-driven** behavior
- **Plugin architecture** for extensions

For detailed implementation guidelines, see the development documentation and code comments throughout the system.