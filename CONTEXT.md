# CONTEXT.md - Metanalyst Agent Development Plan

## Project Overview

The **metanalyst-agent** is the first open-source project by Nobrega Medtech, focused on automated meta-analysis generation using Python and LangGraph. The project implements a multi-agent system with intelligent orchestration for medical literature research, extraction, analysis, and report generation.

Useful links:
https://langchain-ai.github.io/langgraph/reference/
https://docs.tavily.com/documentation/api-reference/endpoint/search
https://docs.tavily.com/documentation/api-reference/endpoint/extract

### Main Objective
Create an automated system that performs the entire medical meta-analysis process, from literature search to generating final reports with statistical analyses, graphs, and forest plots.

## System Architecture

### Hub-and-Spoke Architecture with Agents-as-a-Tool

The system follows a "sun" architecture, where a central orchestrator agent invokes specialized agents as tools based on the current state:

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

**Architecture Principles:**
- **Central Hub**: Orchestrator maintains global state and decision logic
- **Agents as Tools**: Each agent is a specialized tool
- **Contextual Decision**: At each iteration, the orchestrator analyzes the state and chooses the next agent
- **Direct Communication**: All agents communicate only with the orchestrator
- **Shared State**: The orchestrator maintains and updates the global state

## Context Engineering and Agents-as-a-Tool Pattern

### Central Orchestrator Philosophy

The **Orchestrator Agent** works like a conductor who, at each iteration, analyzes the current meta-analysis state and decides which specialized agent should be invoked as a tool. This pattern ensures:

- **Centralized Control**: All decision logic resides in the orchestrator
- **Single State**: A single point of truth for the global state
- **Flexibility**: Ability to repeat or skip steps based on context
- **Observability**: All decisions are transparent and loggable



### Agent Details

#### 1. Orchestrator Agent (Central Hub)
- **Role**: Conductor orchestrating the entire meta-analysis symphony
- **Responsibilities:**
  - Define and refine the PICO of the research
  - Analyze current state and decide next agent
  - Manage global state (messages, URLs, data, progress)
  - Implement retry logic and quality control
  - Maintain decision history and justifications
- **Tools**: All other agents as tools
- **Context**: Maintains complete meta-analysis context from start to finish

#### 2. Researcher Agent
- **Tools:** Tavily Search API
- **Responsibilities:**
  - Search scientific literature using specific domains
  - Generate queries based on PICO
  - Return list of candidate URLs
  - Filter results by relevance and quality

#### 3. Processor Agent (Combines Extraction + Vectorization)
- **Tools:** Tavily Extract API, OpenAI Embeddings (text-embedding-3-small), FAISS, GPT-4.1-nano
- **Responsibilities:**
  - Extract full content from URLs using Tavily Extract
  - Process markdown to structured JSON using GPT-4.1-nano
  - Generate Vancouver references
  - Create objective summaries with statistical data
  - Chunk scientific publications intelligently (1000 chars, 100 overlap)
  - Maintain reference tracking for each chunk
  - Generate vector embeddings with text-embedding-3-small
  - Store in local FAISS vector store
  - Manage temporary files and cleanup
  - Update state with processed URLs and vector store status

#### 4. Retriever Agent
- **Tools:** FAISS, Cosine Similarity
- **Responsibilities:**
  - Search for relevant information using PICO
  - Return high-similarity chunks
  - Maintain reference context

#### 6. Writer Agent
- **Responsibilities:**
  - Analyze retrieved chunks
  - Generate structured HTML report
  - Include methodology and results
  - Appropriately cite references

#### 7. Reviewer Agent
- **Responsibilities:**
  - Review report quality
  - Generate improvement feedback
  - Suggest additional searches if necessary
  - Validate compliance with medical standards

#### 8. Analyst Agent
- **Tools:** Matplotlib, Plotly, SciPy, NumPy
- **Responsibilities:**
  - Statistical analyses (meta-analysis, forest plots)
  - Generate graphs and tables
  - Calculate metrics (OR, RR, CI)
  - Create HTML visualizations

#### 9. Editor Agent
- **Responsibilities:**
  - Integrate report + analyses
  - Generate final HTML
  - Structure final document
  - Ensure appropriate formatting

  # EXAMPLES #

## Integration with External APIs

### Tavily Search API

```python
from tavily import TavilyClient

# Basic configuration
client = TavilyClient(api_key="tvly-YOUR_API_KEY")

# Search with specific parameters
response = client.search(
    query="diabetes type 2 metformin efficacy",
    search_depth="advanced",  # Best for medical research
    max_results=15,
    include_domains=[
        "pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov/pmc", "www.cochranelibrary.com",
    "lilacs.bvsalud.org", "scielo.org", "www.embase.com", "www.webofscience.com",
    "www.scopus.com", "www.epistemonikos.org", "www.ebscohost.com",
    "www.tripdatabase.com", "pedro.org.au", "doaj.org", "scholar.google.com",
    "clinicaltrials.gov", "apps.who.int/trialsearch", "www.clinicaltrialsregister.eu",
    "www.isrctn.com", "www.thelancet.com", "www.nejm.org", "jamanetwork.com",
    "www.bmj.com", "www.nature.com/nm", "www.acpjournals.org/journal/aim",
    "journals.plos.org/plosmedicine", "www.jclinepi.com",
    "systematicreviewsjournal.biomedcentral.com", "ascopubs.org/journal/jco",
    "www.ahajournals.org/journal/circ", "www.gastrojournal.org",
    "academic.oup.com/eurheartj", "www.archives-pmr.org", "www.jacc.org",
    "www.scielo.br",
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "acpjournals.org/journal/aim",
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
    "openevidence.com"],
    topic="general"
)
```

**Important Resources:**
- `search_depth="advanced"` for more precise results
- `include_domains` to focus on reliable medical sources
- Controllable `max_results` (5-20)
- Date filtering with `time_range`

### Tavily Extract API

```python
# Content extraction
response = client.extract(
    urls=["https://pubmed.ncbi.nlm.nih.gov/article/123456"],
    extract_depth="advanced",  # Essential for tables and structured data
    format="markdown"
)

# Process result
for result in response['results']:
    url = result['url']
    content = result['raw_content']
    # Process content to JSON structure
```

**Important Resources:**
- `extract_depth="advanced"` for tables and structured content
- `format="markdown"` for better processing
- Rate limiting: 1 credit per 5 URLs (basic), 2 credits per 5 URLs (advanced)

## LangGraph - Multi-Agent Framework

### Fundamental Concepts

#### 1. ReAct Agents
```python
from langgraph.prebuilt import create_react_agent

# Basic agent with tools
agent = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=[tavily_search_tool, extract_tool],
    prompt="You are an expert in medical research..."
)
```

#### 2. Agents-as-a-Tool Pattern
```python
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated

# Each specialized agent is an orchestrator tool
@tool
def call_researcher_agent(
    pico: Dict[str, str],
    state: Annotated[MetanalysisState, InjectedState]
) -> Dict:
    """
    Invoke the researcher agent to search scientific literature.
    
    Args:
        pico: PICO structure to guide the search
        state: Current meta-analysis state
    
    Returns:
        Search results with found URLs
    """
    # Invoke researcher agent as a tool
    researcher_result = researcher_agent.invoke({
        "messages": [{"role": "user", "content": f"Search literature for PICO: {pico}"}],
        "pico": pico
    })
    
    # Process result
    urls_found = extract_urls_from_result(researcher_result)
    
    return {
        "urls_found": urls_found,
        "research_summary": researcher_result["messages"][-1].content,
        "timestamp": datetime.now().isoformat()
    }

@tool
def call_extractor_agent(
    urls: List[str],
    state: Annotated[MetanalysisState, InjectedState]
) -> Dict:
    """
    Invoke the extractor agent to extract content from URLs.
    
    Args:
        urls: List of URLs for extraction
        state: Current meta-analysis state
    
    Returns:
        Extracted papers in structured format
    """
    # Invoke extractor agent as a tool
    extractor_result = extractor_agent.invoke({
        "messages": [{"role": "user", "content": f"Extract content from {len(urls)} URLs"}],
        "urls": urls
    })
    
    # Process result
    extracted_papers = process_extraction_result(extractor_result)
    
    return {
        "extracted_papers": extracted_papers,
        "extraction_summary": extractor_result["messages"][-1].content,
        "urls_processed": urls
    }

# Orchestrator with all tools
orchestrator_tools = [
    call_researcher_agent,
    call_extractor_agent,
    call_vectorizer_agent,
    call_retriever_agent,
    call_writer_agent,
    call_reviewer_agent,
    call_analyst_agent,
    call_editor_agent
]

# Create orchestrator as ReAct agent
orchestrator = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=orchestrator_tools,
    prompt="""
    You are the central orchestrator of an automated medical meta-analysis.
    
    Your function is to analyze the current meta-analysis state and decide which specialized
    agent should be invoked next. You have access to 8 specialized agents as tools:
    
    1. call_researcher_agent - Search scientific literature
    2. call_extractor_agent - Extract content from URLs
    3. call_vectorizer_agent - Create embeddings and vector store
    4. call_retriever_agent - Search relevant information
    5. call_writer_agent - Generate structured report
    6. call_reviewer_agent - Review report quality
    7. call_analyst_agent - Perform statistical analyses
    8. call_editor_agent - Integrate final report
    
    At each iteration, analyze the current state and invoke the next necessary agent.
    Maintain complete context and justify your decisions.
    """
)
```

#### 3. Shared State
```python
from langgraph.graph import MessagesState
from typing import TypedDict, List

class MetanalysisState(TypedDict):
    messages: List[BaseMessage]
    pico: Dict[str, str]
    urls_found: List[str]
    urls_processed: List[str]
    extracted_papers: List[Dict]
    vector_store_ready: bool
    analysis_complete: bool
```

#### 4. Multi-Agent Orchestration
```python
from langgraph.graph import StateGraph, START, END

# Define main graph
workflow = StateGraph(MetanalysisState)

# Add agents
workflow.add_node("orchestrator", orchestrator_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("extractor", extractor_agent)
workflow.add_node("vectorizer", vectorizer_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("editor", editor_agent)

# Define flow
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    lambda state: determine_next_agent(state),
    {
        "researcher": "researcher",
        "extractor": "extractor",
        "vectorizer": "vectorizer",
        "retriever": "retriever",
        "writer": "writer",
        "reviewer": "reviewer",
        "analyst": "analyst",
        "editor": "editor",
        END: END
    }
)

# Compile graph
app = workflow.compile()
```

### Implementation Patterns

#### 1. Handoff Between Agents
```python
from langgraph.prebuilt import InjectedState, InjectedToolCallId

def create_handoff_tool(agent_name: str, description: str):
    @tool(f"transfer_to_{agent_name}", description=description)
    def handoff_tool(
        state: Annotated[MetanalysisState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        return Command(
            goto=agent_name,
            update={"current_agent": agent_name},
            graph=Command.PARENT,
        )
    return handoff_tool
```

#### 2. Functional API for Complex Workflows
```python
from langgraph.func import entrypoint, task

@task
def search_literature(pico: Dict[str, str]):
    """Search medical literature based on PICO."""
    query = f"{pico['patient']} {pico['intervention']} {pico['comparison']} {pico['outcome']}"
    return tavily_client.search(query, search_depth="advanced")

@task
def extract_papers(urls: List[str]):
    """Extract content from paper URLs."""
    return tavily_client.extract(urls, extract_depth="advanced")

@entrypoint()
def metanalysis_workflow(pico: Dict[str, str]):
    # Search literature
    search_results = search_literature(pico).result()
    
    # Extract papers
    papers = extract_papers(search_results['urls']).result()
    
    # Process and analyze
    analysis = process_papers(papers).result()
    
    return analysis
```

#### 3. Memory and State Management
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Configure persistence
checkpointer = MemorySaver()
store = InMemoryStore()

# Agent with memory
agent = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=[search_tool, extract_tool],
    checkpointer=checkpointer,
    store=store
)
```

## Data Structures

### 1. PICO Structure
```python
class PICO(TypedDict):
    patient: str      # Patient/Population
    intervention: str # Intervention
    comparison: str   # Comparison
    outcome: str      # Outcome
```

### 2. Extracted Paper Structure
```python
class ExtractedPaper(TypedDict):
    reference: str    # Vancouver Reference
    url: str         # Original URL
    content: str     # Processed Summary
    metadata: Dict   # Metadata (authors, year, journal, etc.)
    statistics: Dict # Extracted statistical data
```

### 3. Analysis Result Structure
```python
class AnalysisResult(TypedDict):
    forest_plot: str     # Path to forest plot
    summary_stats: Dict  # Summarized statistics
    tables: List[Dict]   # Generated tables
    graphs: List[str]    # Paths to graphs
    conclusions: str     # Analysis conclusions
```

## Implementation Examples

### 1. Researcher Agent
```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_medical_literature(query: str, max_results: int = 15):
    """Search medical literature using Tavily."""
    return tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_domains=[
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
            "cochrane.org",
            "bmj.com",
            "nejm.org"
        ],
        topic="general"
    )

researcher_agent = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=[search_medical_literature],
    prompt="""
    You are a medical research expert. Your function is to:
    1. Analyze the provided PICO
    2. Generate optimized search queries
    3. Search relevant medical literature
    4. Return high-quality URLs
    
    Focus on clinical studies, meta-analyses, and systematic reviews.
    Prioritize reliable sources like PubMed, Cochrane, and high-impact journals.
    """
)
```

### 2. Extractor Agent
```python
@tool
def extract_paper_content(urls: List[str]):
    """Extract content from paper URLs."""
    return tavily_client.extract(
        urls=urls,
        extract_depth="advanced",
        format="markdown"
    )

@tool
def process_to_json(content: str, url: str):
    """Process extracted content to structured JSON."""
    # Use LLM to process content
    prompt = f"""
    Process the following medical content and return a structured JSON:
    
    Content: {content}
    URL: {url}
    
    Return:
    {{
        "reference": "Complete Vancouver reference",
        "url": "{url}",
        "content": "Objective summary with statistical data",
        "metadata": {{
            "authors": ["author1", "author2"],
            "year": 2023,
            "journal": "Journal name",
            "doi": "DOI if available"
        }},
        "statistics": {{
            "sample_size": 1000,
            "relative_risk": 0.85,
            "confidence_interval": [0.70, 1.03],
            "p_value": 0.045
        }}
    }}
    """
    # Implement processing with LLM
    pass

extractor_agent = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=[extract_paper_content, process_to_json],
    prompt="""
    You are an expert in medical data extraction. Your function is to:
    1. Extract full content from medical URLs
    2. Process text to JSON structure
    3. Generate precise Vancouver references
    4. Extract relevant statistical data
    
    Focus on quantitative data: sample size, RR, OR, CI, p-values.
    """
)
```

### 3. Vectorizer Agent
```python
import faiss
import numpy as np
from openai import OpenAI

@tool
def chunk_papers(papers: List[ExtractedPaper], chunk_size: int = 500):
    """Chunk papers maintaining reference tracking."""
    chunks = []
    for paper in papers:
        content = paper['content']
        # Implement intelligent chunking
        paper_chunks = intelligent_chunk(content, chunk_size)
        for chunk in paper_chunks:
            chunks.append({
                'text': chunk,
                'reference': paper['reference'],
                'url': paper['url'],
                'metadata': paper['metadata']
            })
    return chunks

@tool
def generate_embeddings(chunks: List[Dict]):
    """Generate embeddings and store in FAISS."""
    client = OpenAI()
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk['text']
        )
        embeddings.append(response.data[0].embedding)
    
    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index
    faiss.write_index(index, "metanalysis_index.faiss")
    
    return {"status": "success", "total_chunks": len(chunks)}

vectorizer_agent = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=[chunk_papers, generate_embeddings],
    prompt="""
    You are an expert in document vectorization. Your function is to:
    1. Chunk scientific publications intelligently
    2. Maintain reference traceability
    3. Generate vector embeddings using OpenAI
    4. Store in FAISS vector store
    
    Preserve scientific context when chunking documents.
    """
)
```

### 4. Analyst Agent
```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats

@tool
def perform_meta_analysis(papers: List[ExtractedPaper]):
    """Perform statistical meta-analysis."""
    # Extract statistical data
    effect_sizes = []
    variances = []
    
    for paper in papers:
        stats = paper.get('statistics', {})
        if 'relative_risk' in stats and 'confidence_interval' in stats:
            rr = stats['relative_risk']
            ci = stats['confidence_interval']
            
            # Calculate log RR and variance
            log_rr = np.log(rr)
            se = (np.log(ci[1]) - np.log(ci[0])) / (2 * 1.96)
            variance = se ** 2
            
            effect_sizes.append(log_rr)
            variances.append(variance)
    
    # Meta-analysis using random effects
    weights = [1/v for v in variances]
    pooled_effect = np.average(effect_sizes, weights=weights)
    pooled_variance = 1 / sum(weights)
    
    return {
        'pooled_rr': np.exp(pooled_effect),
        'confidence_interval': [
            np.exp(pooled_effect - 1.96 * np.sqrt(pooled_variance)),
            np.exp(pooled_effect + 1.96 * np.sqrt(pooled_variance))
        ],
        'p_value': stats.norm.sf(abs(pooled_effect) / np.sqrt(pooled_variance)) * 2
    }

@tool
def create_forest_plot(papers: List[ExtractedPaper], meta_result: Dict):
    """Create forest plot visualization."""
    fig = go.Figure()
    
    # Add individual studies
    for i, paper in enumerate(papers):
        stats = paper.get('statistics', {})
        if 'relative_risk' in stats:
            rr = stats['relative_risk']
            ci = stats['confidence_interval']
            
            fig.add_trace(go.Scatter(
                x=[rr],
                y=[i],
                mode='markers',
                marker=dict(size=10),
                name=f"Study {i+1}",
                error_x=dict(
                    type='data',
                    array=[ci[1] - rr],
                    arrayminus=[rr - ci[0]]
                )
            ))
    
    # Add pooled result
    pooled_rr = meta_result['pooled_rr']
    pooled_ci = meta_result['confidence_interval']
    
    fig.add_trace(go.Scatter(
        x=[pooled_rr],
        y=[len(papers)],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Pooled Effect',
        error_x=dict(
            type='data',
            array=[pooled_ci[1] - pooled_rr],
            arrayminus=[pooled_rr - pooled_ci[0]]
        )
    ))
    
    fig.update_layout(
        title='Forest Plot - Meta-Analysis Results',
        xaxis_title='Relative Risk',
        yaxis_title='Studies',
        showlegend=True
    )
    
    # Save as HTML
    fig.write_html('forest_plot.html')
    return 'forest_plot.html'

analyst_agent = create_react_agent(
    model="claude-sonnet-4-20250514",
    tools=[perform_meta_analysis, create_forest_plot],
    prompt="""
    You are an expert in statistical analysis and meta-analysis. Your function is to:
    1. Perform statistical analyses of extracted data
    2. Calculate metrics like RR, OR, 95% CI
    3. Generate forest plots and other visualizations
    4. Create result tables
    
    Use appropriate statistical methods for meta-analysis.
    """
)

## Current Implementation Status

### Implemented Agents
- ✅ **Orchestrator Agent**: Central hub for decision-making and state management
- ✅ **Researcher Agent**: Literature search using Tavily API
- ✅ **Processor Agent**: Combined extraction and vectorization pipeline
- 🔄 **Retriever Agent**: In development
- 🔄 **Writer Agent**: In development  
- 🔄 **Reviewer Agent**: In development
- 🔄 **Analyst Agent**: In development
- 🔄 **Editor Agent**: In development

### Current Architecture Implementation

The system now uses a **Processor Agent** that combines the original Extractor and Vectorizer functionalities into a single, efficient pipeline. This approach:

1. **Reduces State Complexity**: Eliminates intermediate storage of extracted papers
2. **Improves Performance**: Processes URLs directly to vector store in one operation
3. **Maintains Token Limits**: Keeps only URLs in state, not full content
4. **Enables Parallel Processing**: Batches URL processing for efficiency

### State Management

The current state structure includes:
- `url_not_processed`: URLs waiting to be processed
- `url_processed`: URLs already processed by the Processor Agent
- `vector_store_ready`: Boolean indicating if vector store is ready
- `pico`: PICO structure for research guidance

### Key Implementation Details

#### Processor Agent Pipeline
1. **Extract**: Uses Tavily Extract API to get markdown content
2. **Process**: GPT-4.1-nano converts markdown to structured JSON with Vancouver references
3. **Chunk**: Intelligent chunking with 1000 chars + 100 overlap
4. **Vectorize**: OpenAI text-embedding-3-small generates embeddings
5. **Store**: FAISS vector store with metadata preservation
6. **Cleanup**: Automatic cleanup of temporary files

#### Error Handling
- Comprehensive error handling with graceful degradation
- Batch processing to handle API rate limits
- Proper fallback mechanisms for JSON parsing failures
- Type ignore directives for known FAISS library issues

#### File Structure Updates
```
metanalyst-agent/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py ✅
│   │   ├── researcher.py ✅
│   │   ├── processor.py ✅ (combines extraction + vectorization)
│   │   ├── retriever.py 🔄
│   │   ├── writer.py 🔄
│   │   ├── reviewer.py 🔄
│   │   ├── analyst.py 🔄
│   │   └── editor.py 🔄
│   ├── tools/
│   │   ├── tavily_tools.py ✅
│   │   ├── processor_tools.py ✅
│   │   ├── orchestrator_tools.py ✅
│   │   └── (other tools in development)
│   ├── models/
│   │   ├── state.py ✅
│   │   └── schemas.py ✅
│   └── utils/
│       └── config.py ✅
```

### Next Development Steps
1. **Complete Retriever Agent**: Vector similarity search
2. **Implement Writer Agent**: HTML report generation
3. **Add Reviewer Agent**: Quality control and feedback
4. **Develop Analyst Agent**: Statistical analysis and forest plots
5. **Create Editor Agent**: Final report integration

---

### Original Prompt

Esse foi meu prompt original:

Vamos começar hoje o primeiro projeto open source da Nobrega Medtech. O metanalyst-agente. Um agente usando python e langgraph para a geração automatizada de metanálises.

Estamos começando do zero aqui.

Em termos de arquitetura, pensei em um fluxo orchestrator com subagents. Ou seja, um agente orquestrador, que chama outros agentes por tool calls. Cada um dos agentes terá suas próprias tools e responsabilidades. Usaremos o claude em todos eles.

Teremos no grafo:

1-Agente orquestrador: chama qual é o próximo agente a ser executado com base no estado. Ele também define a PICO que guiará a pesquisa.
2-Agente pesquisador: pesquisa literatura científica em domínios específicos do campo include_domains do tavily_search, com o intuito de encontrar urls candidatos.
3-Agente extrator: usa uma tool de tavily extract para extrair uma determinada url, obtendo o texto e, entao, dispondo em um json com uma estrutura de duas chaves: reference, url e content. A chave reference guardará a referencia para aquela publicacao em formato vancouver (que o agente obtem por meio da extração, descobrindo autores, ano, titulo, revista, doi, edicao, etc), já a chave content guardará um resumo daquela publicacao, mantendo as informacoes mais importantes e insights objetivos, como tamanho da amostra, risco relativo, odds ratio, etc. O agente pode ter uma tool que faz todo esse processo, ou entao uma tool de extracao e uma tool de processamento. O processamento também será feito por LLMs. Os objetos json com reference e content sao armazenados no estado.
4-Agente Vetorizador: Tem uma tool que vai chunkar as publicacoes cientificas (objetos com reference, url e content) do estado, mantendo reference como indexacao e rastreio da origem de cada chunk, além da tool para gerar os embeddings desses vetores por meio do modelo text-embedding-small-3 da openai. As tool que gera os vetores vai, ao fim da execução, subi-los em um vectorbase local, talvez FAISS? Ao fim o estado deve estar limpo, mantendo apenas as urls que já foram acessadas para evitar repeticoes.
5-Agente retriever: Busca por cosine a partir da PICO, que já está armazenada no estado, no vector store pelas informacoes relevantes de acordo com as demandas do orquestrador. Armazena os chunks selecionados no estado.
6-Agente escritor: Analisa as informacoes obtidas e gera uma versao do relatorio final. Relatorio em html.
7-Agente revisor: Revisa o relatorio gerado e gera feedbacks para o orquestrador, que poderá chamar o retriever ou até o pesquisador novamente. 
8-Agente analista: Le o relatorio gerado e plotta tabelas e gráficos, bem como análises matemáticas e estatísticas usando tools adequadas para isso. Gera um html com as análises, gráficos, tabelas, etc. Deve incluir, se possível, um forest plot.
9-Editor: Junta o relatorio gerado com as analises e plots gerados para formar o html final da metanalise.
