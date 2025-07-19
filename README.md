# Metanalyst-Agent

The first open-source project by Nobrega Medtech for automated meta-analysis generation using Python and LangGraph.

## Overview

Metanalyst-Agent is an AI-first multi-agent system that performs complete medical meta-analyses automatically, from literature search to generating final reports with statistical analyses, graphs, and forest plots.

## Architecture

The system follows a **Hub-and-Spoke Architecture** with **Agents-as-a-Tool**, where a central orchestrator agent invokes specialized agents as tools based on the current state.

### System Agents

- **Orchestrator Agent**: Central hub that conducts the meta-analysis symphony
- **Researcher Agent**: Searches scientific literature using Tavily and specialized domains
- **Processor Agent**: Extracts content and vectorizes articles using Tavily Extract + OpenAI
- **Retriever Agent**: Performs semantic search on the vector store
- **Analyst Agent**: Performs statistical analyses and creates visualizations
- **Writer Agent**: Generates structured HTML reports with citations
- **Reviewer Agent**: Reviews report quality and suggests improvements
- **Editor Agent**: Integrates final report with analyses

## Key Features

- **AI-First Approach**: LLMs handle all complex tasks (extraction, referencing, querying)
- **Autonomous Multi-Agent System**: Each agent makes independent decisions using ReAct pattern
- **Comprehensive Meta-Analysis**: From PICO definition to forest plots
- **Quality Control**: Iterative refinement with quality thresholds
- **Persistent Memory**: Long-term and short-term memory using LangGraph stores and checkpointers

## Installation

### Option 1: Install from PyPI (when available)
```bash
pip install metanalyst-agent
```

### Option 2: Install from source
```bash
# Clone the repository
git clone https://github.com/nobrega-medtech/metanalyst-agent.git
cd metanalyst-agent

# Install in development mode
pip install -e .

# Or install with extras
pip install -e ".[dev,production]"
```

### Option 3: Using Docker (when available)
```bash
docker pull nobregamedtech/metanalyst-agent:latest
docker run -it --env-file .env nobregamedtech/metanalyst-agent
```

## Configuration

Set up your environment variables:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your API keys
export OPENAI_API_KEY="your_openai_key_here"
export TAVILY_API_KEY="your_tavily_key_here"
```

## Quick Start

### Basic Usage

```python
import asyncio
from metanalyst_agent import MetanalystAgent

async def main():
    # Initialize the agent
    agent = MetanalystAgent()
    
    # Run a meta-analysis
    result = await agent.run(
        query="Meta-analysis on mindfulness vs CBT for anxiety treatment in adults",
        max_articles=20,
        quality_threshold=0.8,
        max_time_minutes=15
    )
    
    if result["success"]:
        print("✅ Meta-analysis completed!")
        print(f"📊 Articles processed: {result['execution_summary']['total_articles_processed']}")
        print(f"📈 Quality score: {result['quality_assessment']['overall_quality']:.2f}")
        
        # Access results
        pico = result["pico_framework"]
        stats = result["statistical_analysis"]
        report = result["final_report"]
        
    else:
        print(f"❌ Error: {result['error']}")

# Run the async function
asyncio.run(main())
```

### Command Line Interface

```bash
# Run a meta-analysis from command line
metanalyst-agent analyze "mindfulness vs CBT for anxiety" --max-articles 20 --timeout 15

# Check system status
metanalyst-agent status

# View help
metanalyst-agent --help
```

### Advanced Configuration

```python
from metanalyst_agent import MetanalystAgent
from metanalyst_agent.config import Settings

# Custom configuration
settings = Settings(
    openai_model="gpt-4o",
    default_max_articles=100,
    default_quality_threshold=0.9,
    chunk_size=1200,
    chunk_overlap=150
)

agent = MetanalystAgent(settings)
```

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
DATABASE_URL=postgresql://user:pass@localhost:5432/metanalysis
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.