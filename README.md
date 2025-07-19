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

```bash
# Clone the repository
git clone https://github.com/nobrega-medtech/metanalyst-agent.git
cd metanalyst-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```python
from metanalyst_agent import MetanalystAgent

# Initialize the agent
agent = MetanalystAgent()

# Run a meta-analysis
result = agent.run(
    query="Meta-analysis on mindfulness vs CBT for anxiety treatment in adults",
    max_articles=50,
    quality_threshold=0.85
)

print(result.final_report)
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