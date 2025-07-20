"""
Main module for the Metanalyst-Agent system.
Integrates all components and provides the primary interface for running meta-analyses.
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, AsyncGenerator, List
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config.settings import settings
from .state.meta_analysis_state import MetaAnalysisState, create_initial_state
# Lazy imports to avoid circular dependencies and missing function errors
# from .agents.orchestrator_agent import create_orchestrator_agent, generate_pico_from_query
# from .agents.research_agent import create_research_agent
# from .agents.processor_agent import create_processor_agent


class MetanalystAgent:
    """
    Main class for the Metanalyst-Agent system.
    
    Provides a high-level interface for running automated meta-analyses
    using a multi-agent system with LangGraph orchestration.
    """
    
    def __init__(
        self,
        use_persistent_storage: bool = False,
        database_url: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the Metanalyst-Agent system.
        
        Args:
            use_persistent_storage: Whether to use persistent storage (PostgreSQL)
            database_url: Database URL for persistent storage
            debug: Enable debug mode with additional logging
        """
        
        self.use_persistent_storage = use_persistent_storage
        self.database_url = database_url or settings.database_url
        self.debug = debug
        
        # Create directories
        settings.create_directories()
        
        # Initialize storage components
        if use_persistent_storage:
            self.checkpointer = PostgresSaver.from_conn_string(self.database_url)
            self.store = PostgresStore.from_conn_string(self.database_url)
        else:
            self.checkpointer = MemorySaver()
            self.store = InMemoryStore()
        
        # Initialize agents (lazy loading)
        self._orchestrator_agent = None
        self._research_agent = None
        self._processor_agent = None
        
        # Build the main graph
        self.graph = self._build_graph()
    
    def create_initial_state(
        self,
        research_question: str,
        config: Optional[Dict[str, Any]] = None
    ) -> MetaAnalysisState:
        """
        Create initial state for meta-analysis.
        
        Args:
            research_question: The research question for the meta-analysis
            config: Additional configuration parameters
        
        Returns:
            Initial state dictionary
        """
        return create_initial_state(
            research_question=research_question,
            meta_analysis_id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4()),
            config=config or {}
        )
    
    @property
    def orchestrator_agent(self):
        """Lazy loading for orchestrator agent"""
        if self._orchestrator_agent is None:
            from .agents.orchestrator_agent import create_orchestrator_agent
            self._orchestrator_agent = create_orchestrator_agent()
        return self._orchestrator_agent
    
    @property
    def research_agent(self):
        """Lazy loading for research agent"""
        if self._research_agent is None:
            from .agents.research_agent import create_research_agent
            self._research_agent = create_research_agent()
        return self._research_agent
    
    @property
    def processor_agent(self):
        """Lazy loading for processor agent"""
        if self._processor_agent is None:
            from .agents.processor_agent import create_processor_agent
            self._processor_agent = create_processor_agent()
        return self._processor_agent
        
        print("🔬 Metanalyst-Agent initialized successfully!")
        if debug:
            print(f"📊 Storage: {'Persistent' if use_persistent_storage else 'In-Memory'}")
            print(f"🧠 Model: {settings.openai_model}")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the main LangGraph workflow for the meta-analysis process.
        
        Returns:
            Compiled LangGraph workflow
        """
        
        # Create the state graph
        builder = StateGraph(MetaAnalysisState)
        
        # Add agent nodes
        builder.add_node("orchestrator", self.orchestrator_agent)
        builder.add_node("researcher", self.research_agent)
        builder.add_node("processor", self.processor_agent)
        
        # Define the workflow
        builder.add_edge(START, "orchestrator")
        
        # All agents return to orchestrator for coordination
        builder.add_edge("researcher", "orchestrator")
        builder.add_edge("processor", "orchestrator")
        
        # Compile the graph with storage
        graph = builder.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
        
        return graph
    
    def run(
        self,
        query: str,
        max_articles: int = 50,
        quality_threshold: float = 0.8,
        max_time_minutes: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a complete meta-analysis synchronously.
        
        Args:
            query: User's meta-analysis request
            max_articles: Maximum number of articles to process
            quality_threshold: Minimum quality threshold
            max_time_minutes: Maximum execution time in minutes
            **kwargs: Additional configuration parameters
        
        Returns:
            Meta-analysis results
        """
        
        return asyncio.run(self.arun(
            query=query,
            max_articles=max_articles,
            quality_threshold=quality_threshold,
            max_time_minutes=max_time_minutes,
            **kwargs
        ))
    
    async def arun(
        self,
        query: str,
        max_articles: int = 50,
        quality_threshold: float = 0.8,
        max_time_minutes: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a complete meta-analysis asynchronously.
        
        Args:
            query: User's meta-analysis request
            max_articles: Maximum number of articles to process
            quality_threshold: Minimum quality threshold
            max_time_minutes: Maximum execution time in minutes
            **kwargs: Additional configuration parameters
        
        Returns:
            Meta-analysis results
        """
        
        print(f"🚀 Starting meta-analysis: {query[:100]}...")
        
        # Generate PICO framework from query
        from .agents.orchestrator_agent import generate_pico_from_query
        pico = generate_pico_from_query(query)
        print(f"📋 PICO Framework:")
        print(f"   Population: {pico['P']}")
        print(f"   Intervention: {pico['I']}")
        print(f"   Comparison: {pico['C']}")
        print(f"   Outcome: {pico['O']}")
        
        # Create initial state
        initial_state = create_initial_state(
            research_question=query,
            meta_analysis_id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4()),
            config={
                "max_articles": max_articles,
                "quality_threshold": quality_threshold,
                **kwargs
            }
        )
        
        # Update with generated PICO
        initial_state.update({
            "pico": pico,
            "research_question": query,
            "current_phase": "search"  # Skip PICO definition since we generated it
        })
        
        # Configuration for execution
        config = {
            "recursion_limit": settings.default_recursion_limit,
            "configurable": {
                "thread_id": initial_state["thread_id"],
                "checkpoint_ns": "meta_analysis",
            }
        }
        
        # Execute with timeout
        try:
            async with asyncio.timeout(max_time_minutes * 60):
                final_state = None
                
                async for state_update in self.graph.astream(
                    initial_state,
                    config,
                    stream_mode="values"
                ):
                    final_state = state_update
                    
                    # Progress reporting
                    if self.debug and state_update.get("messages"):
                        last_message = state_update["messages"][-1]
                        agent = state_update.get("current_agent", "system")
                        print(f"[{agent}] {last_message.content[:150]}...")
                    
                    # Check for completion
                    if state_update.get("final_report") or state_update.get("force_stop"):
                        break
                
                print("✅ Meta-analysis completed successfully!")
                return self._format_results(final_state)
                
        except asyncio.TimeoutError:
            print(f"⏰ Meta-analysis timed out after {max_time_minutes} minutes")
            # Try to recover partial results
            return self._format_results(initial_state, partial=True)
        
        except Exception as e:
            print(f"❌ Meta-analysis failed: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def stream(
        self,
        query: str,
        max_articles: int = 50,
        quality_threshold: float = 0.8,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream meta-analysis progress in real-time.
        
        Args:
            query: User's meta-analysis request
            max_articles: Maximum number of articles to process
            quality_threshold: Minimum quality threshold
            **kwargs: Additional configuration parameters
        
        Yields:
            Real-time progress updates
        """
        
        # Generate PICO and create initial state
        from .agents.orchestrator_agent import generate_pico_from_query
        pico = generate_pico_from_query(query)
        initial_state = create_initial_state(
            research_question=query,
            meta_analysis_id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4()),
            config={
                "max_articles": max_articles,
                "quality_threshold": quality_threshold,
                **kwargs
            }
        )
        initial_state.update({
            "pico": pico,
            "research_question": query,
            "current_phase": "search"
        })
        
        # Configuration
        config = {
            "recursion_limit": settings.default_recursion_limit,
            "configurable": {
                "thread_id": initial_state["thread_id"],
                "checkpoint_ns": "meta_analysis",
            }
        }
        
        # Stream updates
        async for state_update in self.graph.astream(
            initial_state,
            config,
            stream_mode="values"
        ):
            # Format progress update
            progress = {
                "phase": state_update.get("current_phase", "unknown"),
                "agent": state_update.get("current_agent", "system"),
                "articles_found": len(state_update.get("candidate_urls", [])),
                "articles_processed": len(state_update.get("processed_articles", [])),
                "quality_scores": state_update.get("quality_scores", {}),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add latest message if available
            if state_update.get("messages"):
                progress["latest_message"] = state_update["messages"][-1].content
            
            # Check for completion
            if state_update.get("final_report"):
                progress["status"] = "completed"
                progress["final_report"] = state_update["final_report"]
            elif state_update.get("force_stop"):
                progress["status"] = "stopped"
            else:
                progress["status"] = "running"
            
            yield progress
            
            # Break on completion
            if progress["status"] in ["completed", "stopped"]:
                break
    
    def _format_results(self, state: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
        """
        Format the final results from the meta-analysis.
        
        Args:
            state: Final state from the graph execution
            partial: Whether these are partial results due to timeout
        
        Returns:
            Formatted results dictionary
        """
        
        results = {
            "status": "partial" if partial else "completed",
            "meta_analysis_id": state.get("meta_analysis_id"),
            "pico": state.get("pico", {}),
            "execution_summary": {
                "articles_found": len(state.get("candidate_urls", [])),
                "articles_processed": len(state.get("processed_articles", [])),
                "articles_failed": len(state.get("failed_urls", [])),
                "final_phase": state.get("current_phase"),
                "total_iterations": state.get("global_iterations", 0),
                "quality_scores": state.get("quality_scores", {}),
                "execution_time": state.get("execution_time", {}),
            },
            "statistical_analysis": state.get("statistical_analysis", {}),
            "forest_plots": state.get("forest_plots", []),
            "quality_assessments": state.get("quality_assessments", {}),
            "final_report": state.get("final_report"),
            "citations": state.get("citations", []),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
        }
        
        # Add error information if available
        if state.get("force_stop"):
            results["stop_reason"] = state.get("stop_reason")
        
        # Add agent logs for debugging
        if self.debug:
            results["agent_logs"] = state.get("agent_logs", [])
            results["iteration_timeline"] = state.get("iteration_timeline", [])
        
        return results
    
    def get_status(self, meta_analysis_id: str) -> Dict[str, Any]:
        """
        Get the current status of a running meta-analysis.
        
        Args:
            meta_analysis_id: ID of the meta-analysis
        
        Returns:
            Current status information
        """
        
        # This would query the checkpointer/store for current state
        # Implementation depends on storage backend
        return {"status": "not_implemented"}
    
    def list_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent meta-analyses.
        
        Args:
            limit: Maximum number of analyses to return
        
        Returns:
            List of meta-analysis summaries
        """
        
        # This would query the store for historical analyses
        # Implementation depends on storage backend
        return []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        if hasattr(self.checkpointer, 'close'):
            self.checkpointer.close()
        if hasattr(self.store, 'close'):
            self.store.close()


# Convenience function for quick meta-analyses
def run_meta_analysis(
    query: str,
    max_articles: int = 50,
    quality_threshold: float = 0.8,
    debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a meta-analysis with default settings.
    
    Args:
        query: Meta-analysis request
        max_articles: Maximum articles to process
        quality_threshold: Quality threshold
        debug: Enable debug mode
        **kwargs: Additional parameters
    
    Returns:
        Meta-analysis results
    """
    
    with MetanalystAgent(debug=debug) as agent:
        return agent.run(
            query=query,
            max_articles=max_articles,
            quality_threshold=quality_threshold,
            **kwargs
        )


# Example usage
if __name__ == "__main__":
    # Example meta-analysis
    query = """
    Systematic review and meta-analysis comparing the effectiveness of mindfulness-based 
    interventions versus cognitive behavioral therapy for reducing anxiety symptoms in adults
    with generalized anxiety disorder. Include forest plot analysis.
    """
    
    # Run with debug mode
    results = run_meta_analysis(
        query=query,
        max_articles=30,
        quality_threshold=0.8,
        debug=True
    )
    
    print("\n" + "="*80)
    print("META-ANALYSIS RESULTS")
    print("="*80)
    print(f"Status: {results.get('status')}")
    print(f"Articles processed: {results.get('execution_summary', {}).get('articles_processed', 0)}")
    
    if results.get('final_report'):
        print("\nFinal report generated successfully!")
    else:
        print("\nPartial results only - may need more time or manual intervention.")
    
    print("="*80)