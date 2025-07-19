"""Main multi-agent graph for the Metanalyst-Agent system"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ..state.iteration_state import IterationState
from ..state.meta_analysis_state import create_initial_state
from ..agents.supervisor_agent import create_supervisor_agent
from ..agents.researcher_agent import create_researcher_agent
from ..agents.processor_agent import create_processor_agent
from ..agents.analyst_agent import create_analyst_agent
from ..config.settings import Settings


class MetaAnalysisGraph:
    """Main graph class for the Metanalyst-Agent system"""
    
    def __init__(self, settings: Settings):
        """
        Initialize the MetaAnalysisGraph
        
        Args:
            settings: Configuration settings for the system
        """
        self.settings = settings
        self.graph = None
        self.checkpointer = MemorySaver()  # In-memory for development
        self.store = InMemoryStore()  # In-memory for development
        self._build_graph()
    
    def _build_graph(self):
        """Build the multi-agent graph with all specialized agents"""
        
        # Create all agents
        agents = self._create_agents()
        
        # Build the graph
        builder = StateGraph(IterationState)
        
        # Add supervisor agent (central hub)
        builder.add_node("supervisor", agents["supervisor"])
        
        # Add specialized agents
        builder.add_node("researcher", agents["researcher"])
        builder.add_node("processor", agents["processor"])
        builder.add_node("analyst", agents["analyst"])
        # Note: writer, reviewer, editor agents would be added here when implemented
        
        # Define the workflow edges
        builder.add_edge(START, "supervisor")
        
        # All agents return to supervisor after execution (hub-and-spoke)
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("processor", "supervisor")
        builder.add_edge("analyst", "supervisor")
        
        # Compile the graph
        self.graph = builder.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create all specialized agents"""
        
        # Convert settings to dict for agent creation
        settings_dict = {
            "openai_api_key": self.settings.openai_api_key,
            "openai_model": self.settings.openai_model,
            "tavily_api_key": self.settings.tavily_api_key
        }
        
        return {
            "supervisor": create_supervisor_agent(settings_dict),
            "researcher": create_researcher_agent(settings_dict),
            "processor": create_processor_agent(settings_dict),
            "analyst": create_analyst_agent(settings_dict)
        }
    
    async def execute(
        self,
        query: str,
        max_articles: int = 50,
        quality_threshold: float = 0.8,
        max_time_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a complete meta-analysis
        
        Args:
            query: The research question for meta-analysis
            max_articles: Maximum number of articles to process
            quality_threshold: Quality threshold for completion
            max_time_minutes: Maximum execution time in minutes
            
        Returns:
            Complete meta-analysis results
        """
        
        # Create initial state
        initial_state = create_initial_state(
            research_question=query,
            config={
                "max_articles": max_articles,
                "quality_threshold": quality_threshold,
                "max_time_minutes": max_time_minutes
            },
            max_iterations=10,
            quality_threshold=quality_threshold
        )
        
        # Generate unique thread ID
        thread_id = f"meta_analysis_{uuid.uuid4()}"
        
        # Configuration for execution
        config = {
            "recursion_limit": 100,  # Allow for complex workflows
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "metanalysis"
            }
        }
        
        try:
            # Execute with timeout
            async with asyncio.timeout(max_time_minutes * 60):
                final_state = await self._execute_with_monitoring(
                    initial_state, 
                    config,
                    max_time_minutes
                )
                
            return self._format_results(final_state)
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Meta-analysis timed out after {max_time_minutes} minutes",
                "partial_results": await self._recover_partial_results(thread_id)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "thread_id": thread_id
            }
    
    async def _execute_with_monitoring(
        self,
        initial_state: IterationState,
        config: Dict[str, Any],
        max_time_minutes: int
    ) -> IterationState:
        """Execute the graph with progress monitoring"""
        
        start_time = datetime.now()
        final_state = None
        
        # Stream execution for monitoring
        async for state in self.graph.astream(
            initial_state,
            config,
            stream_mode="values"
        ):
            final_state = state
            
            # Monitor progress
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            remaining = max_time_minutes - elapsed
            
            # Update state with time tracking
            if "execution_times" not in final_state:
                final_state["execution_times"] = {}
            
            final_state["execution_times"]["elapsed_minutes"] = elapsed
            final_state["execution_times"]["remaining_minutes"] = remaining
            
            # Check termination conditions
            if self._should_terminate(final_state, remaining):
                break
                
            # Log progress
            self._log_progress(final_state, elapsed)
        
        return final_state or initial_state
    
    def _should_terminate(self, state: IterationState, remaining_minutes: float) -> bool:
        """Check if execution should terminate"""
        
        # Time-based termination
        if remaining_minutes <= 1:
            return True
        
        # Quality-based termination
        if state.get("quality_satisfied", False):
            return True
        
        # Force stop flag
        if state.get("force_stop", False):
            return True
        
        # Completion flag
        if state.get("current_phase") == "completed":
            return True
        
        return False
    
    def _log_progress(self, state: IterationState, elapsed_minutes: float):
        """Log execution progress"""
        
        phase = state.get("current_phase", "unknown")
        agent = state.get("current_agent", "unknown")
        articles = state.get("total_articles_processed", 0)
        
        print(f"[{elapsed_minutes:.1f}min] Phase: {phase}, Agent: {agent}, Articles: {articles}")
    
    def _format_results(self, final_state: IterationState) -> Dict[str, Any]:
        """Format final results for return"""
        
        return {
            "success": True,
            "meta_analysis_id": final_state.get("meta_analysis_id"),
            "thread_id": final_state.get("thread_id"),
            "research_question": final_state.get("research_question"),
            "current_phase": final_state.get("current_phase"),
            "execution_summary": {
                "total_articles_processed": final_state.get("total_articles_processed", 0),
                "total_articles_failed": final_state.get("total_articles_failed", 0),
                "execution_times": final_state.get("execution_times", {}),
                "quality_scores": final_state.get("quality_scores", {}),
                "global_iterations": final_state.get("global_iterations", 0)
            },
            "pico_framework": final_state.get("pico", {}),
            "search_results": {
                "queries_used": final_state.get("search_queries", []),
                "candidate_urls": final_state.get("candidate_urls", []),
                "processed_articles": final_state.get("processed_articles", [])
            },
            "statistical_analysis": final_state.get("meta_analysis_results", {}),
            "forest_plots": final_state.get("forest_plots", []),
            "final_report": final_state.get("final_report"),
            "citations": final_state.get("citations", []),
            "quality_assessment": {
                "overall_quality": final_state.get("quality_scores", {}).get("overall", 0),
                "component_scores": final_state.get("quality_scores", {}),
                "quality_satisfied": final_state.get("quality_satisfied", False)
            },
            "messages": [
                {
                    "role": msg.type if hasattr(msg, 'type') else "system",
                    "content": msg.content if hasattr(msg, 'content') else str(msg),
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in final_state.get("messages", [])[-10:]  # Last 10 messages
            ],
            "completed_at": datetime.now().isoformat()
        }
    
    async def _recover_partial_results(self, thread_id: str) -> Dict[str, Any]:
        """Recover partial results in case of timeout or error"""
        
        try:
            # Get latest state from checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            checkpoints = list(self.checkpointer.list(config))
            
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                partial_state = latest_checkpoint.checkpoint.get("channel_values", {})
                return self._format_results(partial_state)
            
        except Exception as e:
            return {"error": f"Could not recover partial results: {str(e)}"}
        
        return {"error": "No partial results available"}
    
    def get_execution_status(self, thread_id: str) -> Dict[str, Any]:
        """Get current execution status for a thread"""
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoints = list(self.checkpointer.list(config))
            
            if checkpoints:
                latest = checkpoints[-1]
                state = latest.checkpoint.get("channel_values", {})
                
                return {
                    "thread_id": thread_id,
                    "current_phase": state.get("current_phase", "unknown"),
                    "current_agent": state.get("current_agent", "unknown"),
                    "progress": {
                        "articles_processed": state.get("total_articles_processed", 0),
                        "global_iterations": state.get("global_iterations", 0),
                        "quality_scores": state.get("quality_scores", {})
                    },
                    "last_updated": latest.checkpoint.get("ts", "unknown")
                }
            else:
                return {"error": "Thread not found"}
                
        except Exception as e:
            return {"error": str(e)}