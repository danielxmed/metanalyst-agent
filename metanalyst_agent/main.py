"""
Main MetanalystAgent Interface

This module provides the main interface for the Metanalyst-Agent system,
integrating all components into a cohesive multi-agent meta-analysis platform.
"""

import os
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config.settings import settings
from .state.meta_analysis_state import MetaAnalysisState, MetaAnalysisResult, create_initial_state
from .graph.multi_agent_graph import build_meta_analysis_graph
from .utils.logging_setup import setup_logging
from .utils.database_setup import ensure_databases_ready


class MetanalystAgent:
    """
    Main interface for the Metanalyst-Agent system
    
    This class provides a high-level interface for running automated meta-analyses
    using the multi-agent system architecture.
    """
    
    def __init__(
        self,
        config_overrides: Optional[Dict[str, Any]] = None,
        use_memory_backend: bool = False
    ):
        """
        Initialize the Metanalyst-Agent system
        
        Args:
            config_overrides: Optional configuration overrides
            use_memory_backend: Whether to use in-memory backends (for testing)
        """
        
        # Setup configuration
        self.config = config_overrides or {}
        self.use_memory_backend = use_memory_backend
        
        # Setup logging
        self.logger = setup_logging(settings.log_level, settings.log_file)
        
        # Initialize persistence backends
        self.checkpointer = None
        self.store = None
        self.graph = None
        
        # Create necessary directories
        settings.create_directories()
        
        self.logger.info("Metanalyst-Agent initialized successfully")
    
    async def initialize_backends(self):
        """Initialize database backends and graph"""
        
        try:
            if self.use_memory_backend:
                # Use in-memory backends for development/testing
                self.checkpointer = MemorySaver()
                self.store = InMemoryStore()
                self.logger.info("Using in-memory backends")
            else:
                # Ensure databases are ready
                await ensure_databases_ready(settings.get_database_configs())
                
                # Initialize PostgreSQL backends
                self.checkpointer = PostgresSaver.from_conn_string(settings.database_url)
                self.store = PostgresStore.from_conn_string(settings.database_url)
                self.logger.info("Using PostgreSQL backends")
            
            # Build the multi-agent graph
            self.graph = build_meta_analysis_graph(
                checkpointer=self.checkpointer,
                store=self.store,
                settings=settings
            )
            
            self.logger.info("Multi-agent graph built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backends: {str(e)}")
            raise
    
    async def run(
        self,
        query: str,
        max_articles: Optional[int] = None,
        quality_threshold: Optional[float] = None,
        max_time_minutes: int = 30,
        recursion_limit: Optional[int] = None,
        thread_id: Optional[str] = None
    ) -> MetaAnalysisResult:
        """
        Run a complete automated meta-analysis
        
        Args:
            query: The research question for meta-analysis
            max_articles: Maximum number of articles to process
            quality_threshold: Minimum quality threshold for completion
            max_time_minutes: Maximum execution time in minutes
            recursion_limit: Maximum recursion limit for the graph
            thread_id: Optional thread ID for conversation persistence
            
        Returns:
            Complete meta-analysis results
        """
        
        # Ensure backends are initialized
        if not self.graph:
            await self.initialize_backends()
        
        # Generate unique identifiers
        meta_analysis_id = str(uuid.uuid4())
        if not thread_id:
            thread_id = f"meta_analysis_{meta_analysis_id}"
        
        self.logger.info(f"Starting meta-analysis: {meta_analysis_id}")
        self.logger.info(f"Research query: {query}")
        
        # Create initial state
        initial_state = create_initial_state(
            research_question=query,
            meta_analysis_id=meta_analysis_id,
            thread_id=thread_id,
            config={
                **self.config,
                "max_articles": max_articles or settings.default_max_articles,
                "quality_threshold": quality_threshold or settings.default_quality_threshold,
                "agent_limits": settings.agent_limits,
                "quality_thresholds": settings.quality_thresholds,
                "max_global_iterations": settings.default_max_iterations,
                "recursion_limit": recursion_limit or settings.default_recursion_limit
            }
        )
        
        # Add initial user message
        initial_state["messages"] = [HumanMessage(content=query)]
        
        # Configuration for graph execution
        graph_config = {
            "recursion_limit": recursion_limit or settings.default_recursion_limit,
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "meta_analysis",
                "user_id": "system"
            }
        }
        
        try:
            # Execute with timeout
            final_state = await asyncio.wait_for(
                self._execute_meta_analysis(initial_state, graph_config),
                timeout=max_time_minutes * 60
            )
            
            # Convert to result object
            result = self._create_result_from_state(final_state)
            
            self.logger.info(f"Meta-analysis completed: {meta_analysis_id}")
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Meta-analysis timed out after {max_time_minutes} minutes")
            
            # Try to recover partial results
            partial_state = await self._recover_partial_results(thread_id, graph_config)
            if partial_state:
                result = self._create_result_from_state(partial_state, partial=True)
                result.execution_log.append({
                    "event": "timeout",
                    "message": f"Execution timed out after {max_time_minutes} minutes",
                    "timestamp": datetime.now().isoformat()
                })
                return result
            
            raise TimeoutError(f"Meta-analysis timed out after {max_time_minutes} minutes")
            
        except Exception as e:
            self.logger.error(f"Meta-analysis failed: {str(e)}")
            raise
    
    async def _execute_meta_analysis(
        self, 
        initial_state: MetaAnalysisState, 
        config: Dict[str, Any]
    ) -> MetaAnalysisState:
        """Execute the meta-analysis using the multi-agent graph"""
        
        final_state = None
        
        # Stream execution for monitoring
        async for chunk in self.graph.astream(
            initial_state,
            config,
            stream_mode="values"
        ):
            final_state = chunk
            
            # Log progress
            if chunk.get("messages"):
                last_message = chunk["messages"][-1]
                current_agent = chunk.get("current_agent", "unknown")
                current_phase = chunk.get("current_phase", "unknown")
                
                self.logger.debug(
                    f"[{current_agent}|{current_phase}] {last_message.content[:100]}..."
                )
            
            # Check for completion conditions
            if chunk.get("current_phase") == "completed":
                self.logger.info("Meta-analysis completed successfully")
                break
            
            if chunk.get("force_stop") or chunk.get("emergency_stop"):
                self.logger.warning("Meta-analysis stopped due to emergency condition")
                break
            
            if chunk.get("human_intervention_requested"):
                self.logger.warning("Meta-analysis paused for human intervention")
                break
        
        return final_state or initial_state
    
    async def _recover_partial_results(
        self, 
        thread_id: str, 
        config: Dict[str, Any]
    ) -> Optional[MetaAnalysisState]:
        """Attempt to recover partial results from checkpoints"""
        
        try:
            # Get the latest checkpoint
            checkpoints = list(self.checkpointer.list(config))
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x.metadata.get("step", 0))
                return latest.checkpoint
        except Exception as e:
            self.logger.error(f"Failed to recover partial results: {str(e)}")
        
        return None
    
    def _create_result_from_state(
        self, 
        final_state: MetaAnalysisState, 
        partial: bool = False
    ) -> MetaAnalysisResult:
        """Create MetaAnalysisResult from final state"""
        
        # Calculate execution time
        start_time = final_state.get("created_at", datetime.now())
        end_time = final_state.get("updated_at", datetime.now())
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        execution_time = (end_time - start_time).total_seconds() / 60  # minutes
        
        # Extract statistical results
        statistical_analysis = final_state.get("statistical_analysis", {})
        
        # Create result object
        result = MetaAnalysisResult(
            meta_analysis_id=final_state["meta_analysis_id"],
            research_question=final_state["research_question"],
            pico=final_state.get("pico", {}),
            
            # Process summary
            total_articles_screened=final_state.get("total_articles_found", 0),
            total_articles_included=final_state.get("total_articles_included", 0),
            processing_time_minutes=execution_time,
            quality_score=self._calculate_overall_quality_score(final_state),
            
            # Results
            final_report=final_state.get("final_report", ""),
            statistical_analysis=statistical_analysis,
            forest_plots=[],  # Will be populated from analysis results
            citations=final_state.get("vancouver_citations", []),
            
            # Quality metrics
            heterogeneity_i2=statistical_analysis.get("heterogeneity", {}).get("I_squared", 0.0),
            overall_effect_size=statistical_analysis.get("random_effects", {}).get("pooled_effect", 0.0),
            confidence_interval=statistical_analysis.get("random_effects", {}).get("confidence_interval", [0.0, 0.0]),
            p_value=statistical_analysis.get("random_effects", {}).get("p_value", 1.0),
            
            # Metadata
            generated_at=datetime.now(),
            agent_performance=self._extract_agent_performance(final_state),
            execution_log=final_state.get("agent_logs", [])
        )
        
        # Add partial flag if applicable
        if partial:
            result.execution_log.append({
                "event": "partial_completion",
                "message": "Results are partial due to early termination",
                "timestamp": datetime.now().isoformat()
            })
        
        return result
    
    def _calculate_overall_quality_score(self, state: MetaAnalysisState) -> float:
        """Calculate overall quality score from state"""
        
        quality_scores = state.get("quality_scores", {})
        if not quality_scores:
            return 0.0
        
        # Weighted average based on agent importance
        weights = {
            "researcher": 0.2,
            "processor": 0.2,
            "retriever": 0.1,
            "analyst": 0.3,
            "writer": 0.1,
            "reviewer": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for agent, score in quality_scores.items():
            weight = weights.get(agent, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _extract_agent_performance(self, state: MetaAnalysisState) -> Dict[str, Dict[str, Any]]:
        """Extract agent performance metrics from state"""
        
        performance = {}
        
        # Extract from agent logs
        agent_logs = state.get("agent_logs", [])
        agent_iterations = state.get("agent_iterations", {})
        quality_scores = state.get("quality_scores", {})
        
        for agent_name in ["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor"]:
            agent_performance = {
                "iterations": agent_iterations.get(agent_name, 0),
                "quality_score": quality_scores.get(agent_name, 0.0),
                "execution_time": 0.0,
                "success_rate": 1.0,
                "errors": []
            }
            
            # Calculate metrics from logs
            agent_specific_logs = [log for log in agent_logs if log.get("agent") == agent_name]
            
            if agent_specific_logs:
                # Calculate execution time
                timestamps = [
                    datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
                    for log in agent_specific_logs
                    if log.get("timestamp")
                ]
                
                if len(timestamps) > 1:
                    agent_performance["execution_time"] = (max(timestamps) - min(timestamps)).total_seconds()
                
                # Count errors
                errors = [log for log in agent_specific_logs if log.get("type") == "error"]
                agent_performance["errors"] = [e.get("error", "Unknown error") for e in errors]
                
                # Calculate success rate
                total_actions = len(agent_specific_logs)
                failed_actions = len(errors)
                agent_performance["success_rate"] = (total_actions - failed_actions) / total_actions if total_actions > 0 else 1.0
            
            performance[agent_name] = agent_performance
        
        return performance
    
    async def get_analysis_status(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current status of an ongoing meta-analysis
        
        Args:
            thread_id: Thread ID of the meta-analysis
            
        Returns:
            Current status information
        """
        
        if not self.checkpointer:
            await self.initialize_backends()
        
        try:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "meta_analysis"
                }
            }
            
            # Get latest checkpoint
            checkpoints = list(self.checkpointer.list(config))
            if not checkpoints:
                return {"status": "not_found", "message": "No analysis found with this thread ID"}
            
            latest = max(checkpoints, key=lambda x: x.metadata.get("step", 0))
            state = latest.checkpoint
            
            return {
                "status": "found",
                "meta_analysis_id": state.get("meta_analysis_id"),
                "current_phase": state.get("current_phase"),
                "current_agent": state.get("current_agent"),
                "articles_processed": len(state.get("processed_articles", [])),
                "quality_scores": state.get("quality_scores", {}),
                "last_updated": state.get("updated_at"),
                "force_stop": state.get("force_stop", False),
                "emergency_stop": state.get("emergency_stop", False)
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def resume_analysis(
        self, 
        thread_id: str,
        max_time_minutes: int = 30
    ) -> MetaAnalysisResult:
        """
        Resume a paused or interrupted meta-analysis
        
        Args:
            thread_id: Thread ID of the analysis to resume
            max_time_minutes: Maximum execution time for resumption
            
        Returns:
            Completed meta-analysis results
        """
        
        if not self.graph:
            await self.initialize_backends()
        
        # Get current status
        status = await self.get_analysis_status(thread_id)
        if status["status"] != "found":
            raise ValueError(f"Cannot resume analysis: {status.get('message', 'Unknown error')}")
        
        self.logger.info(f"Resuming meta-analysis: {status['meta_analysis_id']}")
        
        # Configuration for resumption
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "meta_analysis"
            }
        }
        
        try:
            # Resume execution
            final_state = await asyncio.wait_for(
                self._resume_execution(config),
                timeout=max_time_minutes * 60
            )
            
            result = self._create_result_from_state(final_state)
            
            self.logger.info(f"Meta-analysis resumed and completed: {status['meta_analysis_id']}")
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Resumed meta-analysis timed out after {max_time_minutes} minutes")
            raise TimeoutError(f"Resumed execution timed out after {max_time_minutes} minutes")
    
    async def _resume_execution(self, config: Dict[str, Any]) -> MetaAnalysisState:
        """Resume execution from latest checkpoint"""
        
        final_state = None
        
        # Continue from latest checkpoint
        async for chunk in self.graph.astream(
            None,  # Will load from checkpoint
            config,
            stream_mode="values"
        ):
            final_state = chunk
            
            # Log progress
            if chunk.get("messages"):
                last_message = chunk["messages"][-1]
                current_agent = chunk.get("current_agent", "unknown")
                current_phase = chunk.get("current_phase", "unknown")
                
                self.logger.debug(
                    f"[RESUME|{current_agent}|{current_phase}] {last_message.content[:100]}..."
                )
            
            # Check for completion
            if chunk.get("current_phase") == "completed":
                break
            
            if chunk.get("force_stop") or chunk.get("emergency_stop"):
                break
        
        return final_state
    
    def close(self):
        """Close database connections and cleanup resources"""
        
        if self.checkpointer and hasattr(self.checkpointer, 'close'):
            self.checkpointer.close()
        
        if self.store and hasattr(self.store, 'close'):
            self.store.close()
        
        self.logger.info("Metanalyst-Agent closed successfully")


# Convenience function for simple usage
async def run_meta_analysis(
    query: str,
    max_articles: int = 50,
    quality_threshold: float = 0.8,
    max_time_minutes: int = 30,
    use_memory_backend: bool = False
) -> MetaAnalysisResult:
    """
    Convenience function to run a meta-analysis with default settings
    
    Args:
        query: Research question for meta-analysis
        max_articles: Maximum articles to process
        quality_threshold: Quality threshold for completion
        max_time_minutes: Maximum execution time
        use_memory_backend: Use in-memory storage (for testing)
        
    Returns:
        Complete meta-analysis results
    """
    
    agent = MetanalystAgent(use_memory_backend=use_memory_backend)
    
    try:
        result = await agent.run(
            query=query,
            max_articles=max_articles,
            quality_threshold=quality_threshold,
            max_time_minutes=max_time_minutes
        )
        return result
    finally:
        agent.close()


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        query = """
        Meta-analysis comparing the effectiveness of mindfulness-based interventions 
        versus cognitive behavioral therapy for treating anxiety disorders in adults.
        Include only randomized controlled trials from the last 10 years.
        """
        
        result = await run_meta_analysis(
            query=query,
            max_articles=30,
            quality_threshold=0.85,
            max_time_minutes=45,
            use_memory_backend=True  # For testing
        )
        
        print(f"Meta-analysis completed!")
        print(f"Studies included: {result.total_articles_included}")
        print(f"Overall effect size: {result.overall_effect_size:.3f}")
        print(f"Quality score: {result.quality_score:.2f}")
        print(f"Execution time: {result.processing_time_minutes:.1f} minutes")
    
    asyncio.run(main())