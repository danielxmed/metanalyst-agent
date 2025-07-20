"""
Main module for the Metanalyst-Agent system.
Integrates all components and provides the primary interface for running meta-analyses.
"""

import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, AsyncGenerator, List, Literal
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from .config.settings import settings
from .state.meta_analysis_state import MetaAnalysisState, create_initial_state

logger = logging.getLogger(__name__)


class MetanalystAgent:
    """
    Main class for the Metanalyst-Agent system.
    
    Provides a high-level interface for running automated meta-analyses
    using a multi-agent system with LangGraph orchestration.
    """
    
    def __init__(
        self,
        use_persistent_storage: bool = True,  # Changed to True by default
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
        self.use_persistent_storage = use_persistent_storage
        self._postgres_checkpointer_cm = None
        self._postgres_store_cm = None
        
        if use_persistent_storage:
            # Store the context managers for proper cleanup
            self._postgres_checkpointer_cm = PostgresSaver.from_conn_string(self.database_url)
            self._postgres_store_cm = PostgresStore.from_conn_string(self.database_url)
            
            # Enter the context managers
            self.checkpointer = self._postgres_checkpointer_cm.__enter__()
            self.store = self._postgres_store_cm.__enter__()
        else:
            self.checkpointer = MemorySaver()
            self.store = InMemoryStore()
        
        # Initialize agents (lazy loading)
        self._orchestrator_agent = None
        self._research_agent = None
        self._processor_agent = None
        
        # Build the main graph
        self.graph = self._build_graph()
        
        print("🔬 Metanalyst-Agent initialized successfully!")
        storage_type = "PostgreSQL" if use_persistent_storage else "In-Memory"
        print(f"📊 Storage: {storage_type}")
        print(f"🧠 Model: {settings.openai_model}")
    
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
    
    def _orchestrator_node(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """
        Orchestrator node wrapper that handles state updates.
        """
        # Execute orchestrator agent (it's a function that takes state directly)
        result = self.orchestrator_agent(state)
        
        # Update state with new messages
        new_state = dict(state)
        new_state["messages"] = result.get("messages", state.get("messages", []))
        
        # Update current agent tracking
        new_state["current_agent"] = "orchestrator"
        
        # Increment global iterations
        new_state["global_iterations"] = state.get("global_iterations", 0) + 1
        
        return new_state
    
    def _research_node(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Research agent node wrapper."""
        # Execute research agent (it's a function that takes state directly)
        result = self.research_agent(state)
        
        new_state = dict(state)
        new_state["messages"] = result.get("messages", state.get("messages", []))
        new_state["current_agent"] = "researcher"
        
        # Extract search results from messages
        messages = result.get("messages", [])
        candidate_urls = list(new_state.get("candidate_urls", []))  # Start with existing URLs
        
        # Process all messages looking for tool results
        for msg in messages:
            # Check if message has tool_calls (AI message calling tools)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == "search_literature":
                        # Tool was called, results will be in next message
                        continue
            
            # Check if this is a tool result message
            if hasattr(msg, 'name') and msg.name == "search_literature":
                # This is the actual tool result
                try:
                    # Tool results are typically in the content field as JSON
                    if isinstance(msg.content, str):
                        # Try to parse as JSON
                        import json
                        try:
                            articles = json.loads(msg.content)
                        except:
                            # Not JSON, skip
                            continue
                    elif isinstance(msg.content, list):
                        articles = msg.content
                    else:
                        continue
                    
                    # Add articles to candidate URLs
                    for article in articles:
                        if isinstance(article, dict) and article.get("url"):
                            # Check if URL already exists
                            existing_urls = {item["url"] for item in candidate_urls}
                            if article["url"] not in existing_urls:
                                candidate_urls.append({
                                    "url": article["url"],
                                    "title": article.get("title", ""),
                                    "relevance_score": article.get("score", 0.0),
                                    "snippet": article.get("snippet", ""),
                                    "source_domain": article.get("source_domain", "")
                                })
                    
                except Exception as e:
                    logger.error(f"Error processing search results: {e}")
        
        # Update state with new URLs
        new_state["candidate_urls"] = candidate_urls
        
        # Log progress
        if len(candidate_urls) > len(state.get("candidate_urls", [])):
            new_articles = len(candidate_urls) - len(state.get("candidate_urls", []))
            logger.info(f"Added {new_articles} new articles. Total: {len(candidate_urls)}")
        else:
            logger.warning(f"No new articles found. Total remains: {len(candidate_urls)}")
            # Debug: check if we processed any search messages
            search_messages = [msg for msg in messages if hasattr(msg, 'name') and msg.name == "search_literature"]
            logger.warning(f"Search messages found: {len(search_messages)}")
            if search_messages:
                logger.warning(f"First search message content type: {type(search_messages[0].content)}")
                logger.warning(f"First search message content length: {len(str(search_messages[0].content))}")
        
        # Update phase if needed
        if state.get("current_phase") == "pico_definition":
            new_state["current_phase"] = "search"
        
        return new_state
    
    def _processor_node(self, state: MetaAnalysisState) -> MetaAnalysisState:
        """Processor agent node wrapper."""
        # Execute processor agent (it's a function that takes state directly)
        result = self.processor_agent(state)
        
        new_state = dict(state)
        new_state["messages"] = result.get("messages", state.get("messages", []))
        new_state["current_agent"] = "processor"
        
        # Extract processed articles from messages
        messages = result.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    if tool_call["name"] == "extract_article_content" and "result" in tool_call:
                        # Add to processed articles
                        processed = new_state.get("processed_articles", [])
                        processed.append(tool_call["result"])
                        new_state["processed_articles"] = processed
        
        # Update phase if needed
        if state.get("current_phase") == "search":
            new_state["current_phase"] = "extraction"
        
        return new_state
    
    def _route_orchestrator(self, state: MetaAnalysisState) -> Literal["researcher", "processor", "retriever", "analyst", "writer", "reviewer", "editor", END]:
        """
        Route decisions from orchestrator based on messages and state.
        """
        # Get all messages from the current iteration
        messages = state.get("messages", [])
        if not messages:
            return END
        
        # Anti-loop protection: Check for repeated processor failures
        processor_failures = 0
        processor_iterations = 0
        current_agent = state.get("current_agent", "")
        
        # Count recent processor iterations and failures
        for msg in reversed(messages[-10:]):  # Check last 10 messages
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
            
            if "processor" in content.lower():
                processor_iterations += 1
                if any(failure_term in content.lower() for failure_term in [
                    "error extracting content", "failed", "intervention", "repeated failures"
                ]):
                    processor_failures += 1
        
        # If too many processor failures or iterations, try alternative route
        if processor_failures >= 3 or processor_iterations >= 5:
            logger.warning(f"Detected processor loop: {processor_failures} failures, {processor_iterations} iterations")
            
            # If we have some URLs, try a different approach
            articles_found = len(state.get("candidate_urls", []))
            if articles_found > 0:
                logger.info("Breaking processor loop - requesting more articles from researcher")
                return "researcher"  # Get more/different articles
            else:
                logger.info("Breaking processor loop - ending execution due to persistent failures")
                return END  # End if no articles to work with
        
        # Check last few messages for tool calls and results
        for msg in reversed(messages[-5:]):  # Check last 5 messages
            content = str(msg.content) if hasattr(msg, 'content') else str(msg)
            
            # Check for explicit completion signals
            if any(phrase in content.lower() for phrase in ["signal_completion", "meta-analysis completed", "analysis complete"]):
                return END
            
            # Check for tool call results (our simplified handoff format)
            if "transfer_to_" in content:
                if "transfer_to_researcher" in content:
                    return "researcher"
                elif "transfer_to_processor" in content:
                    # Additional check: don't return to processor if already failing
                    if processor_failures < 2:
                        return "processor"
                    else:
                        logger.warning("Preventing return to failing processor - trying researcher instead")
                        return "researcher"
                elif "transfer_to_retriever" in content:
                    return "retriever"
                elif "transfer_to_analyst" in content:
                    return "analyst"
                elif "transfer_to_writer" in content:
                    return "writer"
                elif "transfer_to_reviewer" in content:
                    return "reviewer"
                elif "transfer_to_editor" in content:
                    return "editor"
            
            # Check if this is a tool message with our format
            if hasattr(msg, 'name') and msg.name and msg.name.startswith('transfer_to_'):
                agent_name = msg.name.replace('transfer_to_', '')
                if agent_name in ['researcher', 'processor', 'retriever', 'analyst', 'writer', 'reviewer', 'editor']:
                    # Additional protection for processor loops
                    if agent_name == 'processor' and processor_failures >= 2:
                        logger.warning("Preventing return to failing processor via tool message")
                        return "researcher"
                    return agent_name
        
        # Intelligent phase-based routing - continue workflow instead of ending
        phase = state.get("current_phase", "pico_definition")
        articles_found = len(state.get("candidate_urls", []))
        articles_processed = len(state.get("processed_articles", []))
        quality_scores = state.get("quality_scores", {})
        
        # Phase progression logic
        if phase in ["pico_definition", "search"]:
            if articles_found < 5:  # Need minimum articles
                return "researcher"
            else:
                return "processor"  # Start processing found articles
                
        elif phase == "extraction":
            # Move to analysis if we have enough processed articles
            if articles_processed >= 3:
                return "analyst"  # Start analysis phase
            # Continue processing if more articles to process
            elif articles_found > articles_processed and articles_processed < 20:
                return "processor"
            # Need more articles if processing failed
            else:
                return "researcher"
                
        elif phase == "analysis":
            # Check if analysis is complete
            if state.get("statistical_analysis") and quality_scores.get("analyst", 0) >= 0.7:
                return "writer"  # Move to writing
            elif articles_processed >= 3:
                return "analyst"  # Continue/retry analysis
            else:
                return "researcher"  # Need more data
                
        elif phase == "writing":
            if state.get("draft_report"):
                return "reviewer"  # Review the draft
            else:
                return "writer"  # Continue writing
                
        elif phase == "review":
            if quality_scores.get("reviewer", 0) >= 0.8:
                return "editor"  # Final editing
            elif state.get("review_feedback"):
                return "writer"  # Address feedback
            else:
                return "reviewer"  # Continue review
                
        elif phase == "editing":
            if state.get("final_report"):
                return END  # Actually complete
            else:
                return "editor"  # Finish editing
        
        # Default: continue with researcher to ensure progress
        return "researcher"
        
    def _build_graph(self) -> StateGraph:
        """
        Build the main LangGraph workflow for the meta-analysis process.
        
        Returns:
            Compiled LangGraph workflow
        """
        
        # Create the state graph
        builder = StateGraph(MetaAnalysisState)
        
        # Add agent nodes with wrappers
        builder.add_node("orchestrator", self._orchestrator_node)
        builder.add_node("researcher", self._research_node)
        builder.add_node("processor", self._processor_node)
        
        # Define the workflow
        builder.add_edge(START, "orchestrator")
        
        # Add conditional routing from orchestrator
        builder.add_conditional_edges(
            "orchestrator",
            self._route_orchestrator,
            {
                "researcher": "researcher",
                "processor": "processor",
                "retriever": "researcher",  # Temporarily route to researcher until retriever is implemented
                "analyst": "researcher",    # Temporarily route to researcher until analyst is implemented  
                "writer": "researcher",     # Temporarily route to researcher until writer is implemented
                "reviewer": "researcher",   # Temporarily route to researcher until reviewer is implemented
                "editor": "researcher",     # Temporarily route to researcher until editor is implemented
                END: END
            }
        )
        
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
        
        # Import here to avoid circular imports
        from .agents.orchestrator_agent import generate_pico_from_query
        
        # Generate PICO framework from query
        pico = generate_pico_from_query(query)
        print(f"📋 PICO Framework:")
        print(f"   Population: {pico['P']}")
        print(f"   Intervention: {pico['I']}")
        print(f"   Comparison: {pico['C']}")
        print(f"   Outcome: {pico['O']}")
        
        # Create initial state
        meta_analysis_id = str(uuid.uuid4())
        thread_id = f"thread_{meta_analysis_id[:8]}"
        
        config = {
            "max_articles": max_articles,
            "quality_threshold": quality_threshold,
            "quality_thresholds": {
                "researcher": quality_threshold,
                "processor": quality_threshold,
                "analyst": quality_threshold,
                "writer": quality_threshold,
                "reviewer": quality_threshold * 1.1,  # Higher standard for review
            },
            **kwargs
        }
        
        initial_state = create_initial_state(
            research_question=query,
            meta_analysis_id=meta_analysis_id,
            thread_id=thread_id,
            config=config
        )
        
        # Update with generated PICO
        initial_state.update({
            "pico": pico,
            "research_question": query,
            "current_phase": "search",  # Skip PICO definition since we generated it
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=f"I'll help you conduct a meta-analysis. I've extracted the PICO framework from your request:\n\n"
                                f"Population: {pico['P']}\n"
                                f"Intervention: {pico['I']}\n"
                                f"Comparison: {pico['C']}\n"
                                f"Outcome: {pico['O']}\n\n"
                                f"Let me start by searching for relevant literature.")
            ]
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
                    if state_update.get("messages"):
                        last_message = state_update["messages"][-1]
                        agent = state_update.get("current_agent", "system")
                        
                        # Extract clean message content
                        content = last_message.content
                        if isinstance(content, str):
                            # Remove tool call artifacts
                            if "transfer_to_" in content:
                                # Extract just the meaningful part
                                parts = content.split('\n')
                                for part in parts:
                                    if part and not "transfer_to_" in part:
                                        print(f"[{agent}] {part[:150]}...")
                                        break
                            else:
                                print(f"[{agent}] {content[:150]}...")
                    
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
            if self.debug:
                import traceback
                traceback.print_exc()
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
        
        # Import here to avoid circular imports
        from .agents.orchestrator_agent import generate_pico_from_query
        
        # Generate PICO and create initial state
        pico = generate_pico_from_query(query)
        
        meta_analysis_id = str(uuid.uuid4())
        thread_id = f"thread_{meta_analysis_id[:8]}"
        
        config = {
            "max_articles": max_articles,
            "quality_threshold": quality_threshold,
            "quality_thresholds": {
                "researcher": quality_threshold,
                "processor": quality_threshold,
                "analyst": quality_threshold,
                "writer": quality_threshold,
                "reviewer": quality_threshold * 1.1,
            },
            **kwargs
        }
        
        initial_state = create_initial_state(
            research_question=query,
            meta_analysis_id=meta_analysis_id,
            thread_id=thread_id,
            config=config
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
        # Cleanup PostgreSQL context managers if needed
        if self.use_persistent_storage:
            try:
                if self._postgres_store_cm:
                    self._postgres_store_cm.__exit__(exc_type, exc_val, exc_tb)
                if self._postgres_checkpointer_cm:
                    self._postgres_checkpointer_cm.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                print(f"Warning: Error closing PostgreSQL connections: {e}")
        
        # For other storage types
        else:
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
    use_persistent_storage: bool = False,  # Default to in-memory for compatibility
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
    
    with MetanalystAgent(use_persistent_storage=use_persistent_storage, debug=debug) as agent:
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
