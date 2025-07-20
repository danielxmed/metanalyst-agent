"""
Base agent functionality for all specialized agents in the metanalyst-agent system.
Provides common utilities for state access and message formatting.
"""

from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from ..config.settings import settings
from ..state.meta_analysis_state import MetaAnalysisState


def create_agent_with_state_context(
    name: str,
    system_prompt: str,
    tools: List[Any],
    model: Any = None
) -> Any:
    """
    Create an agent that automatically includes state context in its messages.
    
    Args:
        name: Agent name
        system_prompt: System prompt for the agent
        tools: List of tools available to the agent
        model: LLM model (uses default if not provided)
    
    Returns:
        Configured agent with state context handling
    """
    
    if model is None:
        model = ChatOpenAI(**settings.get_openai_config())
    
    # Create base agent
    base_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
    )
    
    # Wrapper that adds state context
    def agent_with_context(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke agent with state context included in messages.
        """
        
        # Build context message with current state
        context_parts = []
        
        # Current phase
        context_parts.append(f"CURRENT PHASE: {state.get('current_phase', 'unknown')}")
        
        # PICO if available
        pico = state.get("pico", {})
        if pico:
            context_parts.append(f"\nPICO FRAMEWORK:")
            context_parts.append(f"- Population: {pico.get('P', 'Not defined')}")
            context_parts.append(f"- Intervention: {pico.get('I', 'Not defined')}")
            context_parts.append(f"- Comparison: {pico.get('C', 'Not defined')}")
            context_parts.append(f"- Outcome: {pico.get('O', 'Not defined')}")
        
        # Research question
        if state.get("research_question"):
            context_parts.append(f"\nRESEARCH QUESTION: {state['research_question']}")
        
        # Current status
        context_parts.append(f"\nCURRENT STATUS:")
        context_parts.append(f"- Articles found: {len(state.get('candidate_urls', []))}")
        context_parts.append(f"- Articles processed: {len(state.get('processed_articles', []))}")
        context_parts.append(f"- Articles failed: {len(state.get('failed_urls', []))}")
        
        # Quality scores
        quality_scores = state.get("quality_scores", {})
        if quality_scores:
            context_parts.append(f"\nQUALITY SCORES:")
            for component, score in quality_scores.items():
                context_parts.append(f"- {component}: {score:.2f}")
        
        # Agent iterations
        agent_iterations = state.get("agent_iterations", {})
        if agent_iterations:
            context_parts.append(f"\nAGENT ITERATIONS:")
            for agent, count in agent_iterations.items():
                context_parts.append(f"- {agent}: {count}")
        
        # Build context message
        context_message = SystemMessage(
            content=f"CURRENT META-ANALYSIS STATE:\n\n" + "\n".join(context_parts)
        )
        
        # Get existing messages
        messages = state.get("messages", [])
        
        # Create enhanced input with context
        enhanced_input = {
            **state,
            "messages": [context_message] + messages
        }
        
        # Invoke base agent with enhanced input
        result = base_agent.invoke(enhanced_input)
        
        # Return result maintaining state structure
        return result
    
    # Set name attribute
    agent_with_context.__name__ = f"{name}_with_context"
    
    return agent_with_context


def extract_tool_results(messages: List[BaseMessage]) -> Dict[str, Any]:
    """
    Extract tool call results from agent messages.
    
    Args:
        messages: List of messages from agent
    
    Returns:
        Dictionary of extracted tool results
    """
    
    results = {
        "search_results": [],
        "extracted_articles": [],
        "analysis_results": {},
        "report_content": None,
        "review_feedback": [],
    }
    
    for message in messages:
        if hasattr(message, 'tool_calls'):
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name", "")
                
                # Extract based on tool type
                if tool_name == "search_literature":
                    if "results" in tool_call:
                        results["search_results"].extend(tool_call["results"])
                
                elif tool_name == "extract_article_content":
                    if "result" in tool_call:
                        results["extracted_articles"].append(tool_call["result"])
                
                elif tool_name in ["perform_meta_analysis", "create_forest_plot"]:
                    if "result" in tool_call:
                        results["analysis_results"][tool_name] = tool_call["result"]
                
                elif tool_name == "generate_report_section":
                    if "content" in tool_call:
                        results["report_content"] = tool_call["content"]
                
                elif tool_name == "review_quality":
                    if "feedback" in tool_call:
                        results["review_feedback"].append(tool_call["feedback"])
    
    return results


def format_agent_response(
    agent_name: str,
    action: str,
    details: Dict[str, Any],
    next_step: str = None
) -> AIMessage:
    """
    Format a standardized agent response message.
    
    Args:
        agent_name: Name of the agent
        action: Action taken by the agent
        details: Details about the action
        next_step: Suggested next step
    
    Returns:
        Formatted AIMessage
    """
    
    content_parts = [
        f"[{agent_name.upper()}] {action}",
        ""
    ]
    
    # Add details
    for key, value in details.items():
        if isinstance(value, list):
            content_parts.append(f"{key}: {len(value)} items")
        elif isinstance(value, dict):
            content_parts.append(f"{key}: {len(value)} entries")
        else:
            content_parts.append(f"{key}: {value}")
    
    # Add next step if provided
    if next_step:
        content_parts.append("")
        content_parts.append(f"Next step: {next_step}")
    
    return AIMessage(
        content="\n".join(content_parts),
        name=agent_name
    )
