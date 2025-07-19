"""
Iteration State Management for Metanalyst-Agent

This module provides specialized state management for controlling iterations
and preventing infinite loops in the multi-agent system.
"""

from typing import TypedDict, Dict, List, Any, Literal
from .meta_analysis_state import MetaAnalysisState


class IterationState(MetaAnalysisState):
    """
    Extended state with advanced iteration control capabilities
    
    This extends the base MetaAnalysisState with additional fields
    specifically for managing complex iteration patterns and preventing loops.
    """
    
    # Advanced iteration metrics
    iteration_timeline: List[Dict[str, Any]]  # Detailed iteration history
    average_iteration_time: float
    estimated_completion_time: float
    
    # Performance tracking
    agent_performance_history: Dict[str, List[Dict[str, Any]]]
    quality_improvement_trends: Dict[str, List[float]]
    
    # Dynamic limits (learned from performance)
    dynamic_agent_limits: Dict[str, int]
    adaptive_quality_thresholds: Dict[str, float]
    
    # Stagnation detection
    stagnation_counters: Dict[str, int]
    stagnation_thresholds: Dict[str, int]
    
    # Resource usage tracking
    memory_usage: Dict[str, float]
    processing_time_per_article: List[float]
    api_call_counts: Dict[str, int]
    
    # Emergency controls
    emergency_stop_triggers: List[str]
    graceful_shutdown_initiated: bool
    termination_reason: str


def create_iteration_tracking_record(
    agent_name: str,
    iteration_number: int,
    start_time: float,
    end_time: float,
    quality_score: float,
    improvement: float,
    action_taken: str,
    outcome: str
) -> Dict[str, Any]:
    """
    Create a standardized iteration tracking record
    
    Args:
        agent_name: Name of the agent performing the iteration
        iteration_number: Current iteration number for this agent
        start_time: Timestamp when iteration started
        end_time: Timestamp when iteration ended
        quality_score: Quality score achieved in this iteration
        improvement: Improvement from previous iteration
        action_taken: Description of action taken
        outcome: Outcome of the iteration
        
    Returns:
        Standardized iteration record dictionary
    """
    
    return {
        "agent": agent_name,
        "iteration": iteration_number,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "quality_score": quality_score,
        "improvement": improvement,
        "action_taken": action_taken,
        "outcome": outcome,
        "timestamp": end_time
    }


def update_agent_performance(
    state: IterationState,
    agent_name: str,
    performance_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update agent performance tracking
    
    Args:
        state: Current iteration state
        agent_name: Name of the agent
        performance_data: Performance metrics to record
        
    Returns:
        State update dictionary
    """
    
    # Update performance history
    history = state.get("agent_performance_history", {})
    if agent_name not in history:
        history[agent_name] = []
    
    history[agent_name].append(performance_data)
    
    # Keep only last 20 records per agent
    if len(history[agent_name]) > 20:
        history[agent_name] = history[agent_name][-20:]
    
    # Update quality trends
    quality_trends = state.get("quality_improvement_trends", {})
    if agent_name not in quality_trends:
        quality_trends[agent_name] = []
    
    if "quality_score" in performance_data:
        quality_trends[agent_name].append(performance_data["quality_score"])
        if len(quality_trends[agent_name]) > 10:
            quality_trends[agent_name] = quality_trends[agent_name][-10:]
    
    return {
        "agent_performance_history": history,
        "quality_improvement_trends": quality_trends
    }


def calculate_dynamic_limits(state: IterationState, agent_name: str) -> int:
    """
    Calculate dynamic iteration limits based on agent performance
    
    Args:
        state: Current iteration state
        agent_name: Name of the agent
        
    Returns:
        Calculated dynamic limit for the agent
    """
    
    base_limit = state.get("agent_limits", {}).get(agent_name, 5)
    performance_history = state.get("agent_performance_history", {}).get(agent_name, [])
    
    if not performance_history:
        return base_limit
    
    # Calculate average iterations needed for success
    successful_iterations = [
        p["iteration"] for p in performance_history 
        if p.get("outcome") == "success"
    ]
    
    if successful_iterations:
        avg_needed = sum(successful_iterations) / len(successful_iterations)
        # Add 20% margin
        dynamic_limit = int(avg_needed * 1.2)
        return max(base_limit, min(dynamic_limit, 10))  # Between base and 10
    
    return base_limit


def detect_stagnation(state: IterationState, agent_name: str) -> bool:
    """
    Detect if an agent is stuck in stagnation
    
    Args:
        state: Current iteration state
        agent_name: Name of the agent to check
        
    Returns:
        True if stagnation detected, False otherwise
    """
    
    quality_trends = state.get("quality_improvement_trends", {}).get(agent_name, [])
    
    if len(quality_trends) < 3:
        return False
    
    # Check if quality has not improved in last 3 iterations
    recent_trends = quality_trends[-3:]
    improvements = [
        recent_trends[i] - recent_trends[i-1] 
        for i in range(1, len(recent_trends))
    ]
    
    # Stagnation if all improvements are less than 0.02 (2%)
    return all(improvement < 0.02 for improvement in improvements)


def should_trigger_emergency_stop(state: IterationState) -> tuple[bool, str]:
    """
    Check if emergency stop should be triggered
    
    Args:
        state: Current iteration state
        
    Returns:
        Tuple of (should_stop, reason)
    """
    
    # Check global iteration limit
    if state.get("global_iterations", 0) >= state.get("max_global_iterations", 10):
        return True, "Global iteration limit exceeded"
    
    # Check if multiple agents are stagnating
    stagnating_agents = [
        agent for agent in state.get("agent_iterations", {}).keys()
        if detect_stagnation(state, agent)
    ]
    
    if len(stagnating_agents) >= 2:
        return True, f"Multiple agents stagnating: {stagnating_agents}"
    
    # Check resource usage
    memory_usage = state.get("memory_usage", {})
    if memory_usage.get("total", 0) > 0.9:  # 90% memory usage
        return True, "Memory usage too high"
    
    # Check processing time
    processing_times = state.get("processing_time_per_article", [])
    if processing_times and len(processing_times) > 5:
        avg_time = sum(processing_times[-5:]) / 5
        if avg_time > 300:  # 5 minutes per article
            return True, "Processing time too slow"
    
    return False, ""