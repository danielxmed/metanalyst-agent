"""Handoff tools for agent-to-agent communication and control transfer"""

from typing import Annotated, Dict, Any
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.graph import MessagesState

from ..state.iteration_state import IterationState


def create_handoff_tool(*, agent_name: str, description: str):
    """
    Factory function to create handoff tools for transferring control between agents
    
    Args:
        agent_name: Name of the target agent to transfer control to
        description: Description of when and why to use this handoff
        
    Returns:
        Tool function for handoff to the specified agent
    """
    
    tool_name = f"transfer_to_{agent_name}"
    
    @tool(tool_name, description=description)
    def handoff_tool(
        reason: Annotated[str, "Detailed reason for transferring to this agent"],
        context: Annotated[str, "Context and information relevant for the next agent"],
        priority: Annotated[str, "Priority level: low, medium, high, urgent"] = "medium",
        expected_outcome: Annotated[str, "What you expect this agent to accomplish"] = "",
        state: Annotated[IterationState, InjectedState] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = None,
    ) -> Command:
        """
        Transfer control to another agent with comprehensive context
        
        Args:
            reason: Why the transfer is happening
            context: Relevant information for the next agent
            priority: Priority level for the task
            expected_outcome: Expected results from the target agent
            state: Current system state (injected)
            tool_call_id: Tool call identifier (injected)
            
        Returns:
            Command to transfer control to the target agent
        """
        
        # Create tool message confirming the transfer
        tool_message = ToolMessage(
            content=f"Transferring control to {agent_name}. Reason: {reason}",
            tool_call_id=tool_call_id,
            name=tool_name
        )
        
        # Create context message for the receiving agent
        context_message = {
            "role": "system", 
            "content": f"""
HANDOFF RECEIVED FROM: {state.get('current_agent', 'unknown')}

TRANSFER REASON: {reason}

CONTEXT: {context}

PRIORITY: {priority}

EXPECTED OUTCOME: {expected_outcome}

CURRENT PHASE: {state.get('current_phase', 'unknown')}

PROGRESS SUMMARY:
- Total articles processed: {state.get('total_articles_processed', 0)}
- Current quality score: {state.get('quality_scores', {}).get('overall', 0)}
- Iterations completed: {state.get('global_iterations', 0)}

Please proceed with your specialized tasks based on this context.
""".strip()
        }
        
        # Update state with handoff information
        update_dict = {
            "messages": [tool_message, context_message],
            "current_agent": agent_name,
            "last_handoff": {
                "from_agent": state.get('current_agent', 'unknown'),
                "to_agent": agent_name,
                "reason": reason,
                "context": context,
                "priority": priority,
                "timestamp": "2024-01-01T00:00:00Z"  # Would use datetime.now() in real implementation
            }
        }
        
        # Track agent iterations
        if "agent_iterations" in state:
            agent_iterations = state["agent_iterations"].copy()
            agent_iterations[agent_name] = agent_iterations.get(agent_name, 0) + 1
            update_dict["agent_iterations"] = agent_iterations
        
        return Command(
            goto=agent_name,
            update=update_dict,
            graph=Command.PARENT  # Navigate to parent graph
        )
    
    return handoff_tool


# Create specific handoff tools for each agent
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description=(
        "Transfer to the Researcher Agent when you need to search for more scientific literature, "
        "generate new search queries, or assess article relevance. Use this when you need more "
        "articles to meet quality thresholds or when current search results are insufficient."
    )
)

transfer_to_processor = create_handoff_tool(
    agent_name="processor", 
    description=(
        "Transfer to the Processor Agent when you have URLs of articles that need to be processed. "
        "This agent will extract content, analyze statistical data, generate citations, and create "
        "vector embeddings. Use when you have collected relevant articles that need processing."
    )
)

transfer_to_retriever = create_handoff_tool(
    agent_name="retriever",
    description=(
        "Transfer to the Retriever Agent when you need to search the vector store for specific "
        "information, find similar studies, or gather relevant chunks for analysis. Use after "
        "articles have been processed and vectorized."
    )
)

transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description=(
        "Transfer to the Analyst Agent when you have sufficient statistical data to perform "
        "meta-analysis calculations, create forest plots, assess heterogeneity, or conduct "
        "sensitivity analyses. Use when data extraction is complete and ready for analysis."
    )
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer",
    description=(
        "Transfer to the Writer Agent when statistical analysis is complete and you need to "
        "generate structured reports, synthesize findings, and create comprehensive documentation. "
        "Use when analysis results are ready for report generation."
    )
)

transfer_to_reviewer = create_handoff_tool(
    agent_name="reviewer",
    description=(
        "Transfer to the Reviewer Agent when you have a draft report that needs quality review, "
        "bias assessment, or methodological validation. Use to ensure the meta-analysis meets "
        "scientific standards and identify areas for improvement."
    )
)

transfer_to_editor = create_handoff_tool(
    agent_name="editor",
    description=(
        "Transfer to the Editor Agent for final report integration, formatting, and preparation "
        "of the complete meta-analysis document. Use when all components are ready for final "
        "compilation and presentation."
    )
)


@tool
def request_supervisor_intervention(
    issue_description: Annotated[str, "Description of the issue requiring supervisor attention"],
    severity: Annotated[str, "Severity level: low, medium, high, critical"] = "medium",
    suggested_action: Annotated[str, "Suggested action to resolve the issue"] = "",
    state: Annotated[IterationState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Request intervention from the supervisor agent for issues that require oversight
    
    Args:
        issue_description: What problem needs supervisor attention
        severity: How critical the issue is
        suggested_action: Recommended course of action
        state: Current system state (injected)
        tool_call_id: Tool call identifier (injected)
        
    Returns:
        Command to transfer control to supervisor
    """
    
    tool_message = ToolMessage(
        content=f"Requesting supervisor intervention: {issue_description}",
        tool_call_id=tool_call_id,
        name="request_supervisor_intervention"
    )
    
    intervention_message = {
        "role": "system",
        "content": f"""
SUPERVISOR INTERVENTION REQUESTED

REQUESTING AGENT: {state.get('current_agent', 'unknown')}
SEVERITY: {severity}

ISSUE: {issue_description}

SUGGESTED ACTION: {suggested_action}

CURRENT STATE:
- Phase: {state.get('current_phase', 'unknown')}
- Global iterations: {state.get('global_iterations', 0)}
- Quality satisfied: {state.get('quality_satisfied', False)}
- Force stop flag: {state.get('force_stop', False)}

Please assess the situation and determine the appropriate course of action.
""".strip()
    }
    
    return Command(
        goto="supervisor",
        update={
            "messages": [tool_message, intervention_message],
            "intervention_requested": True,
            "intervention_details": {
                "requesting_agent": state.get('current_agent', 'unknown'),
                "issue": issue_description,
                "severity": severity,
                "suggested_action": suggested_action,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        },
        graph=Command.PARENT
    )


@tool 
def signal_completion(
    completion_reason: Annotated[str, "Reason why the task is considered complete"],
    quality_assessment: Annotated[str, "Assessment of the quality of completed work"],
    final_outputs: Annotated[str, "Description of final outputs produced"] = "",
    state: Annotated[IterationState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Signal that the current agent's work is complete and ready for next phase
    
    Args:
        completion_reason: Why the work is considered complete
        quality_assessment: Assessment of work quality
        final_outputs: What was produced
        state: Current system state (injected)
        tool_call_id: Tool call identifier (injected)
        
    Returns:
        Command to return to supervisor for next phase decision
    """
    
    tool_message = ToolMessage(
        content=f"Work completed: {completion_reason}",
        tool_call_id=tool_call_id,
        name="signal_completion"
    )
    
    completion_message = {
        "role": "system",
        "content": f"""
WORK COMPLETION SIGNAL

COMPLETING AGENT: {state.get('current_agent', 'unknown')}

COMPLETION REASON: {completion_reason}

QUALITY ASSESSMENT: {quality_assessment}

OUTPUTS PRODUCED: {final_outputs}

The agent has finished its current tasks and is ready for the next phase of the meta-analysis.
""".strip()
    }
    
    return Command(
        goto="supervisor",
        update={
            "messages": [tool_message, completion_message],
            "work_completed": True,
            "completion_details": {
                "completing_agent": state.get('current_agent', 'unknown'),
                "reason": completion_reason,
                "quality": quality_assessment,
                "outputs": final_outputs,
                "timestamp": "2024-01-01T00:00:00Z"
            }
        },
        graph=Command.PARENT
    )


@tool
def request_quality_check(
    work_description: Annotated[str, "Description of work that needs quality checking"],
    quality_concerns: Annotated[str, "Specific quality concerns or areas to focus on"] = "",
    state: Annotated[IterationState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Request quality check from the reviewer agent
    
    Args:
        work_description: What work needs to be reviewed
        quality_concerns: Specific areas of concern
        state: Current system state (injected)
        tool_call_id: Tool call identifier (injected)
        
    Returns:
        Command to transfer to reviewer agent
    """
    
    return transfer_to_reviewer(
        reason=f"Quality check requested for: {work_description}",
        context=f"Quality concerns: {quality_concerns}" if quality_concerns else "General quality review requested",
        priority="high",
        expected_outcome="Quality assessment and improvement recommendations",
        state=state,
        tool_call_id=tool_call_id
    )