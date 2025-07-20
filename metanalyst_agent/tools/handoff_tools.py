"""Handoff tools for agent-to-agent communication and control transfer"""

from typing import Annotated, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..state.meta_analysis_state import MetaAnalysisState


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
    ) -> str:
        """
        Transfer control to another agent with comprehensive context
        
        Args:
            reason: Why the transfer is happening
            context: Relevant information for the next agent
            priority: Priority level for the task
            expected_outcome: Expected results from the target agent
            
        Returns:
            Confirmation message of the transfer
        """
        
        # Return a simple string that will be used by the orchestrator to route
        return f"transfer_to_{agent_name}|{reason}|{context}|{priority}|{expected_outcome}"
    
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
) -> str:
    """
    Request intervention from the supervisor agent for issues that require oversight
    
    Args:
        issue_description: What problem needs supervisor attention
        severity: How critical the issue is
        suggested_action: Recommended course of action
        
    Returns:
        Confirmation message of the intervention request
    """
    
    return f"supervisor_intervention|{issue_description}|{severity}|{suggested_action}"


@tool 
def signal_completion(
    completion_reason: Annotated[str, "Reason why the task is considered complete"],
    quality_assessment: Annotated[str, "Assessment of the quality of completed work"],
    final_outputs: Annotated[str, "Description of final outputs produced"] = "",
) -> str:
    """
    Signal that the current agent's work is complete and ready for next phase
    
    Args:
        completion_reason: Why the work is considered complete
        quality_assessment: Assessment of work quality
        final_outputs: What was produced
        
    Returns:
        Confirmation message of completion
    """
    
    return f"work_completed|{completion_reason}|{quality_assessment}|{final_outputs}"


@tool
def request_quality_check(
    work_description: Annotated[str, "Description of work that needs quality checking"],
    quality_concerns: Annotated[str, "Specific quality concerns or areas to focus on"] = "",
) -> str:
    """
    Request quality check from the reviewer agent
    
    Args:
        work_description: What work needs to be reviewed
        quality_concerns: Specific areas of concern
        
    Returns:
        Confirmation message of quality check request
    """
    
    return f"quality_check_requested|{work_description}|{quality_concerns}"
