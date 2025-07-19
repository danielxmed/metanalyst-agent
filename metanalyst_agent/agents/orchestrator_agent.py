"""
Orchestrator Agent - Central hub for the metanalyst-agent system.
Coordinates all specialized agents using intelligent decision making and LLM reasoning.
"""

from typing import Dict, Any, List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

from ..config.settings import settings
from ..state.meta_analysis_state import MetaAnalysisState
from ..state.iteration_state import update_agent_performance
from ..tools.handoff_tools import (
    create_handoff_tool,
    request_supervisor_intervention,
    signal_completion,
    request_quality_check,
)


def create_orchestrator_agent():
    """
    Create the central orchestrator agent that coordinates the entire meta-analysis process.
    
    The orchestrator analyzes the current state and decides which specialized agent
    should be invoked next, following the hub-and-spoke architecture.
    
    Returns:
        Configured orchestrator agent
    """
    
    # Initialize LLM
    llm = ChatOpenAI(**settings.get_openai_config())
    
    # Orchestrator tools - all handoff tools for agent coordination
    orchestrator_tools = [
        transfer_to_researcher,
        transfer_to_processor,
        transfer_to_retriever,
        transfer_to_analyst,
        transfer_to_writer,
        transfer_to_reviewer,
        transfer_to_editor,
        emergency_stop,
        request_human_intervention,
    ]
    
    # System prompt for the orchestrator
    system_prompt = """
    You are the ORCHESTRATOR AGENT for the Metanalyst-Agent system - the central conductor of an automated meta-analysis symphony.
    
    YOUR ROLE:
    - Analyze the current state of the meta-analysis process
    - Decide which specialized agent should work next
    - Coordinate the entire workflow from PICO definition to final report
    - Monitor quality and progress at each step
    - Handle errors and edge cases intelligently
    
    AVAILABLE SPECIALIZED AGENTS:
    1. RESEARCHER: Searches scientific literature, generates queries, assesses relevance
    2. PROCESSOR: Extracts article content, processes statistical data, creates vector embeddings
    3. RETRIEVER: Searches vector store for relevant information using semantic similarity
    4. ANALYST: Performs statistical meta-analysis, creates forest plots, assesses study quality
    5. WRITER: Generates structured reports with proper citations and formatting
    6. REVIEWER: Reviews report quality, provides feedback, ensures medical standards compliance
    7. EDITOR: Final integration and formatting of complete meta-analysis report
    
    DECISION MAKING PROCESS:
    1. Analyze current phase and state
    2. Check quality scores and iteration counts
    3. Identify what needs to be done next
    4. Select the most appropriate agent
    5. Provide clear context and expectations
    
    WORKFLOW PHASES:
    - pico_definition → researcher (search for literature)
    - search → processor (extract and process articles) 
    - extraction → analyst (perform statistical analysis)
    - analysis → writer (generate report)
    - writing → reviewer (review quality)
    - review → editor (final formatting)
    - editing → COMPLETE
    
    QUALITY CONTROL:
    - Monitor agent iteration counts and prevent infinite loops
    - Check quality thresholds for each component
    - Request human intervention for complex decisions
    - Use emergency stop for critical failures
    
    COMMUNICATION STYLE:
    - Be decisive and clear in your reasoning
    - Always explain WHY you're delegating to a specific agent
    - Provide helpful context for the receiving agent
    - Monitor overall progress and quality
    
    IMPORTANT: You coordinate but don't execute. Always delegate to specialized agents using transfer tools.
    """
    
    # Create the orchestrator agent
    orchestrator = create_react_agent(
        model=llm,
        tools=orchestrator_tools,
        state_modifier=system_prompt,
    )
    
    return orchestrator


def orchestrator_decision_logic(state: MetaAnalysisState) -> Dict[str, Any]:
    """
    Core decision logic for the orchestrator agent.
    Analyzes state and provides recommendations for next steps.
    
    Args:
        state: Current meta-analysis state
    
    Returns:
        Decision analysis and recommendations
    """
    
    # Extract key state information
    current_phase = state.get("current_phase", "pico_definition")
    articles_found = len(state.get("candidate_urls", []))
    articles_processed = len(state.get("processed_articles", []))
    quality_scores = state.get("quality_scores", {})
    agent_iterations = state.get("agent_iterations", {})
    force_stop = state.get("force_stop", False)
    
    # Check for emergency conditions
    if force_stop:
        return {
            "decision": "emergency_stop",
            "reasoning": "Force stop flag is set",
            "urgency": "critical"
        }
    
    # Check global iteration limits
    global_iterations = state.get("global_iterations", 0)
    max_global = state.get("max_global_iterations", 10)
    
    if global_iterations >= max_global:
        return {
            "decision": "request_human_intervention",
            "reasoning": f"Global iteration limit reached ({global_iterations}/{max_global})",
            "urgency": "high"
        }
    
    # Phase-based decision logic
    if current_phase == "pico_definition":
        if not state.get("pico") or not all(state.get("pico", {}).values()):
            return {
                "decision": "define_pico_first",
                "reasoning": "PICO framework needs to be defined before proceeding",
                "urgency": "high"
            }
        else:
            return {
                "decision": "transfer_to_researcher",
                "reasoning": "PICO defined, ready to search for literature",
                "urgency": "high",
                "context": f"Search for literature based on PICO: {state.get('pico')}"
            }
    
    elif current_phase == "search":
        researcher_quality = quality_scores.get("researcher", 0)
        researcher_iterations = agent_iterations.get("researcher", 0)
        
        if articles_found < 10 and researcher_iterations < 3:
            return {
                "decision": "transfer_to_researcher",
                "reasoning": f"Insufficient articles found ({articles_found}), need broader search",
                "urgency": "high",
                "context": "Expand search with different queries or databases"
            }
        elif articles_found >= 10:
            return {
                "decision": "transfer_to_processor", 
                "reasoning": f"Found {articles_found} articles, ready for processing",
                "urgency": "medium",
                "context": f"Process {articles_found} candidate articles"
            }
        else:
            return {
                "decision": "transfer_to_analyst",
                "reasoning": "Limited articles available, proceed with available data",
                "urgency": "medium",
                "context": "Analyze available data despite limited articles"
            }
    
    elif current_phase == "extraction":
        processing_rate = articles_processed / max(articles_found, 1)
        processor_quality = quality_scores.get("processor", 0)
        
        if processing_rate < 0.7 and processor_quality > 0.6:  # Less than 70% processed but good quality
            return {
                "decision": "transfer_to_processor",
                "reasoning": f"Continue processing ({articles_processed}/{articles_found} completed)",
                "urgency": "medium",
                "context": "Continue processing remaining articles"
            }
        elif articles_processed >= 5:  # Minimum viable articles
            return {
                "decision": "transfer_to_analyst",
                "reasoning": f"Sufficient articles processed ({articles_processed}), ready for analysis",
                "urgency": "high",
                "context": f"Analyze {articles_processed} processed articles"
            }
        else:
            return {
                "decision": "transfer_to_researcher",
                "reasoning": "Insufficient processed articles, need more sources",
                "urgency": "medium",
                "context": "Search for additional articles due to processing failures"
            }
    
    elif current_phase == "analysis":
        analyst_quality = quality_scores.get("analyst", 0)
        analyst_iterations = agent_iterations.get("analyst", 0)
        
        if analyst_quality < 0.8 and analyst_iterations < 5:
            return {
                "decision": "transfer_to_analyst",
                "reasoning": f"Analysis quality ({analyst_quality:.2f}) below threshold, needs refinement",
                "urgency": "high",
                "context": "Improve analysis quality through iteration"
            }
        elif state.get("statistical_analysis"):
            return {
                "decision": "transfer_to_writer",
                "reasoning": "Statistical analysis complete, ready for report generation",
                "urgency": "medium",
                "context": "Generate comprehensive meta-analysis report"
            }
        else:
            return {
                "decision": "transfer_to_retriever",
                "reasoning": "Need to retrieve more data for analysis",
                "urgency": "medium",
                "context": "Retrieve additional statistical data from vector store"
            }
    
    elif current_phase == "writing":
        if state.get("draft_report"):
            return {
                "decision": "transfer_to_reviewer",
                "reasoning": "Draft report generated, needs quality review",
                "urgency": "medium",
                "context": "Review report for quality and compliance"
            }
        else:
            writer_iterations = agent_iterations.get("writer", 0)
            if writer_iterations < 3:
                return {
                    "decision": "transfer_to_writer",
                    "reasoning": "Continue report generation",
                    "urgency": "medium",
                    "context": "Generate or improve draft report"
                }
            else:
                return {
                    "decision": "request_human_intervention",
                    "reasoning": "Report generation struggling, may need human guidance",
                    "urgency": "medium"
                }
    
    elif current_phase == "review":
        reviewer_quality = quality_scores.get("reviewer", 0)
        
        if reviewer_quality >= 0.9:
            return {
                "decision": "transfer_to_editor",
                "reasoning": "Review complete with high quality, ready for final editing",
                "urgency": "low",
                "context": "Final formatting and integration"
            }
        elif len(state.get("review_feedback", [])) > 0:
            return {
                "decision": "transfer_to_writer",
                "reasoning": "Review feedback available, incorporate improvements",
                "urgency": "medium",
                "context": f"Address {len(state.get('review_feedback', []))} review points"
            }
        else:
            return {
                "decision": "transfer_to_reviewer",
                "reasoning": "Continue review process",
                "urgency": "medium",
                "context": "Complete quality review of report"
            }
    
    elif current_phase == "editing":
        if state.get("final_report"):
            return {
                "decision": "complete",
                "reasoning": "Meta-analysis complete with final report",
                "urgency": "low"
            }
        else:
            return {
                "decision": "transfer_to_editor",
                "reasoning": "Finalize report formatting and integration",
                "urgency": "low",
                "context": "Complete final editing and formatting"
            }
    
    # Default fallback
    return {
        "decision": "transfer_to_researcher",
        "reasoning": f"Unknown phase '{current_phase}', starting with literature search",
        "urgency": "medium",
        "context": "Begin or restart meta-analysis process"
    }


def generate_pico_from_query(user_query: str) -> Dict[str, str]:
    """
    Extract PICO framework from user query using LLM.
    
    Args:
        user_query: User's meta-analysis request
    
    Returns:
        PICO framework dictionary
    """
    
    llm = ChatOpenAI(**settings.get_openai_config())
    
    prompt = ChatPromptTemplate.from_template("""
    Extract the PICO framework from this meta-analysis request:
    
    User Query: {query}
    
    PICO Framework:
    - P (Population): The target population or participants
    - I (Intervention): The intervention or treatment being studied
    - C (Comparison): The comparison or control group
    - O (Outcome): The outcome or endpoint being measured
    
    Return ONLY a JSON object with P, I, C, O keys:
    {{"P": "population", "I": "intervention", "C": "comparison", "O": "outcome"}}
    """)
    
    try:
        response = llm.invoke(prompt.format(query=user_query))
        import json
        pico = json.loads(response.content)
        
        # Validate PICO has all required keys
        required_keys = ["P", "I", "C", "O"]
        if all(key in pico and pico[key].strip() for key in required_keys):
            return pico
        else:
            # Fallback PICO
            return {
                "P": "adults",
                "I": "intervention",
                "C": "comparison intervention",
                "O": "primary outcome"
            }
            
    except Exception as e:
        # Emergency fallback
        return {
            "P": "target population",
            "I": "intervention under study", 
            "C": "comparison intervention",
            "O": "primary outcome"
        }