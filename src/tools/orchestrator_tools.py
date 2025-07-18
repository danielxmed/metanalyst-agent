"""
Orchestrator tools for the Metanalyst Agent system.

This module contains all the tools available to the central orchestrator agent,
implementing the agents-as-a-tool pattern for the hub-and-spoke architecture.
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import json
from datetime import datetime

from ..models.state import MetanalysisState, update_state_step, log_error
from ..models.schemas import PICO, validate_pico, create_pico
from ..utils.config import get_config


@tool
def define_pico_structure(
    user_query: str
) -> str:
    """
    Analyze a natural language query and extract PICO components using LLM.
    
    Args:
        user_query: Natural language description of the meta-analysis request
        
    Returns:
        JSON string with extracted PICO structure and status
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        
        config = get_config()
        
        # Initialize LLM
        llm = ChatAnthropic(
            model_name=config.llm.primary_model,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens_to_sample=1000,
            timeout=30.0,
            stop=[]
        )
        
        # Create extraction prompt
        system_prompt = """You are a medical research expert specialized in extracting PICO components from natural language queries.

PICO stands for:
- Patient/Population: Who is the target group?
- Intervention: What treatment/intervention is being studied?
- Comparison: What is the control or comparison group?
- Outcome: What results/effects are being measured?

Your task is to analyze the user's query and extract these components clearly and concisely in English, even if the query is in Portuguese or other languages.

Return ONLY a JSON object with this exact structure:
{
  "patient": "clear description of target population",
  "intervention": "specific intervention or treatment",
  "comparison": "control group or comparison intervention",
  "outcome": "primary outcomes being measured"
}

Guidelines:
- Be specific but concise
- Use medical terminology when appropriate
- If a component is not explicitly mentioned, make reasonable assumptions
- Focus on the most relevant clinical outcomes
- Keep descriptions under 100 characters each"""

        user_prompt = f"""Extract PICO components from this query:

Query: "{user_query}"

Analyze this request and extract the PICO components. Return only the JSON object."""

        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parse LLM response - handle both string and list formats
        if isinstance(response.content, str):
            response_content = response.content.strip()
        elif isinstance(response.content, list):
            # If content is a list, join the text parts
            response_content = "".join([
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response.content
            ]).strip()
        else:
            response_content = str(response.content).strip()
        
        # Extract JSON from response (handle cases where LLM adds extra text)
        try:
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                pico_extracted = json.loads(json_str)
            else:
                # Fallback: try to parse entire response as JSON
                pico_extracted = json.loads(response_content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {response_content}")
        
        # Validate extracted PICO
        required_fields = ["patient", "intervention", "comparison", "outcome"]
        for field in required_fields:
            if field not in pico_extracted or not pico_extracted[field].strip():
                raise ValueError(f"Missing or empty PICO component: {field}")
        
        # Create structured PICO
        pico = create_pico(
            patient=pico_extracted["patient"],
            intervention=pico_extracted["intervention"],
            comparison=pico_extracted["comparison"],
            outcome=pico_extracted["outcome"]
        )
        
        # Validate PICO structure
        pico_dict = {
            "patient": pico["patient"],
            "intervention": pico["intervention"], 
            "comparison": pico["comparison"],
            "outcome": pico["outcome"]
        }
        
        if not validate_pico(pico_dict):
            raise ValueError("Invalid PICO structure extracted")
        
        result = {
            "success": True,
            "pico": pico,
            "original_query": user_query,
            "extraction_method": "LLM_analysis",
            "llm_response": response_content,
            "message": "PICO structure successfully extracted from natural language query",
            "next_step": "generate_research_query"
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "original_query": user_query,
            "message": f"Failed to extract PICO structure from query: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def call_researcher_agent(
    pico_structure: Dict[str, str],
    max_results: Optional[int] = None
) -> str:
    """
    Invoke the researcher agent to search for relevant URLs using Tavily.
    
    The researcher agent generates its own semantic queries from PICO and finds URLs.
    No separate research_query needed - the researcher handles this internally.
    
    Args:
        pico_structure: PICO structure to guide the search
        max_results: Maximum number of results to return
    
    Returns:
        JSON string with found URLs and search status
    """
    try:
        from ..agents.researcher import researcher_node
        from ..tools.tavily_tools import get_tavily_client
        
        config = get_config()
        max_results = max_results or config.search.max_papers_per_search
        
        # Check if Tavily is available
        tavily_client = get_tavily_client()
        if not tavily_client:
            result = {
                "success": False,
                "error": "Tavily API not available - check API key configuration",
                "message": "Cannot perform literature search without Tavily access"
            }
            return json.dumps(result, ensure_ascii=False)
        
        # Generate a basic query description for logging
        basic_query = f"{pico_structure.get('intervention', '')} {pico_structure.get('patient', '')} {pico_structure.get('outcome', '')}"
        print(f"🔍 Calling researcher agent for PICO-based search: {basic_query}")
        
        # Prepare state for researcher node using create_initial_state
        from ..models.state import create_initial_state
        temp_state = create_initial_state(
            pico=pico_structure,
            workflow_id="orchestrator_call"
        )
        # Set a placeholder research_query - researcher will generate its own
        temp_state["research_query"] = "PICO-based semantic search"
        temp_state["current_step"] = "url_search"
        
        # Execute researcher node directly
        researcher_result = researcher_node(temp_state)
        
        # Check if researcher succeeded
        if "error_log" in researcher_result:
            result = {
                "success": False,
                "error": researcher_result.get("error_log", [])[-1].get("message", "Unknown error"),
                "pico": pico_structure,
                "message": "Researcher agent failed to find URLs",
                "urls_found": []
            }
            return json.dumps(result, ensure_ascii=False)
        
        # Extract URLs from researcher result
        urls_found = researcher_result.get("urls_found", [])
        search_summary = researcher_result.get("search_summary", "")
        total_urls = len(urls_found)
        
        # Success result
        result_data = {
            "success": True,
            "search_executed": True,
            "pico_structure": pico_structure,
            "total_urls_found": total_urls,
            "urls_found": urls_found,
            "search_summary": search_summary,
            "max_results_requested": max_results,
            "search_timestamp": researcher_result.get("last_researcher_action"),
            "message": f"Researcher successfully found {total_urls} relevant URLs using semantic PICO search",
            "next_step": "extract_content" if urls_found else "refine_search",
            "quality_note": "URLs found are ready for content extraction by extractor agent"
        }
        
        return json.dumps(result_data, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "pico": pico_structure,
            "message": f"Failed to invoke researcher agent: {str(e)}",
            "urls_found": []
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def call_processor_agent(
    urls_to_process: list
) -> str:
    """
    Invoke the processor agent to handle complete URL processing pipeline.
    
    The processor agent combines extraction and vectorization into a single efficient process:
    1. Extracts content from URLs using Firecrawl API (focuses on main content)
    2. Processes markdown to structured JSON using GPT-4.1-mini
    3. Chunks content intelligently (1000 chars, 100 overlap)
    4. Generates embeddings with text-embedding-3-small
    5. Stores in local vector store
    6. Manages temporary files and cleanup
    
    Args:
        urls_to_process: List of URLs to process through the pipeline
    
    Returns:
        JSON string with processing results and state updates
    """
    try:
        from ..tools.processor_tools import process_urls
        
        print(f"🔄 Calling processor agent for {len(urls_to_process)} URLs...")
        
        # Call the processor tool directly
        result_str = process_urls.invoke({"url_list": urls_to_process})
        
        # Parse the result to ensure it's valid JSON
        result = json.loads(result_str)
        
        # Add metadata for orchestrator
        result["agent_called"] = "processor"
        result["processing_method"] = "combined_extraction_vectorization"
        result["timestamp"] = datetime.now().isoformat()
        
        if result.get("success"):
            print(f"✅ Processor agent successfully processed {len(result.get('url_processed', []))} URLs")
            result["next_step"] = "write_report" if result.get("vector_store_ready") else "retry_processing"
        else:
            print(f"❌ Processor agent failed: {result.get('error', 'Unknown error')}")
            result["next_step"] = "retry_processing"
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": f"Failed to invoke processor agent: {str(e)}",
            "url_processed": [],
            "url_not_processed": urls_to_process,
            "vector_store_ready": False
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def call_writer_agent(
    vector_store_path: str,
    pico_structure: dict
) -> str:
    """
    Invoke the writer agent to generate the initial report draft.
    
    Args:
        vector_store_path: Path to the vector store
        pico_structure: PICO structure for the analysis
        
    Returns:
        JSON string with report draft and status
    """
    try:
        # TODO: This will be replaced with actual writer agent when implemented
        # For now, return placeholder results
        
        report_draft = f"""
        <html>
        <head><title>Meta-Analysis Report Draft</title></head>
        <body>
        <h1>Meta-Analysis Report</h1>
        <h2>PICO Structure</h2>
        <p>Patient: {pico_structure.get('patient', 'N/A')}</p>
        <p>Intervention: {pico_structure.get('intervention', 'N/A')}</p>
        <p>Comparison: {pico_structure.get('comparison', 'N/A')}</p>
        <p>Outcome: {pico_structure.get('outcome', 'N/A')}</p>
        <h2>Results</h2>
        <p>This is a draft report based on the analyzed literature...</p>
        </body>
        </html>
        """
        
        result = {
            "success": True,
            "report_draft": report_draft,
            "sections_created": ["introduction", "methods", "results", "discussion"],
            "word_count": 500,
            "message": "Successfully generated initial report draft",
            "next_step": "review_report"
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": "Failed to generate report draft"
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def call_reviewer_agent(
    report_draft: str
) -> str:
    """
    Invoke the reviewer agent to review report quality.
    
    Args:
        report_draft: The report draft to review
        
    Returns:
        JSON string with review feedback and status
    """
    try:
        # TODO: This will be replaced with actual reviewer agent when implemented
        # For now, return placeholder results
        
        review_feedback = {
            "overall_quality": 8.5,
            "methodology_score": 9.0,
            "clarity_score": 8.0,
            "completeness_score": 8.0,
            "needs_more_research": False,
            "approval_status": "approved",
            "suggestions": [
                "Consider adding more details in the methodology section",
                "Include confidence intervals in all statistical results"
            ],
            "required_changes": []
        }
        
        result = {
            "success": True,
            "review_feedback": review_feedback,
            "report_approved": True,
            "message": "Report review completed - approved for next phase",
            "next_step": "statistical_analysis"
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": "Failed to review report"
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def call_analyst_agent(
    extracted_papers: list,
    analysis_type: str = "meta_analysis"
) -> str:
    """
    Invoke the analyst agent to perform statistical analysis.
    
    Args:
        extracted_papers: List of papers with statistical data
        analysis_type: Type of analysis to perform
        
    Returns:
        JSON string with analysis results and status
    """
    try:
        # TODO: This will be replaced with actual analyst agent when implemented
        # For now, return placeholder results
        
        statistical_analysis = {
            "pooled_effect_size": 0.85,
            "confidence_interval": [0.70, 1.03],
            "p_value": 0.045,
            "heterogeneity": {
                "i_squared": 45.2,
                "q_statistic": 12.5,
                "p_heterogeneity": 0.08
            },
            "studies_included": len(extracted_papers),
            "total_participants": sum([paper.get("statistics", {}).get("sample_size", 0) for paper in extracted_papers])
        }
        
        result = {
            "success": True,
            "statistical_analysis": statistical_analysis,
            "forest_plot_path": "outputs/figures/forest_plot.html",
            "tables_created": ["summary_statistics", "study_characteristics"],
            "figures_created": ["forest_plot", "funnel_plot"],
            "message": "Statistical analysis completed successfully",
            "next_step": "integrate_final_report"
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": "Failed to perform statistical analysis"
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def call_editor_agent(
    report_draft: str,
    statistical_analysis: dict,
    figures_paths: list
) -> str:
    """
    Invoke the editor agent to create the final integrated report.
    
    Args:
        report_draft: Initial report draft
        statistical_analysis: Results from statistical analysis
        figures_paths: Paths to generated figures
        
    Returns:
        JSON string with final report and status
    """
    try:
        # TODO: This will be replaced with actual editor agent when implemented
        # For now, return placeholder results
        
        final_report = f"""
        <html>
        <head><title>Final Meta-Analysis Report</title></head>
        <body>
        <h1>Complete Meta-Analysis Report</h1>
        {report_draft.replace('<html><head><title>Meta-Analysis Report Draft</title></head><body>', '')}
        <h2>Statistical Analysis</h2>
        <p>Pooled Effect Size: {statistical_analysis.get('pooled_effect_size', 'N/A')}</p>
        <p>95% CI: {statistical_analysis.get('confidence_interval', 'N/A')}</p>
        <p>P-value: {statistical_analysis.get('p_value', 'N/A')}</p>
        <h2>Figures</h2>
        <p>Forest Plot: <a href="{figures_paths[0] if figures_paths else '#'}">View</a></p>
        </body>
        </html>
        """
        
        result = {
            "success": True,
            "final_report": final_report,
            "final_report_path": "outputs/final_report.html",
            "sections_integrated": ["analysis", "figures", "tables"],
            "completed_at": datetime.now().isoformat(),
            "message": "Final report successfully created and integrated",
            "workflow_complete": True
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": "Failed to create final report"
        }
        return json.dumps(result, ensure_ascii=False)


@tool
def get_workflow_status() -> str:
    """
    Get the current status and summary of the workflow.
    
    Returns:
        JSON string with workflow status information
    """
    try:
        # This tool can be used by the orchestrator to understand current state
        result = {
            "success": True,
            "message": "Workflow status retrieved",
            "available_tools": [
                "define_pico_structure",
                "generate_research_query", 
                "call_researcher_agent",
                "call_extractor_agent",
                "call_vectorizer_agent",
                "call_writer_agent",
                "call_reviewer_agent",
                "call_analyst_agent",
                "call_editor_agent"
            ],
            "workflow_steps": [
                "1. Define PICO structure",
                "2. Generate research query",
                "3. Search literature",
                "4. Extract content",
                "5. Create vector store",
                "6. Write report draft",
                "7. Review report",
                "8. Perform statistical analysis",
                "9. Create final report"
            ]
        }
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "message": "Failed to get workflow status"
        }
        return json.dumps(result, ensure_ascii=False)


# List of all orchestrator tools for easy import
ORCHESTRATOR_TOOLS = [
    define_pico_structure,
    call_researcher_agent,
    call_processor_agent,
    call_writer_agent,
    call_reviewer_agent,
    call_analyst_agent,
    call_editor_agent,
    get_workflow_status
]
