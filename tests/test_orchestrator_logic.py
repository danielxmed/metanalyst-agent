"""
Test script to demonstrate the orchestrator decision logic without external dependencies.

This script shows how the multi-agent system workflow decision engine works
by simulating the complete meta-analysis process step by step.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid

# Simplified state structure (without LangGraph dependencies)
class SimpleMetanalysisState(dict):
    """Simplified state for testing purposes."""
    pass


def create_simple_pico(patient: str, intervention: str, comparison: str, outcome: str) -> Dict[str, str]:
    """Create a simple PICO structure."""
    return {
        "patient": patient.strip(),
        "intervention": intervention.strip(), 
        "comparison": comparison.strip(),
        "outcome": outcome.strip()
    }


def validate_simple_pico(pico: Dict[str, str]) -> bool:
    """Validate PICO structure."""
    required_fields = ["patient", "intervention", "comparison", "outcome"]
    return all(
        field in pico and isinstance(pico[field], str) and len(pico[field].strip()) > 0
        for field in required_fields
    )


def create_simple_initial_state(pico: Optional[Dict[str, str]] = None) -> SimpleMetanalysisState:
    """Create initial state for testing."""
    return SimpleMetanalysisState({
        "messages": [],
        "pico": pico,
        "research_query": None,
        "urls_found": [],
        "search_results": [],
        "urls_processed": [],
        "extracted_papers": [],
        "processed_papers": [],
        "vector_store_ready": False,
        "vector_store_path": None,
        "chunks_created": None,
        "relevant_chunks": [],
        "report_draft": None,
        "report_approved": False,
        "review_feedback": None,
        "statistical_analysis": None,
        "forest_plot_path": None,
        "analysis_tables": [],
        "final_report": None,
        "final_report_path": None,
        "current_step": "initialize",
        "current_agent": None,
        "step_history": [],
        "error_log": [],
        "workflow_id": str(uuid.uuid4()),
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "total_papers_found": 0,
        "total_papers_processed": 0
    })


class SimpleOrchestratorDecisionEngine:
    """
    Simplified version of the orchestrator decision engine for testing.
    
    This demonstrates the core decision logic without external dependencies.
    """
    
    @staticmethod
    def determine_next_agent(state: SimpleMetanalysisState) -> str:
        """
        Analyze the current state and determine the next agent to invoke.
        
        This is the core decision logic implementing the contextual flow
        described in the CONTEXT.md file.
        """
        # Step 1: Check if PICO is defined and valid
        if not state.get("pico") or not validate_simple_pico(state["pico"]):
            return "define_pico"
        
        # Step 2: If PICO is defined but no research query, generate query
        if not state.get("research_query"):
            return "query_generator"
        
        # Step 3: If query exists but no URLs found, search literature
        if not state.get("urls_found") or len(state["urls_found"]) == 0:
            return "researcher"
        
        # Step 4: If URLs exist but not all processed, extract content
        urls_found = state.get("urls_found", [])
        urls_processed = state.get("urls_processed", [])
        if len(urls_processed) < len(urls_found):
            return "extractor"
        
        # Step 5: If papers extracted but vector store not ready, vectorize
        if (state.get("extracted_papers") and 
            len(state["extracted_papers"]) > 0 and 
            not state.get("vector_store_ready", False)):
            return "vectorizer"
        
        # Step 6: If vector store ready but no report draft, write report
        if (state.get("vector_store_ready", False) and 
            not state.get("report_draft")):
            return "writer"
        
        # Step 7: If report exists but not reviewed, review it
        if (state.get("report_draft") and 
            not state.get("review_feedback")):
            return "reviewer"
        
        # Step 8: If review suggests more research, go back to researcher
        review_feedback = state.get("review_feedback", {})
        if review_feedback.get("needs_more_research", False):
            return "researcher"
        
        # Step 9: If report approved but no statistical analysis, analyze
        if (state.get("report_approved", False) and 
            not state.get("statistical_analysis")):
            return "analyst"
        
        # Step 10: If analysis ready but not integrated, edit final report
        if (state.get("statistical_analysis") and 
            not state.get("final_report")):
            return "editor"
        
        # Step 11: If everything is complete, finish
        if state.get("final_report"):
            return "END"
        
        # Default fallback
        return "END"
    
    @staticmethod
    def get_decision_rationale(state: SimpleMetanalysisState, next_agent: str) -> str:
        """Provide human-readable rationale for the decision."""
        rationales = {
            "define_pico": "❌ PICO structure is missing or invalid. Need to define research question.",
            "query_generator": "📝 PICO is defined but research query not generated yet.", 
            "researcher": "🔍 Need to search for literature or additional sources requested by reviewer.",
            "extractor": "📄 URLs found but content not yet extracted from all sources.",
            "vectorizer": "🧠 Papers extracted but vector store not created yet.",
            "writer": "✍️  Vector store ready but initial report not written.",
            "reviewer": "👁️  Report draft exists but quality review not performed.",
            "analyst": "📊 Report approved but statistical analysis not performed.",
            "editor": "📝 Analysis complete but not integrated into final report.",
            "END": "✅ All workflow steps completed successfully."
        }
        
        return rationales.get(next_agent, f"🔄 Proceeding to {next_agent}")


def simulate_agent_execution(agent_name: str, state: SimpleMetanalysisState) -> None:
    """Simulate the execution of each agent by updating the state."""
    
    if agent_name == "define_pico":
        # This would normally be done by user input or LLM
        state["pico"] = state.get("pico")  # Should already be set
        state["current_step"] = "pico_defined"
        print("   ✅ PICO structure validated and accepted")
        
    elif agent_name == "query_generator":
        pico = state["pico"]
        query_parts = [pico["patient"], pico["intervention"], pico["comparison"], pico["outcome"]]
        state["research_query"] = " AND ".join([f'"{part}"' for part in query_parts if part.strip()])
        state["current_step"] = "query_generated"
        print(f"   ✅ Generated research query: {state['research_query']}")
        
    elif agent_name == "researcher":
        # Simulate finding research URLs
        state["urls_found"] = [
            "https://pubmed.ncbi.nlm.nih.gov/example1",
            "https://pubmed.ncbi.nlm.nih.gov/example2",
            "https://cochranelibrary.com/example3",
            "https://bmj.com/example4",
            "https://nejm.org/example5"
        ]
        state["total_papers_found"] = len(state["urls_found"])
        state["current_step"] = "research_complete"
        print(f"   ✅ Found {len(state['urls_found'])} relevant paper URLs")
        
    elif agent_name == "extractor":
        # Simulate extracting content from URLs
        urls = state["urls_found"]
        state["extracted_papers"] = [
            {"paper_id": f"paper_{i+1}", "title": f"Example Paper {i+1}", 
             "url": url, "content": f"Extracted content from paper {i+1}"}
            for i, url in enumerate(urls)
        ]
        state["urls_processed"] = urls.copy()
        state["total_papers_processed"] = len(state["extracted_papers"])
        state["current_step"] = "extraction_complete"
        print(f"   ✅ Extracted content from {len(state['extracted_papers'])} papers")
        
    elif agent_name == "vectorizer":
        # Simulate creating vector store
        papers = state["extracted_papers"]
        state["vector_store_ready"] = True
        state["vector_store_path"] = "data/vector_store"
        state["chunks_created"] = len(papers) * 10  # Assume 10 chunks per paper
        state["current_step"] = "vectorization_complete"
        print(f"   ✅ Created vector store with {state['chunks_created']} chunks")
        
    elif agent_name == "writer":
        # Simulate writing report draft
        state["report_draft"] = "<html><h1>Meta-Analysis Report Draft</h1><p>Initial analysis...</p></html>"
        state["current_step"] = "writing_complete" 
        print("   ✅ Generated initial report draft")
        
    elif agent_name == "reviewer":
        # Simulate reviewing the report
        state["review_feedback"] = {
            "overall_quality": 8.5,
            "needs_more_research": False,
            "approval_status": "approved",
            "reviewer_comments": "Good quality report, approved for analysis."
        }
        state["report_approved"] = True
        state["current_step"] = "review_complete"
        print("   ✅ Report reviewed and approved (Quality: 8.5/10)")
        
    elif agent_name == "analyst":
        # Simulate statistical analysis
        state["statistical_analysis"] = {
            "pooled_effect_size": 0.85,
            "confidence_interval": [0.70, 1.03],
            "p_value": 0.045,
            "heterogeneity_i2": 25.3,
            "studies_included": len(state["extracted_papers"])
        }
        state["forest_plot_path"] = "outputs/figures/forest_plot.html"
        state["current_step"] = "analysis_complete"
        print("   ✅ Statistical analysis completed (Pooled effect: 0.85, CI: [0.70, 1.03])")
        
    elif agent_name == "editor":
        # Simulate final report integration
        state["final_report"] = "<html><h1>Final Meta-Analysis Report</h1><p>Complete analysis with statistics...</p></html>"
        state["final_report_path"] = "outputs/final_report.html"
        state["current_step"] = "editing_complete"
        state["completed_at"] = datetime.now().isoformat()
        print("   ✅ Final report generated and saved")


def run_orchestrator_simulation():
    """Run a complete simulation of the orchestrator decision process."""
    
    print("🔬 Metanalyst Agent - Orchestrator Decision Logic Demo")
    print("=" * 60)
    
    # Create example PICO
    pico = create_simple_pico(
        patient="adults with type 2 diabetes",
        intervention="metformin therapy", 
        comparison="placebo or no treatment",
        outcome="glycemic control and cardiovascular events"
    )
    
    print("\n📋 PICO Research Question:")
    print(f"   P (Patient): {pico['patient']}")
    print(f"   I (Intervention): {pico['intervention']}")
    print(f"   C (Comparison): {pico['comparison']}")
    print(f"   O (Outcome): {pico['outcome']}")
    
    # Create initial state
    state = create_simple_initial_state(pico=pico)
    print(f"\n🆔 Workflow ID: {state['workflow_id']}")
    
    # Initialize decision engine
    decision_engine = SimpleOrchestratorDecisionEngine()
    
    # Run simulation
    print("\n🚀 Starting Orchestrator Decision Simulation...")
    print("-" * 60)
    
    max_steps = 15
    step = 0
    
    while step < max_steps:
        step += 1
        
        # Get next agent decision
        next_agent = decision_engine.determine_next_agent(state)
        rationale = decision_engine.get_decision_rationale(state, next_agent)
        
        # Display decision
        print(f"\n📍 Step {step}: {next_agent}")
        print(f"   {rationale}")
        
        # Check if workflow is complete
        if next_agent == "END":
            print("\n🏁 Workflow Simulation Completed Successfully!")
            break
        
        # Simulate agent execution
        simulate_agent_execution(next_agent, state)
        
        # Update step history
        state["step_history"].append(next_agent)
        state["current_agent"] = next_agent
    
    # Display final results
    print("\n" + "=" * 60)
    print("📊 FINAL WORKFLOW SUMMARY")
    print("=" * 60)
    
    print(f"🔢 Total Steps: {step}")
    print(f"📚 Papers Found: {state.get('total_papers_found', 0)}")
    print(f"📄 Papers Processed: {state.get('total_papers_processed', 0)}")
    print(f"🧠 Vector Chunks: {state.get('chunks_created', 0)}")
    print(f"⏱️  Started: {state.get('started_at', 'Unknown')}")
    print(f"✅ Completed: {state.get('completed_at', 'Unknown')}")
    
    print(f"\n📈 Step Sequence: {' → '.join(state['step_history'])}")
    
    if state.get("statistical_analysis"):
        analysis = state["statistical_analysis"]
        print(f"\n🔬 Statistical Results:")
        print(f"   Pooled Effect Size: {analysis['pooled_effect_size']}")
        print(f"   95% CI: {analysis['confidence_interval']}")
        print(f"   P-value: {analysis['p_value']}")
        print(f"   I² Heterogeneity: {analysis['heterogeneity_i2']}%")
        print(f"   Studies Included: {analysis['studies_included']}")
    
    print(f"\n📁 Final Report: {state.get('final_report_path', 'Not generated')}")
    
    return state


if __name__ == "__main__":
    try:
        final_state = run_orchestrator_simulation()
        print("\n✅ Simulation completed successfully!")
        
        # Optional: Display state details
        print(f"\n🔍 Final State Keys: {list(final_state.keys())}")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")
