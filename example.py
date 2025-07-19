#!/usr/bin/env python3
"""
Example usage of the Metanalyst-Agent system

This script demonstrates how to use the metanalyst-agent for automated meta-analysis
generation from a research question to final report with statistical analysis.
"""

import asyncio
import os
from metanalyst_agent import MetanalystAgent


async def main():
    """Main example function"""
    
    # Ensure API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable not set")
        print("Please set your Tavily API key: export TAVILY_API_KEY=your_key_here")
        return
    
    print("🔬 Metanalyst-Agent Example")
    print("=" * 50)
    
    # Initialize the agent
    print("Initializing Metanalyst-Agent...")
    agent = MetanalystAgent()
    
    # Define research question
    research_question = """
    Perform a meta-analysis comparing the effectiveness of mindfulness-based interventions 
    versus cognitive behavioral therapy for treating anxiety disorders in adults. 
    Include forest plots, heterogeneity assessment, and publication bias evaluation.
    """
    
    print(f"Research Question: {research_question.strip()}")
    print("\n🚀 Starting meta-analysis execution...")
    print("This may take several minutes depending on the complexity...")
    
    try:
        # Execute the meta-analysis
        results = await agent.run(
            query=research_question,
            max_articles=20,  # Limit for example
            quality_threshold=0.8,
            max_time_minutes=15  # Shorter time for example
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 META-ANALYSIS RESULTS")
        print("=" * 50)
        
        if results.get("success"):
            print("✅ Meta-analysis completed successfully!")
            
            # Basic information
            print(f"\n📋 Analysis ID: {results.get('meta_analysis_id')}")
            print(f"📋 Research Question: {results.get('research_question')}")
            print(f"📋 Current Phase: {results.get('current_phase')}")
            
            # Execution summary
            summary = results.get("execution_summary", {})
            print(f"\n📈 Execution Summary:")
            print(f"   • Articles Processed: {summary.get('total_articles_processed', 0)}")
            print(f"   • Articles Failed: {summary.get('total_articles_failed', 0)}")
            print(f"   • Global Iterations: {summary.get('global_iterations', 0)}")
            
            # PICO Framework
            pico = results.get("pico_framework", {})
            if pico:
                print(f"\n🎯 PICO Framework:")
                for key, value in pico.items():
                    print(f"   • {key}: {value}")
            
            # Search results
            search_results = results.get("search_results", {})
            print(f"\n🔍 Search Results:")
            print(f"   • Queries Used: {len(search_results.get('queries_used', []))}")
            print(f"   • Candidate URLs: {len(search_results.get('candidate_urls', []))}")
            print(f"   • Processed Articles: {len(search_results.get('processed_articles', []))}")
            
            # Statistical analysis
            stats = results.get("statistical_analysis", {})
            if stats:
                print(f"\n📊 Statistical Analysis:")
                print(f"   • Analysis Available: {'Yes' if stats else 'No'}")
                if "pooled_effect_random" in stats:
                    effect = stats["pooled_effect_random"]
                    print(f"   • Pooled Effect: {effect.get('estimate', 'N/A')}")
                    print(f"   • Confidence Interval: {effect.get('confidence_interval', 'N/A')}")
                    print(f"   • P-value: {effect.get('p_value', 'N/A')}")
            
            # Quality assessment
            quality = results.get("quality_assessment", {})
            print(f"\n⭐ Quality Assessment:")
            print(f"   • Overall Quality: {quality.get('overall_quality', 0):.2f}")
            print(f"   • Quality Satisfied: {quality.get('quality_satisfied', False)}")
            
            # Recent messages
            messages = results.get("messages", [])
            if messages:
                print(f"\n💬 Recent System Messages:")
                for msg in messages[-3:]:  # Show last 3 messages
                    role = msg.get("role", "system")
                    content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                    print(f"   • [{role}]: {content}")
            
            # Final report
            final_report = results.get("final_report")
            if final_report:
                print(f"\n📄 Final Report Available: Yes ({len(final_report)} characters)")
                print("   (Report content would be displayed or saved to file)")
            else:
                print(f"\n📄 Final Report Available: No")
            
        else:
            print("❌ Meta-analysis failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
            # Show partial results if available
            partial = results.get("partial_results", {})
            if partial:
                print(f"\n📋 Partial Results Available:")
                print(f"   • Articles Processed: {partial.get('execution_summary', {}).get('total_articles_processed', 0)}")
                print(f"   • Current Phase: {partial.get('current_phase', 'Unknown')}")
    
    except Exception as e:
        print(f"\n❌ Error executing meta-analysis: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Example completed!")
    print("For production use, consider:")
    print("  • Using PostgreSQL/Redis for persistence")
    print("  • Implementing proper error handling")
    print("  • Adding logging and monitoring")
    print("  • Configuring appropriate timeouts")


if __name__ == "__main__":
    # Run the async example
    asyncio.run(main())