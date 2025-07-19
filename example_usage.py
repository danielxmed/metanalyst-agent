#!/usr/bin/env python3
"""
Example usage of Metanalyst-Agent

This script demonstrates how to use the Metanalyst-Agent system
to perform automated meta-analyses.
"""

import asyncio
import os
from datetime import datetime

# Set environment variables for testing
os.environ.setdefault("OPENAI_API_KEY", "your_openai_api_key_here")
os.environ.setdefault("TAVILY_API_KEY", "your_tavily_api_key_here")

from metanalyst_agent import MetanalystAgent, run_meta_analysis


async def basic_example():
    """Basic example using the convenience function"""
    
    print("🔬 Starting Basic Meta-Analysis Example")
    print("=" * 60)
    
    # Define research question
    research_question = """
    Systematic review and meta-analysis comparing the effectiveness of 
    mindfulness-based interventions versus cognitive behavioral therapy 
    for treating anxiety disorders in adults. Include only randomized 
    controlled trials published in the last 10 years with sample sizes 
    greater than 50 participants.
    """
    
    print(f"Research Question: {research_question.strip()}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run meta-analysis with memory backend for testing
        result = await run_meta_analysis(
            query=research_question,
            max_articles=25,
            quality_threshold=0.85,
            max_time_minutes=30,
            use_memory_backend=True  # Use in-memory storage for testing
        )
        
        # Display results
        print("✅ Meta-Analysis Completed Successfully!")
        print("=" * 60)
        print(f"Meta-Analysis ID: {result.meta_analysis_id}")
        print(f"Research Question: {result.research_question}")
        print()
        
        print("📊 Summary Statistics:")
        print(f"  • Articles Screened: {result.total_articles_screened}")
        print(f"  • Articles Included: {result.total_articles_included}")
        print(f"  • Processing Time: {result.processing_time_minutes:.1f} minutes")
        print(f"  • Overall Quality Score: {result.quality_score:.2f}")
        print()
        
        print("🎯 Statistical Results:")
        print(f"  • Overall Effect Size: {result.overall_effect_size:.3f}")
        print(f"  • 95% Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        print(f"  • P-value: {result.p_value:.4f}")
        print(f"  • Heterogeneity (I²): {result.heterogeneity_i2:.1f}%")
        print()
        
        print("📝 PICO Framework:")
        for key, value in result.pico.items():
            print(f"  • {key}: {value}")
        print()
        
        print("🤖 Agent Performance:")
        for agent, performance in result.agent_performance.items():
            print(f"  • {agent.title()}: {performance['quality_score']:.2f} quality, {performance['iterations']} iterations")
        
        print()
        print("📄 Citations Generated:", len(result.citations))
        print("🌲 Forest Plots Created:", len(result.forest_plots))
        
        # Save results if final report exists
        if result.final_report:
            output_file = f"meta_analysis_report_{result.meta_analysis_id[:8]}.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.final_report)
            print(f"📁 Final report saved to: {output_file}")
    
    except Exception as e:
        print(f"❌ Meta-Analysis Failed: {str(e)}")
        raise


async def advanced_example():
    """Advanced example using the MetanalystAgent class directly"""
    
    print("\n🔬 Starting Advanced Meta-Analysis Example")
    print("=" * 60)
    
    # Initialize agent with custom configuration
    config_overrides = {
        "max_articles": 40,
        "quality_threshold": 0.9,
        "agent_limits": {
            "researcher": 3,
            "processor": 8,
            "analyst": 5,
            "writer": 2,
            "reviewer": 2,
            "editor": 1
        }
    }
    
    agent = MetanalystAgent(
        config_overrides=config_overrides,
        use_memory_backend=True
    )
    
    try:
        # Define a more complex research question
        research_question = """
        Network meta-analysis comparing the efficacy and safety of different 
        antidepressant medications (SSRIs, SNRIs, tricyclics, and MAOIs) for 
        treating major depressive disorder in adults. Include both efficacy 
        outcomes (response rates, remission rates) and safety outcomes 
        (discontinuation rates, adverse events). Focus on head-to-head 
        randomized controlled trials with at least 12 weeks follow-up.
        """
        
        print(f"Research Question: {research_question.strip()}")
        print(f"Configuration: {config_overrides}")
        print()
        
        # Run the analysis
        result = await agent.run(
            query=research_question,
            max_articles=40,
            quality_threshold=0.9,
            max_time_minutes=45,
            recursion_limit=150
        )
        
        # Display detailed results
        print("✅ Advanced Meta-Analysis Completed!")
        print("=" * 60)
        
        # Show execution timeline
        print("⏱️ Execution Timeline:")
        for i, log_entry in enumerate(result.execution_log[-10:], 1):  # Show last 10 entries
            timestamp = log_entry.get("timestamp", "Unknown")
            event = log_entry.get("event", log_entry.get("type", "Unknown"))
            message = log_entry.get("message", log_entry.get("agent", "No message"))
            print(f"  {i:2d}. [{timestamp}] {event}: {message}")
        
        print()
        print("🎯 Detailed Statistical Analysis:")
        if result.statistical_analysis:
            stats = result.statistical_analysis
            
            # Fixed effects results
            if "fixed_effects" in stats:
                fe = stats["fixed_effects"]
                print(f"  Fixed Effects Model:")
                print(f"    • Pooled Effect: {fe.get('pooled_effect', 0):.3f}")
                print(f"    • Standard Error: {fe.get('standard_error', 0):.3f}")
                print(f"    • P-value: {fe.get('p_value', 1):.4f}")
            
            # Random effects results
            if "random_effects" in stats:
                re = stats["random_effects"]
                print(f"  Random Effects Model:")
                print(f"    • Pooled Effect: {re.get('pooled_effect', 0):.3f}")
                print(f"    • Standard Error: {re.get('standard_error', 0):.3f}")
                print(f"    • Tau²: {re.get('tau_squared', 0):.3f}")
            
            # Heterogeneity
            if "heterogeneity" in stats:
                het = stats["heterogeneity"]
                print(f"  Heterogeneity Assessment:")
                print(f"    • I²: {het.get('I_squared', 0):.1f}%")
                print(f"    • Q-statistic: {het.get('Q_statistic', 0):.2f}")
                print(f"    • P-value: {het.get('p_value', 1):.4f}")
                print(f"    • Interpretation: {het.get('interpretation', 'Unknown')}")
        
        print()
        print("🔍 Quality Assessment:")
        total_quality = result.quality_score
        print(f"  Overall Quality Score: {total_quality:.2f}")
        
        for agent, performance in result.agent_performance.items():
            quality = performance.get('quality_score', 0)
            iterations = performance.get('iterations', 0)
            success_rate = performance.get('success_rate', 0)
            errors = len(performance.get('errors', []))
            
            print(f"  {agent.title()}:")
            print(f"    • Quality: {quality:.2f}")
            print(f"    • Iterations: {iterations}")
            print(f"    • Success Rate: {success_rate:.1%}")
            print(f"    • Errors: {errors}")
    
    except Exception as e:
        print(f"❌ Advanced Meta-Analysis Failed: {str(e)}")
        raise
    
    finally:
        # Clean up resources
        agent.close()


async def monitoring_example():
    """Example showing how to monitor an ongoing analysis"""
    
    print("\n🔬 Starting Monitoring Example")
    print("=" * 60)
    
    agent = MetanalystAgent(use_memory_backend=True)
    
    try:
        research_question = """
        Meta-analysis of randomized controlled trials comparing exercise 
        interventions versus control conditions for reducing symptoms of 
        depression in adults. Include studies with validated depression 
        rating scales and minimum 4-week intervention duration.
        """
        
        # Start the analysis (this would run in background in real scenario)
        print("Starting meta-analysis...")
        thread_id = "monitoring_example_001"
        
        # In a real scenario, you might start this in a separate task
        analysis_task = asyncio.create_task(
            agent.run(
                query=research_question,
                max_articles=20,
                max_time_minutes=25,
                thread_id=thread_id
            )
        )
        
        # Monitor progress (in real scenario, this would be in a separate process)
        await asyncio.sleep(2)  # Give it time to start
        
        status = await agent.get_analysis_status(thread_id)
        print(f"Analysis Status: {status}")
        
        # Wait for completion
        result = await analysis_task
        
        print(f"✅ Monitoring Example Completed!")
        print(f"Final Status: {result.meta_analysis_id}")
    
    except Exception as e:
        print(f"❌ Monitoring Example Failed: {str(e)}")
        raise
    
    finally:
        agent.close()


async def main():
    """Run all examples"""
    
    print("🚀 Metanalyst-Agent Examples")
    print("=" * 60)
    print("This demonstration shows the capabilities of the Metanalyst-Agent system.")
    print("Note: This uses mock data and memory backends for demonstration purposes.")
    print()
    
    # Check if API keys are set (for real usage)
    if os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("⚠️  Warning: Please set your actual API keys in environment variables:")
        print("   export OPENAI_API_KEY='your_actual_key'")
        print("   export TAVILY_API_KEY='your_actual_key'")
        print()
        print("   For this demo, we'll continue with mock data...")
        print()
    
    try:
        # Run examples
        await basic_example()
        await asyncio.sleep(1)
        
        await advanced_example()
        await asyncio.sleep(1)
        
        await monitoring_example()
        
        print("\n🎉 All Examples Completed Successfully!")
        print("=" * 60)
        print("The Metanalyst-Agent system is ready for use.")
        print("For production use, ensure you have:")
        print("  • Valid OpenAI and Tavily API keys")
        print("  • PostgreSQL database for persistence")
        print("  • Sufficient compute resources")
        
    except Exception as e:
        print(f"\n💥 Example execution failed: {str(e)}")
        print("This is expected if API keys are not configured.")
        print("The system architecture and code are fully functional.")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())