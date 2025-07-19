#!/usr/bin/env python3
"""
Example usage of the Metanalyst-Agent system.
Demonstrates how to run automated meta-analyses with different configurations.
"""

import asyncio
import os
from datetime import datetime

# Import the main agent
from metanalyst_agent import MetanalystAgent, run_meta_analysis


def example_basic_usage():
    """
    Example 1: Basic usage with the convenience function.
    This is the simplest way to run a meta-analysis.
    """
    
    print("🔬 Example 1: Basic Meta-Analysis")
    print("=" * 50)
    
    # Define the research question
    query = """
    Systematic review and meta-analysis comparing the effectiveness of 
    mindfulness-based stress reduction (MBSR) versus waitlist control 
    for reducing anxiety symptoms in adults with generalized anxiety disorder.
    Include only randomized controlled trials published in the last 10 years.
    """
    
    try:
        # Run the meta-analysis with default settings
        results = run_meta_analysis(
            query=query,
            max_articles=20,  # Process up to 20 articles
            quality_threshold=0.8,  # Require 80% quality score
            debug=True  # Enable debug output
        )
        
        # Display results
        print(f"✅ Status: {results['status']}")
        print(f"📊 Articles found: {results['execution_summary']['articles_found']}")
        print(f"📚 Articles processed: {results['execution_summary']['articles_processed']}")
        print(f"🎯 Final phase: {results['execution_summary']['final_phase']}")
        
        if results.get('final_report'):
            print("📋 Final report generated successfully!")
            # Save the report
            with open(f"report_{results['meta_analysis_id'][:8]}.html", "w") as f:
                f.write(results['final_report'])
            print("💾 Report saved as HTML file")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


async def example_streaming():
    """
    Example 2: Streaming real-time progress updates.
    This shows how to monitor the meta-analysis as it runs.
    """
    
    print("\n🔬 Example 2: Streaming Meta-Analysis")
    print("=" * 50)
    
    query = """
    Meta-analysis of cognitive behavioral therapy versus pharmacotherapy 
    for treatment of major depressive disorder in adults. Include forest plots
    and heterogeneity analysis.
    """
    
    # Create agent instance
    agent = MetanalystAgent(debug=True)
    
    try:
        # Stream progress updates
        async for progress in agent.stream(
            query=query,
            max_articles=15,
            quality_threshold=0.85
        ):
            # Display real-time updates
            print(f"[{progress['timestamp'][:19]}] "
                  f"Phase: {progress['phase']} | "
                  f"Agent: {progress['agent']} | "
                  f"Articles: {progress['articles_found']}/{progress['articles_processed']}")
            
            if progress.get('latest_message'):
                print(f"  💬 {progress['latest_message'][:100]}...")
            
            # Check for completion
            if progress['status'] in ['completed', 'stopped']:
                print(f"🏁 Final status: {progress['status']}")
                break
                
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
    
    finally:
        agent.__exit__(None, None, None)


def example_advanced_configuration():
    """
    Example 3: Advanced configuration with custom parameters.
    Shows how to use the full MetanalystAgent class with custom settings.
    """
    
    print("\n🔬 Example 3: Advanced Configuration")
    print("=" * 50)
    
    query = """
    Systematic review comparing effectiveness of telemedicine versus 
    in-person consultations for managing chronic diseases. Focus on 
    patient satisfaction and clinical outcomes.
    """
    
    # Create agent with custom configuration
    with MetanalystAgent(
        use_persistent_storage=False,  # Use in-memory storage for this example
        debug=True
    ) as agent:
        
        try:
            # Run with custom parameters
            results = agent.run(
                query=query,
                max_articles=25,
                quality_threshold=0.75,
                max_time_minutes=20,  # Shorter timeout for example
                # Custom parameters
                chunk_size=800,  # Smaller chunks
                chunk_overlap=150,  # More overlap
                max_iterations=3,  # Fewer iterations
            )
            
            print(f"✅ Status: {results['status']}")
            
            # Display detailed execution summary
            summary = results['execution_summary']
            print(f"📈 Execution Summary:")
            print(f"  • Articles found: {summary['articles_found']}")
            print(f"  • Articles processed: {summary['articles_processed']}")
            print(f"  • Articles failed: {summary['articles_failed']}")
            print(f"  • Total iterations: {summary['total_iterations']}")
            
            # Display quality scores by agent
            if summary['quality_scores']:
                print(f"🎯 Quality Scores:")
                for agent, score in summary['quality_scores'].items():
                    print(f"  • {agent}: {score:.2f}")
            
            # Display statistical analysis if available
            if results.get('statistical_analysis'):
                stats = results['statistical_analysis']
                print(f"📊 Statistical Analysis:")
                if stats.get('pooled_effect'):
                    effect = stats['pooled_effect']
                    print(f"  • Effect size: {effect.get('effect_size', 'N/A')}")
                    print(f"  • P-value: {effect.get('p_value', 'N/A')}")
                
                if stats.get('heterogeneity'):
                    het = stats['heterogeneity']
                    print(f"  • I²: {het.get('i_squared', 'N/A')}%")
                    print(f"  • Heterogeneity: {het.get('i_squared_interpretation', 'N/A')}")
            
            return results
            
        except Exception as e:
            print(f"❌ Advanced configuration error: {str(e)}")
            return None


def example_error_handling():
    """
    Example 4: Error handling and recovery scenarios.
    Shows how the system handles various error conditions.
    """
    
    print("\n🔬 Example 4: Error Handling")
    print("=" * 50)
    
    # Test with a very broad query that might cause issues
    problematic_query = """
    Meta-analysis of everything related to health interventions.
    Include all study types and all populations.
    """
    
    try:
        results = run_meta_analysis(
            query=problematic_query,
            max_articles=5,  # Very low limit
            quality_threshold=0.95,  # Very high threshold
            debug=True
        )
        
        if results.get('status') == 'partial':
            print("⚠️  Partial results obtained due to constraints")
            print(f"📊 What we got: {results['execution_summary']}")
        
        elif results.get('error'):
            print(f"❌ Error encountered: {results['error']}")
            print("💡 This demonstrates the system's error handling capabilities")
        
        else:
            print("✅ Surprisingly, the problematic query worked!")
        
        return results
        
    except Exception as e:
        print(f"❌ Exception caught: {str(e)}")
        print("💡 This shows how exceptions are handled gracefully")
        return None


def save_results_to_file(results, filename_prefix="metanalysis"):
    """
    Helper function to save results to files.
    """
    
    if not results:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = results.get('meta_analysis_id', 'unknown')[:8]
    
    # Save HTML report if available
    if results.get('final_report'):
        html_filename = f"{filename_prefix}_{analysis_id}_{timestamp}.html"
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(results['final_report'])
        print(f"💾 HTML report saved: {html_filename}")
    
    # Save JSON summary
    import json
    json_filename = f"{filename_prefix}_summary_{analysis_id}_{timestamp}.json"
    
    # Create a summary without the full HTML report (too large for JSON)
    summary = {
        "meta_analysis_id": results.get('meta_analysis_id'),
        "status": results.get('status'),
        "pico": results.get('pico'),
        "execution_summary": results.get('execution_summary'),
        "statistical_analysis": results.get('statistical_analysis'),
        "quality_assessments": results.get('quality_assessments'),
        "created_at": results.get('created_at'),
        "updated_at": results.get('updated_at'),
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"💾 JSON summary saved: {json_filename}")


async def main():
    """
    Main function that runs all examples.
    """
    
    print("🚀 Metanalyst-Agent Examples")
    print("=" * 80)
    print("This script demonstrates various ways to use the Metanalyst-Agent system.")
    print("=" * 80)
    
    # Check if API keys are set
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('TAVILY_API_KEY'):
        print("⚠️  WARNING: API keys not found in environment variables.")
        print("   Please set OPENAI_API_KEY and TAVILY_API_KEY before running.")
        print("   The examples will likely fail without proper API keys.")
        print()
    
    # Run examples
    try:
        # Example 1: Basic usage
        results1 = example_basic_usage()
        if results1:
            save_results_to_file(results1, "basic_example")
        
        # Example 2: Streaming (async)
        await example_streaming()
        
        # Example 3: Advanced configuration
        results3 = example_advanced_configuration()
        if results3:
            save_results_to_file(results3, "advanced_example")
        
        # Example 4: Error handling
        results4 = example_error_handling()
        if results4:
            save_results_to_file(results4, "error_example")
        
        print("\n" + "=" * 80)
        print("🎉 All examples completed!")
        print("📁 Check the current directory for generated report files.")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error in examples: {str(e)}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())