#!/usr/bin/env python3
"""
Example usage of metanalyst-agent with API keys

This example demonstrates how to:
1. Set up API keys
2. Create a MetanalystAgent instance
3. Run a meta-analysis
4. Handle results and errors

Before running this example, you need to:
1. Get API keys from OpenAI and Tavily
2. Set them as environment variables or in a .env file
3. Optionally set up PostgreSQL/Redis/MongoDB databases
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_keys():
    """Check if required API keys are available"""
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these environment variables or add them to a .env file:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        print("   TAVILY_API_KEY=your_tavily_api_key_here")
        return False
    
    print("✅ All required API keys are available")
    return True

def example_basic_usage():
    """Example of basic MetanalystAgent usage"""
    print("\n🔬 Basic Usage Example")
    print("-" * 40)
    
    try:
        from metanalyst_agent import MetanalystAgent
        
        # Create agent instance
        print("Creating MetanalystAgent...")
        agent = MetanalystAgent(
            debug=True,
            use_postgres=False,  # Use in-memory storage for this example
            timeout=300  # 5 minutes timeout
        )
        
        print("✅ MetanalystAgent created successfully!")
        print(f"   Agent type: {type(agent)}")
        print(f"   Settings: OpenAI model = {agent.settings.openai_model}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Error creating MetanalystAgent: {e}")
        return None

async def example_meta_analysis(agent):
    """Example of running a meta-analysis"""
    print("\n📊 Meta-Analysis Example")
    print("-" * 40)
    
    # Example research question
    research_question = (
        "What is the effectiveness of cognitive behavioral therapy "
        "compared to medication for treating depression in adults?"
    )
    
    print(f"Research Question: {research_question}")
    
    try:
        # Run the meta-analysis
        print("\n🚀 Starting meta-analysis...")
        print("Note: This is a demo - the actual analysis would require API calls")
        
        # In a real scenario, you would call:
        # results = await agent.run_meta_analysis(
        #     research_question=research_question,
        #     max_articles=20,
        #     quality_threshold=0.7
        # )
        
        # For this example, we'll just show the structure
        mock_results = {
            "meta_analysis_id": "demo_analysis_001",
            "research_question": research_question,
            "status": "completed",
            "summary": {
                "total_articles_found": 25,
                "articles_included": 15,
                "quality_score": 0.85,
                "effect_size": 0.62,
                "confidence_interval": [0.45, 0.79],
                "heterogeneity": "moderate"
            },
            "pico": {
                "P": "Adults with depression",
                "I": "Cognitive behavioral therapy",
                "C": "Medication (antidepressants)",
                "O": "Depression symptom reduction"
            }
        }
        
        print("✅ Meta-analysis completed successfully!")
        print(f"   Analysis ID: {mock_results['meta_analysis_id']}")
        print(f"   Articles found: {mock_results['summary']['total_articles_found']}")
        print(f"   Articles included: {mock_results['summary']['articles_included']}")
        print(f"   Quality score: {mock_results['summary']['quality_score']}")
        print(f"   Effect size: {mock_results['summary']['effect_size']}")
        
        return mock_results
        
    except Exception as e:
        print(f"❌ Error running meta-analysis: {e}")
        return None

def example_tools_usage():
    """Example of using individual tools"""
    print("\n🛠️ Individual Tools Example")
    print("-" * 40)
    
    try:
        from metanalyst_agent.tools import (
            generate_search_queries,
            search_literature
        )
        from metanalyst_agent.state.meta_analysis_state import create_initial_state
        
        # Create a PICO framework
        pico = {
            "P": "Adults with type 2 diabetes",
            "I": "Metformin",
            "C": "Placebo or other antidiabetic drugs",
            "O": "HbA1c reduction"
        }
        
        print("PICO Framework:")
        for key, value in pico.items():
            print(f"   {key}: {value}")
        
        # Generate search queries (this would normally use LLM)
        print("\n📝 Generated search queries:")
        print("   (Note: Actual query generation requires OpenAI API)")
        mock_queries = [
            "metformin AND diabetes AND HbA1c",
            "metformin therapy diabetes mellitus type 2",
            "metformin effectiveness glycemic control"
        ]
        
        for i, query in enumerate(mock_queries, 1):
            print(f"   {i}. {query}")
        
        print("\n✅ Tools can be used individually")
        return True
        
    except Exception as e:
        print(f"❌ Error using tools: {e}")
        return False

def example_configuration():
    """Example of configuration options"""
    print("\n⚙️ Configuration Example")
    print("-" * 40)
    
    try:
        from metanalyst_agent.config.settings import Settings
        
        # Show current settings
        settings = Settings()
        
        print("Current Configuration:")
        print(f"   OpenAI Model: {settings.openai_model}")
        print(f"   Embedding Model: {settings.openai_embedding_model}")
        print(f"   Vector Dimension: {settings.vector_dimension}")
        print(f"   Quality Threshold: {settings.default_quality_threshold}")
        print(f"   Max Articles: {settings.default_max_articles}")
        print(f"   Max Iterations: {settings.default_max_iterations}")
        print(f"   Log Level: {settings.log_level}")
        
        # Show database configurations (if available)
        db_configs = settings.get_database_configs()
        print(f"\n🗄️ Database URLs configured:")
        for db_type, url in db_configs.items():
            # Hide sensitive parts of URLs
            if url and "://" in url:
                parts = url.split("://")
                if "@" in parts[1]:
                    user_part, host_part = parts[1].split("@", 1)
                    masked_url = f"{parts[0]}://***@{host_part}"
                else:
                    masked_url = url
            else:
                masked_url = url or "Not configured"
            print(f"   {db_type}: {masked_url}")
        
        print("\n✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

async def main():
    """Main example function"""
    print("🧬 Metanalyst-Agent Usage Examples")
    print("=" * 50)
    
    # Check API keys
    if not check_api_keys():
        print("\n⚠️  Cannot run examples without API keys.")
        print("Set up your API keys and try again.")
        return 1
    
    # Run configuration example
    if not example_configuration():
        return 1
    
    # Run tools example
    if not example_tools_usage():
        return 1
    
    # Run basic usage example
    agent = example_basic_usage()
    if not agent:
        return 1
    
    # Run meta-analysis example
    results = await example_meta_analysis(agent)
    if not results:
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 All examples completed successfully!")
    print("\nNext steps:")
    print("1. Set up your API keys in environment variables or .env file")
    print("2. Optionally configure databases for persistence")
    print("3. Run your own meta-analyses using the MetanalystAgent")
    print("4. Check the documentation for advanced usage patterns")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))