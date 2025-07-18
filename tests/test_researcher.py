"""
Quick test for the researcher agent with Tavily integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_basic_setup():
    """Test basic setup and imports"""
    print("🧪 Testing basic setup...")
    
    # Test imports
    try:
        from src.utils.config import get_config, validate_environment
        from src.tools.tavily_tools import get_tavily_client, TAVILY_TOOLS
        from src.agents.researcher import create_researcher_agent
        from src.models.schemas import create_pico, validate_pico
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test environment
    try:
        if not validate_environment():
            print("❌ Environment validation failed")
            return False
        print("✅ Environment validation passed")
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False
    
    # Test configuration
    try:
        config = get_config()
        print(f"✅ Configuration loaded")
        print(f"   • Max papers: {config.search.max_papers_per_search}")
        print(f"   • Embedding model: {config.vector.embedding_model}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Test Tavily client
    try:
        tavily_client = get_tavily_client()
        if tavily_client:
            print("✅ Tavily client available")
        else:
            print("⚠️ Tavily client not available - check API key")
    except Exception as e:
        print(f"❌ Tavily error: {e}")
    
    # Test PICO creation
    try:
        pico = create_pico(
            patient="adults with diabetes",
            intervention="metformin",
            comparison="placebo", 
            outcome="glycemic control"
        )
        
        pico_dict = {
            "patient": pico["patient"],
            "intervention": pico["intervention"],
            "comparison": pico["comparison"],
            "outcome": pico["outcome"]
        }
        
        if validate_pico(pico_dict):
            print("✅ PICO creation and validation successful")
        else:
            print("❌ PICO validation failed")
    except Exception as e:
        print(f"❌ PICO error: {e}")
        return False
    
    # Test researcher agent creation
    try:
        researcher = create_researcher_agent()
        print("✅ Researcher agent created successfully")
        print(f"   • Tools available: {len(TAVILY_TOOLS)}")
    except Exception as e:
        print(f"❌ Researcher agent error: {e}")
        return False
    
    return True


def test_simple_search():
    """Test a simple search operation"""
    print("\n🔍 Testing simple search...")
    
    try:
        from src.tools.tavily_tools import search_medical_literature
        from src.utils.config import get_config
        
        config = get_config()
        
        # Test search tool
        result_json = search_medical_literature.invoke({
            "query": "diabetes metformin effectiveness",
            "max_results": 5,
            "search_depth": "basic"
        })
        
        import json
        result = json.loads(result_json)
        
        if result.get("success"):
            print(f"✅ Search successful")
            print(f"   • Results found: {result.get('total_results', 0)}")
            print(f"   • URLs: {len(result.get('urls_found', []))}")
        else:
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Search test error: {e}")


if __name__ == "__main__":
    print("🚀 Testing Metanalyst Agent Researcher")
    print("=" * 50)
    
    if test_basic_setup():
        test_simple_search()
        print("\n🎉 Testing completed!")
    else:
        print("\n❌ Basic setup failed - skipping advanced tests")
