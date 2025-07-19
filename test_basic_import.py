#!/usr/bin/env python3
"""
Basic test script for metanalyst-agent package
"""

import os
import sys

def test_import():
    """Test that the package can be imported successfully"""
    try:
        import metanalyst_agent
        print("✅ Package imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_settings():
    """Test that settings can be accessed"""
    try:
        from metanalyst_agent.config.settings import Settings
        settings = Settings()
        print("✅ Settings class works")
        print(f"   OpenAI model: {settings.openai_model}")
        print(f"   Vector dimension: {settings.vector_dimension}")
        print(f"   Quality threshold: {settings.default_quality_threshold}")
        return True
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

def test_tools_import():
    """Test that tools can be imported"""
    try:
        from metanalyst_agent.tools import (
            search_literature,
            generate_search_queries,
            extract_article_content,
            calculate_meta_analysis
        )
        print("✅ Tools import successfully")
        return True
    except ImportError as e:
        print(f"❌ Tools import failed: {e}")
        return False

def test_state_classes():
    """Test that state classes work"""
    try:
        from metanalyst_agent.state.meta_analysis_state import create_initial_state
        from metanalyst_agent.state.iteration_state import IterationState
        
        # Create initial state
        initial_state = create_initial_state(
            research_question="Test question",
            meta_analysis_id="test_id",
            thread_id="test_thread"
        )
        print("✅ State classes work")
        print(f"   Initial state type: {type(initial_state)}")
        return True
    except Exception as e:
        print(f"❌ State classes test failed: {e}")
        return False

def test_agent_creation_requires_api_keys():
    """Test that agent creation properly requires API keys"""
    try:
        from metanalyst_agent import MetanalystAgent
        
        # This should fail without API keys
        try:
            agent = MetanalystAgent()
            print("❌ Agent creation should have failed without API keys")
            return False
        except Exception as e:
            if "api_key" in str(e).lower():
                print("✅ Agent creation properly requires API keys")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing metanalyst-agent package...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_settings,
        test_tools_import,
        test_state_classes,
        test_agent_creation_requires_api_keys
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n🔬 Running {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"   Failed: {test.__name__}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The package is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())