"""Basic functionality tests for Metanalyst-Agent"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from metanalyst_agent.config.settings import Settings
from metanalyst_agent.state.meta_analysis_state import create_initial_state
from metanalyst_agent.graph.meta_analysis_graph import MetaAnalysisGraph


class TestBasicFunctionality:
    """Test basic functionality of the Metanalyst-Agent system"""
    
    def test_settings_initialization(self):
        """Test that settings can be initialized"""
        # Test with minimal required settings
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'TAVILY_API_KEY': 'test_key'
        }):
            settings = Settings()
            assert settings.openai_api_key == 'test_key'
            assert settings.tavily_api_key == 'test_key'
            assert settings.openai_model == 'gpt-4o'
    
    def test_initial_state_creation(self):
        """Test that initial state can be created"""
        state = create_initial_state(
            research_question="Test question",
            config={"test": "value"},
            max_iterations=5,
            quality_threshold=0.8
        )
        
        assert state["research_question"] == "Test question"
        assert state["config"]["test"] == "value"
        assert state["max_global_iterations"] == 5
        assert state["quality_thresholds"]["overall"] == 0.8
        assert state["current_phase"] == "initialization"
    
    def test_graph_initialization(self):
        """Test that the graph can be initialized"""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key',
            'TAVILY_API_KEY': 'test_key'
        }):
            settings = Settings()
            graph = MetaAnalysisGraph(settings)
            
            assert graph.graph is not None
            assert graph.checkpointer is not None
            assert graph.store is not None
    
    @pytest.mark.asyncio
    async def test_graph_execution_structure(self):
        """Test that graph execution structure is correct (without actual API calls)"""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key', 
            'TAVILY_API_KEY': 'test_key'
        }):
            settings = Settings()
            graph = MetaAnalysisGraph(settings)
            
            # Mock the actual execution to avoid API calls
            with patch.object(graph, '_execute_with_monitoring') as mock_execute:
                mock_execute.return_value = create_initial_state("test", {})
                
                result = await graph.execute(
                    query="Test query",
                    max_articles=5,
                    quality_threshold=0.8,
                    max_time_minutes=1
                )
                
                assert "success" in result
                assert "meta_analysis_id" in result
                assert "research_question" in result
    
    def test_state_reducers(self):
        """Test state reducer functions"""
        from metanalyst_agent.state.reducers import add_messages, merge_dicts, append_list
        
        # Test add_messages
        existing = []
        new = [Mock(content="test message")]
        result = add_messages(existing, new)
        assert len(result) == 1
        
        # Test merge_dicts
        existing = {"a": 1, "nested": {"b": 2}}
        new = {"a": 2, "nested": {"c": 3}}
        result = merge_dicts(existing, new)
        assert result["a"] == 2
        assert result["nested"]["b"] == 2
        assert result["nested"]["c"] == 3
        
        # Test append_list
        existing = [1, 2]
        new = [3, 4]
        result = append_list(existing, new)
        assert result == [1, 2, 3, 4]
    
    def test_agent_creation(self):
        """Test that agents can be created"""
        from metanalyst_agent.agents.supervisor_agent import create_supervisor_agent
        from metanalyst_agent.agents.researcher_agent import create_researcher_agent
        
        settings_dict = {
            "openai_api_key": "test_key",
            "openai_model": "gpt-4o",
            "tavily_api_key": "test_key"
        }
        
        # Test supervisor agent creation
        supervisor = create_supervisor_agent(settings_dict)
        assert supervisor is not None
        
        # Test researcher agent creation
        researcher = create_researcher_agent(settings_dict)
        assert researcher is not None
    
    def test_tool_imports(self):
        """Test that all tools can be imported"""
        from metanalyst_agent.tools.research_tools import (
            search_literature, generate_search_queries, assess_article_relevance
        )
        from metanalyst_agent.tools.processing_tools import (
            extract_article_content, extract_statistical_data, generate_vancouver_citation
        )
        from metanalyst_agent.tools.analysis_tools import (
            calculate_meta_analysis, create_forest_plot, assess_heterogeneity
        )
        from metanalyst_agent.tools.handoff_tools import (
            transfer_to_researcher, transfer_to_processor, transfer_to_analyst
        )
        
        # All imports should succeed
        assert search_literature is not None
        assert extract_article_content is not None
        assert calculate_meta_analysis is not None
        assert transfer_to_researcher is not None


if __name__ == "__main__":
    pytest.main([__file__])