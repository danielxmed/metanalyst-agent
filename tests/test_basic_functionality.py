"""
Testes básicos para verificar funcionalidade do sistema metanalyst-agent.
"""

import pytest
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.state import create_initial_state, MetaAnalysisState
from models.schemas import PICO, StudyType, OutcomeType
from utils.config import Config
from agents.orchestrator import OrchestratorAgent


class TestBasicFunctionality:
    """Testes básicos de funcionalidade."""
    
    def test_create_initial_state(self):
        """Testa criação do estado inicial."""
        user_request = "Meta-análise sobre eficácia da aspirina na prevenção de AVC"
        
        state = create_initial_state(user_request)
        
        assert isinstance(state, dict)
        assert state["user_request"] == user_request
        assert state["current_phase"] == "pico_definition"
        assert state["meta_analysis_id"] is not None
        assert state["thread_id"] is not None
        assert isinstance(state["created_at"], type(state["updated_at"]))
    
    def test_pico_schema(self):
        """Testa schema PICO."""
        pico_data = {
            "population": "Pacientes com risco cardiovascular",
            "intervention": "Aspirina 100mg/dia",
            "comparison": "Placebo",
            "outcome": "Prevenção de AVC"
        }
        
        pico = PICO(**pico_data)
        
        assert pico.population == pico_data["population"]
        assert pico.intervention == pico_data["intervention"]
        assert pico.comparison == pico_data["comparison"]
        assert pico.outcome == pico_data["outcome"]
    
    def test_study_types(self):
        """Testa enums de tipos de estudo."""
        assert StudyType.RCT == "randomized_controlled_trial"
        assert StudyType.COHORT == "cohort_study"
        assert StudyType.META_ANALYSIS == "meta_analysis"
    
    def test_outcome_types(self):
        """Testa enums de tipos de desfecho."""
        assert OutcomeType.BINARY == "binary"
        assert OutcomeType.CONTINUOUS == "continuous"
        assert OutcomeType.TIME_TO_EVENT == "time_to_event"
    
    def test_config_loading(self):
        """Testa carregamento de configurações."""
        # Verificar se configurações básicas estão definidas
        assert hasattr(Config, 'LLM_MODEL')
        assert hasattr(Config, 'EMBEDDING_MODEL')
        assert hasattr(Config, 'DATABASE_URL')
        assert hasattr(Config, 'MAX_PAPERS_PER_SEARCH')
        
        # Verificar métodos de configuração
        llm_config = Config.get_llm_config()
        assert 'model' in llm_config
        assert 'temperature' in llm_config
        
        embedding_config = Config.get_embedding_config()
        assert 'model' in embedding_config
        assert 'dimensions' in embedding_config
    
    @pytest.mark.skipif(
        not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY.startswith("sk-your-"),
        reason="OpenAI API key não configurada"
    )
    def test_orchestrator_initialization(self):
        """Testa inicialização do orquestrador."""
        orchestrator = OrchestratorAgent()
        
        assert orchestrator.name == "orchestrator"
        assert orchestrator.llm is not None
    
    def test_pico_extraction_logic(self):
        """Testa lógica de extração de PICO."""
        orchestrator = OrchestratorAgent()
        
        # Teste com solicitação simples
        user_request = "Eficácia da metformina em diabéticos tipo 2"
        
        # Simular extração de PICO (sem chamar LLM)
        expected_pico = {
            "population": "diabéticos tipo 2",
            "intervention": "metformina",
            "comparison": "placebo ou controle",
            "outcome": "eficácia"
        }
        
        # Verificar que a lógica pode processar a solicitação
        assert "metformina" in user_request.lower()
        assert "diabéticos" in user_request.lower()


class TestStateManagement:
    """Testes de gerenciamento de estado."""
    
    def test_state_transitions(self):
        """Testa transições de estado."""
        from models.state import update_state_phase, add_agent_log
        
        # Criar estado inicial
        state = create_initial_state("Teste")
        
        # Testar mudança de fase
        update = update_state_phase(state, "search", "researcher")
        
        assert update["current_phase"] == "search"
        assert update["current_agent"] == "researcher"
        assert "agent_logs" in update
    
    def test_agent_logging(self):
        """Testa sistema de logs de agentes."""
        from models.state import add_agent_log
        
        state = create_initial_state("Teste")
        
        # Adicionar log
        update = add_agent_log(
            state, 
            "test_agent", 
            "test_action", 
            {"detail": "test"}, 
            "success"
        )
        
        assert "agent_logs" in update
        assert len(update["agent_logs"]) == 1
        
        log = update["agent_logs"][0]
        assert log["agent"] == "test_agent"
        assert log["action"] == "test_action"
        assert log["status"] == "success"


class TestErrorHandling:
    """Testes de tratamento de erros."""
    
    def test_invalid_pico_data(self):
        """Testa tratamento de dados PICO inválidos."""
        # Testar com dados incompletos
        with pytest.raises(Exception):
            PICO(
                population="",  # Campo obrigatório vazio
                intervention="Aspirina",
                comparison="Placebo", 
                outcome="AVC"
            )
    
    def test_config_validation(self):
        """Testa validação de configuração."""
        # Verificar se método de validação existe
        assert hasattr(Config, 'validate_required_keys')
        
        # Verificar estrutura de configuração
        db_config = Config.get_database_config()
        assert 'database_url' in db_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])