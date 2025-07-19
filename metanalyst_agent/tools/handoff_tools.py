"""
Ferramentas de handoff para transferência de controle entre agentes autônomos.
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from ..models.state import MetaAnalysisState

def create_handoff_tool(*, agent_name: str, description: str):
    """Factory para criar ferramentas de handoff entre agentes"""
    
    tool_name = f"transfer_to_{agent_name}"
    
    @tool(tool_name, description=description)
    def handoff_tool(
        reason: Annotated[str, "Razão detalhada para transferir controle para este agente"],
        context: Annotated[str, "Contexto e informações relevantes para o próximo agente"],
        next_actions: Annotated[str, "Ações específicas que o próximo agente deve realizar"],
        state: Annotated[MetaAnalysisState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Transfere controle para outro agente com contexto detalhado"""
        
        from datetime import datetime
        
        # Criar mensagem de tool result
        tool_message = ToolMessage(
            content=f"✅ Controle transferido para {agent_name}.\n\n"
                   f"**Razão:** {reason}\n\n"
                   f"**Contexto:** {context}\n\n"
                   f"**Próximas ações:** {next_actions}",
            tool_call_id=tool_call_id,
            name=tool_name
        )
        
        # Registrar transição
        transition = {
            "from_agent": state.get("current_agent", "unknown"),
            "to_agent": agent_name,
            "reason": reason,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        # Atualizar estado
        transitions = state.get("agent_transitions", [])
        transitions.append(transition)
        
        return Command(
            goto=agent_name,
            update={
                "messages": state["messages"] + [tool_message],
                "current_agent": agent_name,
                "last_agent": state.get("current_agent"),
                "agent_transitions": transitions,
                "updated_at": datetime.now().isoformat(),
                # Limpar flag de erro se existir
                "error_flag": False,
                "error_details": None
            },
            graph=Command.PARENT
        )
    
    return handoff_tool

# Criar todas as ferramentas de handoff específicas

transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description=(
        "Transferir controle para o Research Agent quando precisar de busca de literatura médica. "
        "Use quando: precisar de mais artigos, definir queries de busca, ou avaliar relevância de estudos."
    )
)

transfer_to_processor = create_handoff_tool(
    agent_name="processor", 
    description=(
        "Transferir controle para o Processor Agent quando tiver URLs de artigos para processar. "
        "Use quando: tiver lista de URLs, precisar extrair conteúdo ou dados estatísticos."
    )
)

transfer_to_vectorizer = create_handoff_tool(
    agent_name="vectorizer",
    description=(
        "Transferir controle para o Vectorizer Agent quando tiver conteúdo processado para vetorizar. "
        "Use quando: artigos foram processados e precisam ser chunked/embedded."
    )
)

transfer_to_retriever = create_handoff_tool(
    agent_name="retriever",
    description=(
        "Transferir controle para o Retriever Agent quando precisar buscar informações específicas. "
        "Use quando: o vector store estiver pronto e precisar recuperar informações relevantes."
    )
)

transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description=(
        "Transferir controle para o Analyst Agent quando tiver dados estatísticos para analisar. "
        "Use quando: dados extraídos estão prontos para meta-análise e visualizações."
    )
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer",
    description=(
        "Transferir controle para o Writer Agent quando a análise estiver completa. "
        "Use quando: resultados estatísticos estão prontos para geração do relatório."
    )
)

transfer_to_reviewer = create_handoff_tool(
    agent_name="reviewer", 
    description=(
        "Transferir controle para o Reviewer Agent quando tiver um relatório para revisar. "
        "Use quando: draft do relatório está pronto e precisa de revisão de qualidade."
    )
)

transfer_to_editor = create_handoff_tool(
    agent_name="editor",
    description=(
        "Transferir controle para o Editor Agent para finalização do documento. "
        "Use quando: relatório foi revisado e precisa de formatação final."
    )
)

# Ferramenta especial para retorno ao supervisor
@tool("return_to_supervisor")
def return_to_supervisor(
    completion_status: Annotated[str, "Status de conclusão da tarefa (completed, needs_help, error)"],
    summary: Annotated[str, "Resumo do que foi realizado"],
    next_steps: Annotated[str, "Sugestões para próximos passos ou necessidades"],
    state: Annotated[MetaAnalysisState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retorna controle ao supervisor com status de conclusão"""
    
    from datetime import datetime
    
    tool_message = ToolMessage(
        content=f"🔄 Retornando ao Supervisor.\n\n"
               f"**Status:** {completion_status}\n\n"
               f"**Resumo:** {summary}\n\n" 
               f"**Próximos passos:** {next_steps}",
        tool_call_id=tool_call_id,
        name="return_to_supervisor"
    )
    
    # Calcular progresso baseado na fase atual
    phase_progress = {
        "initialization": 5,
        "pico_definition": 10,
        "literature_search": 25,
        "content_extraction": 40,
        "data_processing": 50,
        "vectorization": 55,
        "information_retrieval": 60,
        "statistical_analysis": 80,
        "report_generation": 90,
        "quality_review": 95,
        "final_editing": 98,
        "completed": 100
    }
    
    current_progress = phase_progress.get(state.get("current_phase", "initialization"), 0)
    
    return Command(
        goto="supervisor",
        update={
            "messages": state["messages"] + [tool_message],
            "current_agent": "supervisor",
            "last_agent": state.get("current_agent"),
            "completion_percentage": current_progress,
            "updated_at": datetime.now().isoformat(),
            "temp_data": {
                **state.get("temp_data", {}),
                "last_completion_status": completion_status,
                "last_summary": summary
            }
        },
        graph=Command.PARENT
    )

# Ferramenta para situações de erro crítico
@tool("emergency_supervisor_return")
def emergency_supervisor_return(
    error_description: Annotated[str, "Descrição detalhada do erro ocorrido"],
    attempted_action: Annotated[str, "Ação que estava sendo executada quando o erro ocorreu"],
    recovery_suggestion: Annotated[str, "Sugestão para recuperação ou próximos passos"],
    state: Annotated[MetaAnalysisState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retorna ao supervisor em caso de erro crítico"""
    
    from datetime import datetime
    
    tool_message = ToolMessage(
        content=f"🚨 ERRO CRÍTICO - Retorno de emergência ao Supervisor.\n\n"
               f"**Erro:** {error_description}\n\n"
               f"**Ação tentada:** {attempted_action}\n\n"
               f"**Sugestão de recuperação:** {recovery_suggestion}",
        tool_call_id=tool_call_id,
        name="emergency_supervisor_return"
    )
    
    error_details = {
        "agent": state.get("current_agent", "unknown"),
        "error": error_description,
        "action": attempted_action,
        "timestamp": datetime.now().isoformat(),
        "recovery_suggestion": recovery_suggestion
    }
    
    return Command(
        goto="supervisor",
        update={
            "messages": state["messages"] + [tool_message],
            "current_agent": "supervisor",
            "last_agent": state.get("current_agent"),
            "error_flag": True,
            "error_details": error_details,
            "requires_human_intervention": True,
            "updated_at": datetime.now().isoformat()
        },
        graph=Command.PARENT
    )