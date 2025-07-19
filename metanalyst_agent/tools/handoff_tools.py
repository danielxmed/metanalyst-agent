"""
Ferramentas de handoff para transferência autônoma entre agentes.
Permite que cada agente decida quando transferir controle para outro agente especializado.
"""

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph.graph import MessagesState


def create_handoff_tool(*, agent_name: str, description: str) -> tool:
    """
    Factory para criar ferramentas de handoff entre agentes.
    
    Args:
        agent_name: Nome do agente de destino
        description: Descrição de quando usar esta ferramenta
    
    Returns:
        Tool configurada para handoff
    """
    tool_name = f"transfer_to_{agent_name}"
    
    @tool(tool_name, description=description)
    def handoff_tool(
        reason: Annotated[str, "Razão detalhada para transferir para este agente"],
        context: Annotated[str, "Contexto e informações relevantes para o próximo agente"],
        next_actions: Annotated[str, "Ações específicas que o próximo agente deve realizar"],
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Transfere controle para outro agente com contexto detalhado"""
        
        # Mensagem de transferência
        tool_message = {
            "role": "tool",
            "content": f"✅ Transferido para {agent_name}. Razão: {reason}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        
        # Mensagem de contexto para o próximo agente
        context_message = {
            "role": "system",
            "content": (
                f"🔄 HANDOFF RECEBIDO de {state.get('last_agent', 'supervisor')}\n\n"
                f"📋 CONTEXTO: {context}\n\n"
                f"🎯 PRÓXIMAS AÇÕES: {next_actions}\n\n"
                f"💡 RAZÃO DA TRANSFERÊNCIA: {reason}\n\n"
                "Analise o contexto e execute as ações necessárias autonomamente."
            )
        }
        
        return Command(
            goto=agent_name,
            update={
                "messages": state["messages"] + [tool_message, context_message],
                "last_agent": agent_name,
                "handoff_reason": reason,
                "handoff_context": context
            },
            graph=Command.PARENT
        )
    
    return handoff_tool


# Criar ferramentas de handoff específicas para cada agente
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description=(
        "Transferir para o Researcher Agent quando precisar de mais artigos científicos, "
        "refinar queries de busca, ou buscar literatura específica em bases de dados médicas."
    )
)

transfer_to_processor = create_handoff_tool(
    agent_name="processor", 
    description=(
        "Transferir para o Processor Agent quando tiver URLs de artigos para processar, "
        "extrair conteúdo, dados estatísticos ou criar chunks para vetorização."
    )
)

transfer_to_retriever = create_handoff_tool(
    agent_name="retriever",
    description=(
        "Transferir para o Retriever Agent quando precisar buscar informações específicas "
        "no vector store, recuperar dados baseados em PICO ou encontrar estudos similares."
    )
)

transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description=(
        "Transferir para o Analyst Agent quando tiver dados estatísticos suficientes "
        "para realizar meta-análise, calcular effect sizes, criar forest plots ou "
        "avaliar heterogeneidade entre estudos."
    )
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer",
    description=(
        "Transferir para o Writer Agent quando a análise estatística estiver completa "
        "e precisar gerar relatório estruturado, seções de metodologia ou discussão."
    )
)

transfer_to_reviewer = create_handoff_tool(
    agent_name="reviewer",
    description=(
        "Transferir para o Reviewer Agent quando o relatório estiver pronto para "
        "revisão de qualidade, verificação de conformidade PRISMA ou validação científica."
    )
)

transfer_to_editor = create_handoff_tool(
    agent_name="editor",
    description=(
        "Transferir para o Editor Agent quando precisar de edição final, formatação "
        "HTML, integração de gráficos ou preparação do documento final."
    )
)

# Ferramenta especial para finalizar o processo
@tool("complete_meta_analysis")
def complete_meta_analysis(
    final_report_path: Annotated[str, "Caminho para o relatório final gerado"],
    summary: Annotated[str, "Resumo executivo dos resultados da meta-análise"],
    key_findings: Annotated[str, "Principais descobertas e conclusões"],
    state: Annotated[MessagesState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Finaliza o processo de meta-análise com relatório completo"""
    
    completion_message = {
        "role": "assistant",
        "content": (
            f"🎉 META-ANÁLISE CONCLUÍDA COM SUCESSO!\n\n"
            f"📄 Relatório Final: {final_report_path}\n\n"
            f"📊 RESUMO EXECUTIVO:\n{summary}\n\n"
            f"🔍 PRINCIPAIS DESCOBERTAS:\n{key_findings}\n\n"
            "A meta-análise foi concluída seguindo as diretrizes PRISMA e "
            "padrões científicos internacionais."
        )
    }
    
    return Command(
        goto="__end__",
        update={
            "messages": state["messages"] + [completion_message],
            "final_report_path": final_report_path,
            "meta_analysis_complete": True,
            "completion_summary": summary,
            "key_findings": key_findings
        }
    )