from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..tools.writing_tools import create_html_report, format_citations
from ..tools.handoff_tools import complete_meta_analysis

editor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, temperature=0.1)
editor_tools = [create_html_report, format_citations, complete_meta_analysis]

editor_prompt = """Você é um EDITOR AGENT especialista em formatação final.
Prepare o documento final HTML com gráficos integrados."""

editor_agent = create_react_agent(model=editor_llm, tools=editor_tools, state_modifier=editor_prompt)