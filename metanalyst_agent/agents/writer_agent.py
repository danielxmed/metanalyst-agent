from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..tools.writing_tools import generate_report_section, format_citations, create_html_report
from ..tools.handoff_tools import transfer_to_reviewer

writer_llm = ChatOpenAI(model="gpt-4-turbo", api_key=settings.openai_api_key, temperature=0.3)
writer_tools = [generate_report_section, format_citations, create_html_report, transfer_to_reviewer]

writer_prompt = """Você é um WRITER AGENT especialista em redação científica.
Gere relatórios estruturados seguindo diretrizes PRISMA."""

writer_agent = create_react_agent(model=writer_llm, tools=writer_tools, state_modifier=writer_prompt)