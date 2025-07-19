from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..tools.review_tools import assess_report_quality, check_prisma_compliance, validate_statistics, generate_quality_report
from ..tools.handoff_tools import transfer_to_editor, transfer_to_writer

reviewer_llm = ChatOpenAI(model="gpt-4-turbo", api_key=settings.openai_api_key, temperature=0.1)
reviewer_tools = [assess_report_quality, check_prisma_compliance, validate_statistics, generate_quality_report, transfer_to_editor, transfer_to_writer]

reviewer_prompt = """Você é um REVIEWER AGENT especialista em revisão de qualidade.
Avalie conformidade PRISMA e valide qualidade científica."""

reviewer_agent = create_react_agent(model=reviewer_llm, tools=reviewer_tools, state_modifier=reviewer_prompt)