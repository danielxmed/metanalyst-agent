from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..tools.retrieval_tools import search_vector_store, search_by_pico, get_relevant_chunks
from ..tools.handoff_tools import transfer_to_analyst, transfer_to_writer

retriever_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, temperature=0.1)
retriever_tools = [search_vector_store, search_by_pico, get_relevant_chunks, transfer_to_analyst, transfer_to_writer]

retriever_prompt = """Você é um RETRIEVER AGENT especialista em busca semântica.
Busque informações relevantes no vector store baseadas em critérios PICO."""

retriever_agent = create_react_agent(model=retriever_llm, tools=retriever_tools, state_modifier=retriever_prompt)