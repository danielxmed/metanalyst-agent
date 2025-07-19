"""
Processor Agent - Especialista em processamento de artigos.
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from ..config.settings import settings
from ..tools.processing_tools import (
    extract_article_content,
    extract_statistical_data,
    generate_vancouver_citation,
    chunk_and_vectorize,
    process_article_batch
)
from ..tools.handoff_tools import (
    transfer_to_analyst,
    transfer_to_retriever
)

processor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, temperature=0.1)

processor_tools = [
    extract_article_content,
    extract_statistical_data,
    generate_vancouver_citation,
    chunk_and_vectorize,
    process_article_batch,
    transfer_to_analyst,
    transfer_to_retriever
]

processor_prompt = """Você é um PROCESSOR AGENT especialista em extração e processamento de artigos científicos.

RESPONSABILIDADES:
- Extrair conteúdo completo de artigos usando Tavily Extract
- Identificar e extrair dados estatísticos relevantes
- Gerar citações Vancouver
- Criar chunks e vetorizar para busca semântica

PROTOCOLO:
1. Extrair conteúdo de cada URL
2. Identificar dados estatísticos (effect sizes, CIs, p-values)
3. Gerar citações formatadas
4. Criar chunks e embeddings
5. Armazenar no vector store

TRANSFERIR PARA:
- transfer_to_analyst: Quando dados estatísticos suficientes estiverem extraídos
- transfer_to_retriever: Quando vector store estiver populado"""

processor_agent = create_react_agent(
    model=processor_llm,
    tools=processor_tools,
    state_modifier=processor_prompt
)