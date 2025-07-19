"""
Ferramentas de processamento de artigos usando Tavily Extract e LLMs.
"""

from typing import Dict, Any, List, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from ..models.state import MetaAnalysisState, ArticleData
from ..models.config import config
import json
import uuid
from datetime import datetime
import re

# Cliente Tavily global
tavily_client = TavilyClient(api_key=config.tavily_api_key)

# LLM para processamento
processing_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.1,
    api_key=config.openai_api_key
)

@tool("extract_article_content")
def extract_article_content(
    url: Annotated[str, "URL do artigo para extrair conteúdo completo"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Extrai conteúdo completo de artigo usando Tavily Extract API.
    """
    
    try:
        # Extrair conteúdo usando Tavily
        extracted_content = tavily_client.extract(
            urls=[url]
        )
        
        if not extracted_content or not extracted_content.get("results"):
            return {
                "success": False,
                "error": "Falha na extração - conteúdo vazio",
                "url": url
            }
        
        result = extracted_content["results"][0]
        
        return {
            "success": True,
            "url": url,
            "raw_content": result.get("raw_content", ""),
            "extracted_content": result.get("extracted_content", ""),
            "title": result.get("title", ""),
            "extraction_timestamp": datetime.now().isoformat(),
            "content_length": len(result.get("raw_content", ""))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url
        }

@tool("extract_statistical_data")
def extract_statistical_data(
    content: Annotated[str, "Conteúdo completo do artigo extraído"],
    pico: Annotated[Dict[str, str], "Estrutura PICO para contexto da extração"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Extrai dados estatísticos relevantes do artigo usando LLM.
    AI-first approach para identificar e estruturar dados estatísticos.
    """
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em extração de dados estatísticos de artigos médicos para meta-análise.

CONTEXTO PICO:
Population: {population}
Intervention: {intervention} 
Comparison: {comparison}
Outcome: {outcome}

TAREFA: Extraia dados estatísticos relevantes para esta meta-análise do artigo fornecido.

DADOS REQUERIDOS:
1. Metadados do estudo:
   - Tipo de estudo (RCT, cohort, case-control, etc.)
   - Tamanho da amostra total
   - Tamanho do grupo intervenção
   - Tamanho do grupo controle
   - Duração do seguimento
   - Critérios de inclusão/exclusão

2. Resultados estatísticos:
   - Measure de efeito (OR, RR, HR, MD, SMD)
   - Intervalo de confiança (95% CI)
   - Valor p
   - Dados numéricos brutos (eventos/total para cada grupo)
   - Médias e desvios padrão (se aplicável)

3. Qualidade do estudo:
   - Método de randomização
   - Cegamento
   - Perdas de seguimento
   - Fonte de financiamento

FORMATO DE SAÍDA: JSON estruturado apenas.
Se algum dado não estiver disponível, use null.
Se o artigo não for relevante para o PICO, retorne {{"relevant": false}}.

IMPORTANTE: Seja preciso e extraia apenas dados explicitamente mencionados no texto."""),
        ("human", "Artigo para análise:\n\n{content}")
    ])
    
    try:
        # Processar com LLM
        chain = extraction_prompt | processing_llm
        
        response = chain.invoke({
            "population": pico.get("P", ""),
            "intervention": pico.get("I", ""),
            "comparison": pico.get("C", ""),
            "outcome": pico.get("O", ""),
            "content": content[:15000]  # Limitar tamanho para o LLM
        })
        
        # Parsear resposta JSON
        try:
            extracted_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Se não conseguir parsear, tentar extrair JSON do texto
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                raise ValueError("Resposta não está em formato JSON válido")
        
        return {
            "success": True,
            "relevant": extracted_data.get("relevant", True),
            "extracted_data": extracted_data,
            "extraction_method": "llm_processing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "extraction_method": "llm_processing"
        }

@tool("generate_vancouver_citation")
def generate_vancouver_citation(
    article_metadata: Annotated[Dict[str, Any], "Metadados do artigo para gerar citação"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> str:
    """
    Gera citação no formato Vancouver usando LLM para garantir formatação correta.
    """
    
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em citações acadêmicas no formato Vancouver.

TAREFA: Gerar citação no estilo Vancouver baseada nos metadados fornecidos.

FORMATO VANCOUVER:
Autor(es). Título do artigo. Nome da revista. Ano;Volume(Número):Páginas.

REGRAS:
1. Até 6 autores: listar todos
2. Mais de 6 autores: listar os primeiros 6 seguidos de "et al."
3. Título do artigo: apenas primeira palavra maiúscula (exceto nomes próprios)
4. Nome da revista: forma abreviada oficial se possível
5. Se dados faltarem, adapte o formato adequadamente

IMPORTANTE: Retorne apenas a citação formatada, sem explicações."""),
        ("human", "Metadados do artigo:\n{metadata}")
    ])
    
    try:
        chain = citation_prompt | processing_llm
        
        response = chain.invoke({
            "metadata": json.dumps(article_metadata, indent=2)
        })
        
        return response.content.strip()
        
    except Exception as e:
        # Fallback para citação básica
        title = article_metadata.get("title", "Título não disponível")
        authors = article_metadata.get("authors", ["Autor não disponível"])
        journal = article_metadata.get("journal", "Revista não disponível")
        year = article_metadata.get("year", "Ano não disponível")
        
        if isinstance(authors, list) and authors:
            author_str = ", ".join(authors[:6])
            if len(authors) > 6:
                author_str += " et al."
        else:
            author_str = str(authors) if authors else "Autor não disponível"
        
        return f"{author_str}. {title}. {journal}. {year}."

@tool("assess_article_quality")
def assess_article_quality(
    article_data: Annotated[Dict[str, Any], "Dados completos do artigo para avaliação"],
    study_type: Annotated[str, "Tipo de estudo (RCT, cohort, etc.)"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Avalia qualidade metodológica do artigo usando critérios padrão via LLM.
    """
    
    quality_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em avaliação de qualidade metodológica de estudos médicos.

TIPO DE ESTUDO: {study_type}

TAREFA: Avaliar a qualidade metodológica deste estudo usando critérios apropriados.

CRITÉRIOS DE AVALIAÇÃO:

Para RCTs (usar escala Jadad ou Cochrane RoB):
- Randomização adequada
- Ocultação da alocação  
- Cegamento de participantes
- Cegamento de avaliadores
- Dados de outcome completos
- Relato seletivo de outcomes
- Outras fontes de viés

Para estudos observacionais (usar Newcastle-Ottawa):
- Seleção da coorte/casos
- Comparabilidade dos grupos
- Avaliação do outcome/exposição

FORMATO DE SAÍDA: JSON com:
{{
  "overall_quality": "high/moderate/low/very_low",
  "quality_score": número de 0-10,
  "criteria_scores": {{criterio: score}},
  "strengths": [lista de pontos fortes],
  "limitations": [lista de limitações],
  "risk_of_bias": "low/moderate/high",
  "recommendation": "include/exclude/uncertain"
}}"""),
        ("human", "Dados do estudo para avaliação:\n{article_data}")
    ])
    
    try:
        chain = quality_prompt | processing_llm
        
        response = chain.invoke({
            "study_type": study_type,
            "article_data": json.dumps(article_data, indent=2)[:10000]
        })
        
        # Parsear resposta JSON
        try:
            quality_assessment = json.loads(response.content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                quality_assessment = json.loads(json_match.group())
            else:
                # Fallback assessment
                quality_assessment = {
                    "overall_quality": "uncertain",
                    "quality_score": 5.0,
                    "criteria_scores": {},
                    "strengths": ["Avaliação automática indisponível"],
                    "limitations": ["Falha na avaliação detalhada"],
                    "risk_of_bias": "uncertain",
                    "recommendation": "review_manually"
                }
        
        return {
            "success": True,
            "assessment": quality_assessment,
            "assessment_method": "llm_evaluation",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "assessment_method": "llm_evaluation"
        }