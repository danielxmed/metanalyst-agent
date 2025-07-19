"""
Ferramentas de revisão e controle de qualidade usando LLMs.
"""

from typing import Dict, Any, List, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
from datetime import datetime
from ..models.state import MetaAnalysisState
from ..models.config import config

# LLM para revisão
review_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.1,  # Mais determinístico para revisão
    api_key=config.openai_api_key
)

@tool("assess_report_quality")
def assess_report_quality(
    report_content: Annotated[str, "Conteúdo completo do relatório para avaliação"],
    quality_criteria: Annotated[List[str], "Critérios específicos de qualidade"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Avalia qualidade geral do relatório de meta-análise usando critérios científicos.
    """
    
    default_criteria = [
        "prisma_compliance",
        "statistical_accuracy", 
        "bias_assessment",
        "clarity_writing",
        "clinical_relevance",
        "completeness"
    ]
    
    quality_criteria = quality_criteria or default_criteria
    
    assessment_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um revisor especializado em meta-análises médicas.

TAREFA: Avaliar a qualidade do relatório de meta-análise usando critérios científicos rigorosos.

CRITÉRIOS DE AVALIAÇÃO:
1. PRISMA Compliance (0-10): Aderência às diretrizes PRISMA
2. Statistical Accuracy (0-10): Precisão das análises estatísticas
3. Bias Assessment (0-10): Avaliação adequada de vieses
4. Clarity Writing (0-10): Clareza e organização da escrita
5. Clinical Relevance (0-10): Relevância clínica dos resultados
6. Completeness (0-10): Completude das informações

PARA CADA CRITÉRIO, FORNEÇA:
- Score (0-10)
- Justificativa detalhada
- Sugestões específicas de melhoria
- Pontos fortes identificados

CRITÉRIOS ESPECÍFICOS A AVALIAR: {quality_criteria}

FORMATO DE SAÍDA: JSON estruturado com:
{{
  "overall_score": score_geral_0_100,
  "overall_grade": "A/B/C/D/F",
  "criteria_scores": {{
    "criterio": {{"score": 0-10, "justification": "texto", "suggestions": ["lista"]}}
  }},
  "strengths": ["pontos fortes"],
  "major_issues": ["problemas principais"],
  "minor_issues": ["problemas menores"],
  "recommendations": ["recomendações prioritárias"],
  "publication_readiness": "ready/needs_revision/major_revision"
}}"""),
        ("human", "Relatório para avaliação:\n\n{report_content}")
    ])
    
    try:
        chain = assessment_prompt | review_llm
        response = chain.invoke({
            "quality_criteria": quality_criteria,
            "report_content": report_content[:15000]  # Limitar tamanho
        })
        
        # Parsear resposta JSON
        try:
            assessment = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: estrutura básica
            assessment = {
                "overall_score": 70,
                "overall_grade": "B",
                "criteria_scores": {},
                "strengths": ["Avaliação automática aplicada"],
                "major_issues": ["Falha na avaliação detalhada"],
                "minor_issues": [],
                "recommendations": ["Revisar manualmente"],
                "publication_readiness": "needs_revision"
            }
        
        return {
            "success": True,
            "assessment": assessment,
            "review_timestamp": datetime.now().isoformat(),
            "criteria_evaluated": quality_criteria
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "assessment": {}
        }

@tool("check_statistical_validity")
def check_statistical_validity(
    statistical_data: Annotated[Dict[str, Any], "Dados estatísticos e análises realizadas"],
    meta_analysis_type: Annotated[str, "Tipo de meta-análise (fixed-effect, random-effect)"] = "random-effect",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Verifica validade estatística das análises realizadas.
    """
    
    validity_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um biostatístico especializado em meta-análises.

TAREFA: Avaliar a validade estatística da meta-análise realizada.

TIPO DE META-ANÁLISE: {meta_analysis_type}

ASPECTOS A VERIFICAR:
1. Adequação do modelo estatístico escolhido
2. Avaliação de heterogeneidade (I², Q-test)
3. Análise de sensibilidade
4. Avaliação de viés de publicação
5. Intervalos de confiança apropriados
6. Tamanhos de efeito calculados corretamente
7. Análises de subgrupos justificadas

CRITÉRIOS DE VALIDAÇÃO:
- Homogeneidade dos estudos (I² < 50% para fixed-effect)
- Número adequado de estudos (≥ 3 para meta-análise)
- Qualidade metodológica dos estudos incluídos
- Consistência dos resultados
- Transparência na metodologia

FORMATO DE SAÍDA: JSON com:
{{
  "statistical_validity": "valid/questionable/invalid",
  "validity_score": 0-100,
  "heterogeneity_assessment": {{"i_squared": valor, "interpretation": "texto"}},
  "model_appropriateness": {{"chosen_model": "texto", "justification": "texto"}},
  "bias_assessment": {{"publication_bias": "low/moderate/high", "methods_used": ["lista"]}},
  "recommendations": ["lista de recomendações"],
  "warnings": ["avisos importantes"],
  "statistical_issues": ["problemas identificados"]
}}"""),
        ("human", "Dados estatísticos para validação:\n\n{statistical_data}")
    ])
    
    try:
        chain = validity_prompt | review_llm
        response = chain.invoke({
            "meta_analysis_type": meta_analysis_type,
            "statistical_data": json.dumps(statistical_data, indent=2)[:10000]
        })
        
        # Parsear resposta
        try:
            validity_check = json.loads(response.content)
        except:
            # Fallback
            validity_check = {
                "statistical_validity": "questionable",
                "validity_score": 60,
                "heterogeneity_assessment": {"i_squared": "unknown", "interpretation": "Não avaliado"},
                "model_appropriateness": {"chosen_model": meta_analysis_type, "justification": "Padrão"},
                "bias_assessment": {"publication_bias": "unknown", "methods_used": []},
                "recommendations": ["Revisar análises manualmente"],
                "warnings": ["Validação automática falhou"],
                "statistical_issues": ["Verificação incompleta"]
            }
        
        return {
            "success": True,
            "validity_check": validity_check,
            "review_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "validity_check": {}
        }

@tool("suggest_improvements")
def suggest_improvements(
    current_report: Annotated[Dict[str, Any], "Relatório atual com todas as seções"],
    focus_area: Annotated[str, "Área específica para melhorias"] = "overall",
    priority_level: Annotated[str, "Nível de prioridade (high, medium, low)"] = "high",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Sugere melhorias específicas para diferentes aspectos do relatório.
    """
    
    improvement_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um consultor especializado em meta-análises médicas de alta qualidade.

TAREFA: Fornecer sugestões específicas e acionáveis para melhorar o relatório.

ÁREA DE FOCO: {focus_area}
PRIORIDADE: {priority_level}

ÁREAS DE MELHORIA POSSÍVEIS:
- methodology: Métodos e rigor metodológico
- statistical: Análises estatísticas e apresentação
- writing: Clareza e qualidade da escrita
- clinical: Relevância e aplicabilidade clínica
- prisma: Aderência às diretrizes PRISMA
- figures: Qualidade e clareza das figuras
- overall: Avaliação geral

PARA CADA SUGESTÃO, FORNEÇA:
1. Descrição específica da melhoria
2. Justificativa (por que é importante)
3. Como implementar (passos concretos)
4. Impacto esperado na qualidade
5. Recursos necessários

FORMATO DE SAÍDA: JSON com:
{{
  "improvement_category": "{focus_area}",
  "priority_level": "{priority_level}",
  "suggestions": [
    {{
      "title": "Título da sugestão",
      "description": "Descrição detalhada",
      "justification": "Por que é importante",
      "implementation_steps": ["passo 1", "passo 2"],
      "expected_impact": "Alto/Médio/Baixo",
      "effort_required": "Alto/Médio/Baixo",
      "urgency": "Crítico/Importante/Opcional"
    }}
  ],
  "quick_fixes": ["melhorias rápidas"],
  "long_term_improvements": ["melhorias de longo prazo"]
}}"""),
        ("human", "Relatório atual para análise:\n\n{current_report}")
    ])
    
    try:
        chain = improvement_prompt | review_llm
        response = chain.invoke({
            "focus_area": focus_area,
            "priority_level": priority_level,
            "current_report": json.dumps(current_report, indent=2)[:12000]
        })
        
        # Parsear sugestões
        try:
            suggestions = json.loads(response.content)
        except:
            # Fallback
            suggestions = {
                "improvement_category": focus_area,
                "priority_level": priority_level,
                "suggestions": [{
                    "title": "Revisão manual necessária",
                    "description": "Análise automática falhou",
                    "justification": "Sistema não conseguiu processar o relatório",
                    "implementation_steps": ["Revisar manualmente"],
                    "expected_impact": "Médio",
                    "effort_required": "Alto",
                    "urgency": "Importante"
                }],
                "quick_fixes": ["Verificar formatação"],
                "long_term_improvements": ["Melhorar sistema de análise"]
            }
        
        return {
            "success": True,
            "suggestions": suggestions,
            "focus_area": focus_area,
            "generation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestions": {}
        }

@tool("validate_conclusions")
def validate_conclusions(
    conclusions: Annotated[str, "Seção de conclusões do relatório"],
    supporting_data: Annotated[Dict[str, Any], "Dados que suportam as conclusões"],
    clinical_context: Annotated[Dict[str, str], "Contexto clínico (PICO)"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Valida se as conclusões são suportadas pelos dados e análises.
    """
    
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um revisor clínico especializado em evidências médicas.

TAREFA: Validar se as conclusões apresentadas são adequadamente suportadas pelos dados.

CONTEXTO CLÍNICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

CRITÉRIOS DE VALIDAÇÃO:
1. Alinhamento com resultados estatísticos
2. Consideração das limitações
3. Apropriada interpretação clínica
4. Não extrapolação além dos dados
5. Consideração de heterogeneidade
6. Qualidade da evidência (GRADE)

ASPECTOS A VERIFICAR:
- As conclusões refletem os resultados?
- Limitações são adequadamente reconhecidas?
- Força da evidência é apropriadamente caracterizada?
- Recomendações são proporcionais aos achados?
- Contexto clínico é adequadamente considerado?

FORMATO DE SAÍDA: JSON com:
{{
  "conclusion_validity": "strong/moderate/weak/unsupported",
  "evidence_strength": "high/moderate/low/very_low",
  "alignment_score": 0-100,
  "supported_statements": ["declarações bem suportadas"],
  "unsupported_statements": ["declarações problemáticas"],
  "missing_considerations": ["aspectos não considerados"],
  "recommended_revisions": ["revisões sugeridas"],
  "grade_assessment": "qualidade da evidência GRADE",
  "clinical_applicability": "alta/moderada/baixa"
}}"""),
        ("human", "Conclusões para validação:\n{conclusions}\n\nDados de suporte:\n{supporting_data}")
    ])
    
    try:
        chain = validation_prompt | review_llm
        response = chain.invoke({
            "population": clinical_context.get("P", ""),
            "intervention": clinical_context.get("I", ""),
            "comparison": clinical_context.get("C", ""),
            "outcome": clinical_context.get("O", ""),
            "conclusions": conclusions[:5000],
            "supporting_data": json.dumps(supporting_data, indent=2)[:8000]
        })
        
        # Parsear validação
        try:
            validation = json.loads(response.content)
        except:
            # Fallback
            validation = {
                "conclusion_validity": "moderate",
                "evidence_strength": "moderate",
                "alignment_score": 70,
                "supported_statements": ["Avaliação não específica"],
                "unsupported_statements": [],
                "missing_considerations": ["Validação automática limitada"],
                "recommended_revisions": ["Revisar manualmente"],
                "grade_assessment": "Não avaliado",
                "clinical_applicability": "moderada"
            }
        
        return {
            "success": True,
            "validation": validation,
            "validation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "validation": {}
        }

@tool("generate_review_report")
def generate_review_report(
    quality_assessment: Annotated[Dict[str, Any], "Avaliação de qualidade completa"],
    improvement_suggestions: Annotated[Dict[str, Any], "Sugestões de melhoria"],
    statistical_validation: Annotated[Dict[str, Any], "Validação estatística"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> str:
    """
    Gera relatório consolidado de revisão com todas as avaliações.
    """
    
    review_report_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um editor científico especializado em meta-análises.

TAREFA: Criar relatório consolidado de revisão integrando todas as avaliações realizadas.

ESTRUTURA DO RELATÓRIO DE REVISÃO:
1. Sumário Executivo da Revisão
2. Pontuação Geral e Recomendação
3. Avaliação por Critérios
4. Validação Estatística
5. Sugestões de Melhoria Prioritárias
6. Questões Críticas a Resolver
7. Aprovação Condicional ou Rejeição

FORMATO: HTML bem estruturado para apresentação profissional.

DIRETRIZES:
- Linguagem clara e construtiva
- Feedback específico e acionável
- Priorizar sugestões por impacto
- Destacar pontos fortes e fracos
- Fornecer timeline para revisões"""),
        ("human", """Dados para o relatório de revisão:

AVALIAÇÃO DE QUALIDADE:
{quality_assessment}

SUGESTÕES DE MELHORIA:
{improvement_suggestions}

VALIDAÇÃO ESTATÍSTICA:
{statistical_validation}""")
    ])
    
    try:
        chain = review_report_prompt | review_llm
        response = chain.invoke({
            "quality_assessment": json.dumps(quality_assessment, indent=2)[:5000],
            "improvement_suggestions": json.dumps(improvement_suggestions, indent=2)[:5000],
            "statistical_validation": json.dumps(statistical_validation, indent=2)[:5000]
        })
        
        return response.content
        
    except Exception as e:
        return f"""
        <h2>Relatório de Revisão - Erro na Geração</h2>
        <p>Ocorreu um erro na geração do relatório de revisão: {str(e)}</p>
        <p>Por favor, revisar manualmente os componentes individuais.</p>
        """