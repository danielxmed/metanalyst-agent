"""
Ferramentas de revisão de qualidade para validação de relatórios e conformidade PRISMA.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from datetime import datetime

from ..config.settings import settings


@tool
def assess_report_quality(
    report_sections: Dict[str, str],
    meta_analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Avalia a qualidade geral do relatório de meta-análise.
    
    Args:
        report_sections: Seções do relatório
        meta_analysis_results: Resultados da meta-análise
    
    Returns:
        Avaliação de qualidade com score e recomendações
    """
    try:
        quality_assessment = {
            "overall_score": 0,
            "section_scores": {},
            "recommendations": [],
            "strengths": [],
            "weaknesses": [],
            "compliance_issues": []
        }
        
        # Avaliar cada seção
        max_score = 100
        section_weights = {
            "abstract": 15,
            "introduction": 10,
            "methods": 25,
            "results": 25,
            "discussion": 15,
            "conclusion": 10
        }
        
        total_score = 0
        
        for section, content in report_sections.items():
            section_score = evaluate_section_quality(section, content, meta_analysis_results)
            quality_assessment["section_scores"][section] = section_score
            
            weight = section_weights.get(section, 10)
            total_score += (section_score / 100) * weight
        
        quality_assessment["overall_score"] = min(100, total_score)
        
        # Avaliar adequação estatística
        stats_assessment = assess_statistical_adequacy(meta_analysis_results)
        quality_assessment.update(stats_assessment)
        
        # Gerar recomendações baseadas no score
        if quality_assessment["overall_score"] >= 90:
            quality_assessment["quality_level"] = "Excelente"
            quality_assessment["recommendations"].append("Relatório de alta qualidade, pronto para publicação.")
        elif quality_assessment["overall_score"] >= 75:
            quality_assessment["quality_level"] = "Boa"
            quality_assessment["recommendations"].append("Relatório de boa qualidade com pequenos ajustes necessários.")
        elif quality_assessment["overall_score"] >= 60:
            quality_assessment["quality_level"] = "Adequada"
            quality_assessment["recommendations"].append("Relatório adequado, mas requer revisões substanciais.")
        else:
            quality_assessment["quality_level"] = "Inadequada"
            quality_assessment["recommendations"].append("Relatório requer revisão completa antes da finalização.")
        
        quality_assessment["assessment_date"] = datetime.now().isoformat()
        
        return quality_assessment
        
    except Exception as e:
        return {
            "error": f"Erro na avaliação de qualidade: {str(e)}",
            "assessment_date": datetime.now().isoformat()
        }


def evaluate_section_quality(section: str, content: str, meta_results: Dict[str, Any]) -> int:
    """Avalia qualidade de uma seção específica."""
    score = 0
    
    if section == "abstract":
        # Verificar elementos essenciais do abstract
        required_elements = ["objetivo", "métodos", "resultados", "conclusão"]
        for element in required_elements:
            if element.lower() in content.lower():
                score += 25
    
    elif section == "methods":
        # Verificar elementos metodológicos
        method_elements = ["prisma", "critérios", "busca", "seleção", "extração", "análise"]
        for element in method_elements:
            if element.lower() in content.lower():
                score += 15
    
    elif section == "results":
        # Verificar presença de resultados estatísticos
        if meta_results.get("pooled_effect_size") is not None:
            score += 30
        if meta_results.get("confidence_interval"):
            score += 25
        if meta_results.get("heterogeneity"):
            score += 25
        if "forest plot" in content.lower() or "gráfico" in content.lower():
            score += 20
    
    elif section == "discussion":
        # Verificar elementos de discussão
        discussion_elements = ["limitações", "implicações", "heterogeneidade", "viés"]
        for element in discussion_elements:
            if element.lower() in content.lower():
                score += 25
    
    # Verificações gerais para todas as seções
    if len(content) > 200:  # Conteúdo substancial
        score += 10
    if "estudo" in content.lower() or "study" in content.lower():
        score += 10
    
    return min(100, score)


def assess_statistical_adequacy(meta_results: Dict[str, Any]) -> Dict[str, Any]:
    """Avalia adequação estatística da meta-análise."""
    assessment = {
        "statistical_adequacy": "adequate",
        "statistical_issues": [],
        "statistical_strengths": []
    }
    
    # Verificar número de estudos
    n_studies = meta_results.get("studies_included", 0)
    if n_studies < settings.min_studies_for_analysis:
        assessment["statistical_issues"].append(f"Número insuficiente de estudos ({n_studies})")
        assessment["statistical_adequacy"] = "inadequate"
    elif n_studies >= 10:
        assessment["statistical_strengths"].append("Número adequado de estudos para meta-análise robusta")
    
    # Verificar heterogeneidade
    i_squared = meta_results.get("heterogeneity", {}).get("I_squared", 0)
    if i_squared > 75:
        assessment["statistical_issues"].append(f"Heterogeneidade muito alta (I² = {i_squared:.1f}%)")
    elif i_squared <= 25:
        assessment["statistical_strengths"].append("Baixa heterogeneidade entre estudos")
    
    # Verificar intervalos de confiança
    ci = meta_results.get("confidence_interval", [])
    if len(ci) == 2:
        ci_width = abs(ci[1] - ci[0])
        if ci_width > 2:
            assessment["statistical_issues"].append("Intervalo de confiança muito amplo")
        else:
            assessment["statistical_strengths"].append("Intervalo de confiança adequado")
    
    # Verificar significância
    p_value = meta_results.get("p_value", 1)
    if p_value < 0.001:
        assessment["statistical_strengths"].append("Resultado altamente significativo")
    elif p_value < 0.05:
        assessment["statistical_strengths"].append("Resultado estatisticamente significativo")
    
    return assessment


@tool
def check_prisma_compliance(
    report_sections: Dict[str, str],
    meta_analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verifica conformidade com diretrizes PRISMA.
    
    Args:
        report_sections: Seções do relatório
        meta_analysis_results: Resultados da meta-análise
    
    Returns:
        Avaliação de conformidade PRISMA
    """
    try:
        prisma_checklist = {
            "title": False,
            "abstract_structured": False,
            "introduction_rationale": False,
            "methods_protocol": False,
            "methods_eligibility": False,
            "methods_search": False,
            "methods_selection": False,
            "methods_data_extraction": False,
            "methods_quality_assessment": False,
            "methods_statistical_analysis": False,
            "results_study_selection": False,
            "results_study_characteristics": False,
            "results_risk_of_bias": False,
            "results_individual_studies": False,
            "results_synthesis": False,
            "results_additional_analysis": False,
            "discussion_summary": False,
            "discussion_limitations": False,
            "discussion_conclusions": False,
            "funding": False
        }
        
        # Verificar elementos no abstract
        abstract = report_sections.get("abstract", "")
        if all(word in abstract.lower() for word in ["objetivo", "métodos", "resultados", "conclusão"]):
            prisma_checklist["abstract_structured"] = True
        
        # Verificar métodos
        methods = report_sections.get("methods", "")
        if "prisma" in methods.lower():
            prisma_checklist["methods_protocol"] = True
        if "critérios" in methods.lower() and "inclusão" in methods.lower():
            prisma_checklist["methods_eligibility"] = True
        if "busca" in methods.lower() and "base" in methods.lower():
            prisma_checklist["methods_search"] = True
        if "seleção" in methods.lower() and "revisor" in methods.lower():
            prisma_checklist["methods_selection"] = True
        if "extração" in methods.lower():
            prisma_checklist["methods_data_extraction"] = True
        if "qualidade" in methods.lower():
            prisma_checklist["methods_quality_assessment"] = True
        if "análise" in methods.lower() and "estatística" in methods.lower():
            prisma_checklist["methods_statistical_analysis"] = True
        
        # Verificar resultados
        results = report_sections.get("results", "")
        if "estudos" in results.lower() and "incluídos" in results.lower():
            prisma_checklist["results_study_selection"] = True
        if "características" in results.lower():
            prisma_checklist["results_study_characteristics"] = True
        if meta_analysis_results.get("individual_studies"):
            prisma_checklist["results_individual_studies"] = True
        if meta_analysis_results.get("pooled_effect_size") is not None:
            prisma_checklist["results_synthesis"] = True
        if meta_analysis_results.get("heterogeneity"):
            prisma_checklist["results_additional_analysis"] = True
        
        # Verificar discussão
        discussion = report_sections.get("discussion", "")
        if "achados" in discussion.lower() or "resultados" in discussion.lower():
            prisma_checklist["discussion_summary"] = True
        if "limitações" in discussion.lower():
            prisma_checklist["discussion_limitations"] = True
        if "conclusão" in discussion.lower() or "implicações" in discussion.lower():
            prisma_checklist["discussion_conclusions"] = True
        
        # Calcular score de conformidade
        total_items = len(prisma_checklist)
        compliant_items = sum(prisma_checklist.values())
        compliance_score = (compliant_items / total_items) * 100
        
        # Identificar itens em falta
        missing_items = [item for item, compliant in prisma_checklist.items() if not compliant]
        
        return {
            "compliance_score": compliance_score,
            "compliant_items": compliant_items,
            "total_items": total_items,
            "compliance_level": get_compliance_level(compliance_score),
            "missing_items": missing_items,
            "prisma_checklist": prisma_checklist,
            "recommendations": generate_prisma_recommendations(missing_items),
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na verificação PRISMA: {str(e)}",
            "checked_at": datetime.now().isoformat()
        }


def get_compliance_level(score: float) -> str:
    """Determina nível de conformidade PRISMA."""
    if score >= 90:
        return "Excelente"
    elif score >= 80:
        return "Boa"
    elif score >= 70:
        return "Adequada"
    elif score >= 60:
        return "Limitada"
    else:
        return "Inadequada"


def generate_prisma_recommendations(missing_items: List[str]) -> List[str]:
    """Gera recomendações baseadas em itens PRISMA em falta."""
    recommendations = []
    
    critical_items = [
        "methods_eligibility",
        "methods_search", 
        "methods_statistical_analysis",
        "results_synthesis",
        "discussion_limitations"
    ]
    
    for item in missing_items:
        if item in critical_items:
            if item == "methods_eligibility":
                recommendations.append("Adicionar critérios de elegibilidade claros e específicos")
            elif item == "methods_search":
                recommendations.append("Detalhar estratégia de busca e bases de dados utilizadas")
            elif item == "methods_statistical_analysis":
                recommendations.append("Explicar métodos estatísticos utilizados na meta-análise")
            elif item == "results_synthesis":
                recommendations.append("Apresentar resultados da síntese quantitativa (meta-análise)")
            elif item == "discussion_limitations":
                recommendations.append("Discutir limitações do estudo e da evidência")
    
    if len(missing_items) > 10:
        recommendations.append("Revisar estrutura geral do relatório para melhor aderência às diretrizes PRISMA")
    
    return recommendations


@tool
def validate_statistics(
    meta_analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Valida a adequação dos cálculos estatísticos.
    
    Args:
        meta_analysis_results: Resultados da meta-análise
    
    Returns:
        Validação estatística com alertas e recomendações
    """
    try:
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "statistical_power": "adequate"
        }
        
        # Validar número de estudos
        n_studies = meta_analysis_results.get("studies_included", 0)
        if n_studies < 2:
            validation["errors"].append("Número insuficiente de estudos para meta-análise")
            validation["is_valid"] = False
        elif n_studies < 5:
            validation["warnings"].append("Número limitado de estudos pode afetar a robustez dos resultados")
            validation["statistical_power"] = "limited"
        
        # Validar intervalos de confiança
        ci = meta_analysis_results.get("confidence_interval", [])
        effect_size = meta_analysis_results.get("pooled_effect_size", 0)
        
        if len(ci) == 2:
            if ci[0] > ci[1]:
                validation["errors"].append("Intervalo de confiança inválido (limite inferior > superior)")
                validation["is_valid"] = False
            
            # Verificar se effect size está dentro do CI
            if not (ci[0] <= effect_size <= ci[1]):
                validation["errors"].append("Effect size fora do intervalo de confiança")
                validation["is_valid"] = False
            
            # Verificar largura do CI
            ci_width = abs(ci[1] - ci[0])
            if ci_width > 3:
                validation["warnings"].append("Intervalo de confiança muito amplo indica baixa precisão")
        
        # Validar heterogeneidade
        heterogeneity = meta_analysis_results.get("heterogeneity", {})
        i_squared = heterogeneity.get("I_squared", 0)
        
        if i_squared > 90:
            validation["warnings"].append("Heterogeneidade extremamente alta questiona adequação da meta-análise")
        elif i_squared > 75:
            validation["warnings"].append("Heterogeneidade alta requer investigação de subgrupos")
        
        # Validar p-value
        p_value = meta_analysis_results.get("p_value", 1)
        if p_value < 0 or p_value > 1:
            validation["errors"].append("P-value inválido (deve estar entre 0 e 1)")
            validation["is_valid"] = False
        
        # Validar model choice
        model = meta_analysis_results.get("model_used", "")
        if i_squared > 25 and model == "fixed_effects":
            validation["recommendations"].append("Considerar modelo de efeitos aleatórios devido à heterogeneidade")
        
        # Validar tamanho da amostra total
        total_participants = meta_analysis_results.get("total_participants", 0)
        if total_participants < 100:
            validation["warnings"].append("Tamanho amostral total pequeno limita generalização")
        
        return {
            **validation,
            "validated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": f"Erro na validação estatística: {str(e)}",
            "validated_at": datetime.now().isoformat()
        }


@tool
def generate_quality_report(
    quality_assessment: Dict[str, Any],
    prisma_compliance: Dict[str, Any],
    statistical_validation: Dict[str, Any]
) -> str:
    """
    Gera relatório consolidado de qualidade.
    
    Args:
        quality_assessment: Avaliação de qualidade
        prisma_compliance: Conformidade PRISMA
        statistical_validation: Validação estatística
    
    Returns:
        Relatório de qualidade formatado
    """
    try:
        report = f"""
## RELATÓRIO DE QUALIDADE - META-ANÁLISE

### Resumo Executivo
- **Qualidade Geral:** {quality_assessment.get('quality_level', 'N/A')} (Score: {quality_assessment.get('overall_score', 0):.1f}/100)
- **Conformidade PRISMA:** {prisma_compliance.get('compliance_level', 'N/A')} ({prisma_compliance.get('compliance_score', 0):.1f}%)
- **Validação Estatística:** {'✅ Válida' if statistical_validation.get('is_valid', False) else '❌ Inválida'}

### Avaliação por Seções
"""
        
        # Scores por seção
        for section, score in quality_assessment.get("section_scores", {}).items():
            report += f"- **{section.title()}:** {score}/100\n"
        
        report += f"""

### Conformidade PRISMA
- **Itens Atendidos:** {prisma_compliance.get('compliant_items', 0)}/{prisma_compliance.get('total_items', 0)}
- **Itens em Falta:** {len(prisma_compliance.get('missing_items', []))}

### Validação Estatística
"""
        
        if statistical_validation.get("errors"):
            report += "**❌ Erros Críticos:**\n"
            for error in statistical_validation["errors"]:
                report += f"- {error}\n"
        
        if statistical_validation.get("warnings"):
            report += "**⚠️ Alertas:**\n"
            for warning in statistical_validation["warnings"]:
                report += f"- {warning}\n"
        
        report += f"""

### Recomendações Prioritárias
"""
        
        # Combinar recomendações de todas as avaliações
        all_recommendations = []
        all_recommendations.extend(quality_assessment.get("recommendations", []))
        all_recommendations.extend(prisma_compliance.get("recommendations", []))
        all_recommendations.extend(statistical_validation.get("recommendations", []))
        
        for i, rec in enumerate(all_recommendations[:10], 1):  # Top 10 recomendações
            report += f"{i}. {rec}\n"
        
        report += f"""

### Próximos Passos
{'✅ Relatório aprovado para finalização' if quality_assessment.get('overall_score', 0) >= 80 and statistical_validation.get('is_valid', False) else '📝 Revisões necessárias antes da finalização'}

---
*Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}*
"""
        
        return report.strip()
        
    except Exception as e:
        return f"Erro na geração do relatório de qualidade: {str(e)}"