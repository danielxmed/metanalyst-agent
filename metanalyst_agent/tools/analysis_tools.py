"""
Ferramentas para análise estatística e geração de visualizações.
"""

from typing import List, Dict, Any, Annotated, Optional
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from ..models.state import MetaAnalysisState
from ..models.config import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

@tool("calculate_meta_analysis")
def calculate_meta_analysis(
    studies_data: Annotated[List[Dict[str, Any]], "Lista de dados dos estudos"],
    effect_type: Annotated[str, "Tipo de effect size (OR, RR, MD)"] = "OR",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Calcula meta-análise dos estudos incluídos.
    """
    
    try:
        if len(studies_data) < config.analysis.min_studies:
            return {
                "success": False,
                "error": f"Número insuficiente de estudos. Mínimo: {config.analysis.min_studies}",
                "studies_count": len(studies_data)
            }
        
        # Extrair effect sizes e weights
        effect_sizes = []
        weights = []
        study_names = []
        
        for study in studies_data:
            stats = study.get("statistical_data", {})
            
            # Obter effect size baseado no tipo
            effect_size = None
            if effect_type == "OR" and "odds_ratio" in stats:
                effect_size = np.log(stats["odds_ratio"])  # Log OR
            elif effect_type == "RR" and "relative_risk" in stats:
                effect_size = np.log(stats["relative_risk"])  # Log RR
            elif effect_type == "MD" and "mean_difference" in stats:
                effect_size = stats["mean_difference"]
            
            if effect_size is not None:
                # Weight baseado no tamanho da amostra (simplificado)
                sample_size = stats.get("sample_size", 100)
                weight = sample_size  # Peso simplificado
                
                effect_sizes.append(effect_size)
                weights.append(weight)
                study_names.append(study.get("title", f"Study {len(effect_sizes)}"))
        
        if len(effect_sizes) < 2:
            return {
                "success": False,
                "error": "Dados estatísticos insuficientes para meta-análise",
                "valid_studies": len(effect_sizes)
            }
        
        # Converter para arrays numpy
        effect_sizes = np.array(effect_sizes)
        weights = np.array(weights)
        
        # Calcular meta-análise com modelo de efeitos fixos
        pooled_effect = np.average(effect_sizes, weights=weights)
        
        # Calcular variância pooled
        total_weight = np.sum(weights)
        pooled_variance = 1 / total_weight
        pooled_se = np.sqrt(pooled_variance)
        
        # Intervalo de confiança
        confidence_level = config.analysis.confidence_level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = pooled_effect - z_score * pooled_se
        ci_upper = pooled_effect + z_score * pooled_se
        
        # Transformar de volta se necessário (para OR e RR)
        if effect_type in ["OR", "RR"]:
            pooled_effect_transformed = np.exp(pooled_effect)
            ci_lower_transformed = np.exp(ci_lower)
            ci_upper_transformed = np.exp(ci_upper)
        else:
            pooled_effect_transformed = pooled_effect
            ci_lower_transformed = ci_lower
            ci_upper_transformed = ci_upper
        
        # Teste Z para significância
        z_statistic = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        # Calcular heterogeneidade (I²)
        q_statistic = np.sum(weights * (effect_sizes - pooled_effect) ** 2)
        df = len(effect_sizes) - 1
        
        if df > 0:
            p_heterogeneity = 1 - stats.chi2.cdf(q_statistic, df)
            i_squared = max(0, (q_statistic - df) / q_statistic * 100)
            tau_squared = max(0, (q_statistic - df) / (total_weight - np.sum(weights**2) / total_weight))
        else:
            p_heterogeneity = 1.0
            i_squared = 0.0
            tau_squared = 0.0
        
        # Calcular total de participantes
        total_participants = sum(
            study.get("statistical_data", {}).get("sample_size", 0) 
            for study in studies_data
        )
        
        return {
            "success": True,
            "effect_type": effect_type,
            "number_of_studies": len(effect_sizes),
            "total_participants": total_participants,
            "pooled_effect_size": float(pooled_effect_transformed),
            "confidence_interval": [float(ci_lower_transformed), float(ci_upper_transformed)],
            "p_value": float(p_value),
            "z_statistic": float(z_statistic),
            "heterogeneity_i2": float(i_squared),
            "heterogeneity_p": float(p_heterogeneity),
            "tau_squared": float(tau_squared),
            "q_statistic": float(q_statistic),
            "study_names": study_names,
            "individual_effect_sizes": effect_sizes.tolist(),
            "weights": weights.tolist(),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "studies_count": len(studies_data)
        }

@tool("create_forest_plot")
def create_forest_plot(
    meta_analysis_results: Annotated[Dict[str, Any], "Resultados da meta-análise"],
    output_path: Annotated[str, "Caminho para salvar o gráfico"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> str:
    """
    Cria forest plot dos resultados da meta-análise.
    """
    
    try:
        if not meta_analysis_results.get("success"):
            return "Erro: Resultados de meta-análise inválidos"
        
        # Dados do plot
        study_names = meta_analysis_results["study_names"]
        effect_sizes = meta_analysis_results["individual_effect_sizes"]
        weights = meta_analysis_results["weights"]
        pooled_effect = meta_analysis_results["pooled_effect_size"]
        ci_lower, ci_upper = meta_analysis_results["confidence_interval"]
        effect_type = meta_analysis_results["effect_type"]
        
        # Configurar matplotlib
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, max(6, len(study_names) * 0.5 + 2)))
        
        # Transformar effect sizes se necessário
        if effect_type in ["OR", "RR"]:
            plot_effects = np.exp(effect_sizes) if any(e < 0 for e in effect_sizes) else effect_sizes
        else:
            plot_effects = effect_sizes
        
        # Plot individual studies
        y_positions = range(len(study_names))
        
        # Calcular tamanhos dos marcadores baseado nos pesos
        max_weight = max(weights)
        marker_sizes = [50 + (w / max_weight) * 150 for w in weights]
        
        # Plot points
        ax.scatter(plot_effects, y_positions, s=marker_sizes, 
                  alpha=0.7, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Plot pooled effect (diamond)
        diamond_y = len(study_names) + 0.5
        diamond_size = 200
        ax.scatter([pooled_effect], [diamond_y], s=diamond_size, 
                  marker='D', color='red', edgecolors='darkred', linewidth=2)
        
        # Confidence intervals (simplified)
        ax.errorbar([pooled_effect], [diamond_y], 
                   xerr=[[pooled_effect - ci_lower], [ci_upper - pooled_effect]], 
                   fmt='none', color='red', linewidth=2, capsize=5)
        
        # Linha vertical no null effect
        null_value = 1.0 if effect_type in ["OR", "RR"] else 0.0
        ax.axvline(x=null_value, color='black', linestyle='--', alpha=0.5)
        
        # Configurar eixos
        ax.set_yticks(list(y_positions) + [diamond_y])
        ax.set_yticklabels(study_names + ['Pooled'])
        ax.invert_yaxis()
        
        # Labels
        effect_label = {
            "OR": "Odds Ratio",
            "RR": "Relative Risk", 
            "MD": "Mean Difference"
        }.get(effect_type, effect_type)
        
        ax.set_xlabel(f'{effect_label} (95% CI)')
        ax.set_title(f'Forest Plot - {effect_label}\n'
                    f'I² = {meta_analysis_results["heterogeneity_i2"]:.1f}%, '
                    f'p = {meta_analysis_results["p_value"]:.3f}')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar
        if not output_path:
            os.makedirs("./outputs/plots", exist_ok=True)
            output_path = f"./outputs/plots/forest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        return f"Erro ao criar forest plot: {str(e)}"

@tool("assess_heterogeneity")
def assess_heterogeneity(
    meta_analysis_results: Annotated[Dict[str, Any], "Resultados da meta-análise"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Avalia heterogeneidade entre estudos.
    """
    
    try:
        if not meta_analysis_results.get("success"):
            return {
                "success": False,
                "error": "Resultados de meta-análise inválidos"
            }
        
        i_squared = meta_analysis_results["heterogeneity_i2"]
        p_heterogeneity = meta_analysis_results["heterogeneity_p"]
        tau_squared = meta_analysis_results["tau_squared"]
        
        # Interpretar I²
        if i_squared <= 25:
            i2_interpretation = "Baixa heterogeneidade"
        elif i_squared <= 50:
            i2_interpretation = "Heterogeneidade moderada"
        elif i_squared <= 75:
            i2_interpretation = "Heterogeneidade substancial"
        else:
            i2_interpretation = "Heterogeneidade considerável"
        
        # Interpretar p-value
        significant_heterogeneity = p_heterogeneity < 0.10  # Threshold comum para heterogeneidade
        
        # Recomendações
        recommendations = []
        
        if i_squared > config.analysis.heterogeneity_threshold * 100:
            recommendations.append("Considerar modelo de efeitos aleatórios")
            recommendations.append("Investigar fontes de heterogeneidade")
        
        if significant_heterogeneity:
            recommendations.append("Explorar análises de subgrupo")
            recommendations.append("Considerar meta-regressão")
        
        if i_squared > 75:
            recommendations.append("Questionar adequação da meta-análise")
            recommendations.append("Considerar análise qualitativa")
        
        return {
            "success": True,
            "i_squared": i_squared,
            "i2_interpretation": i2_interpretation,
            "p_heterogeneity": p_heterogeneity,
            "significant_heterogeneity": significant_heterogeneity,
            "tau_squared": tau_squared,
            "recommendations": recommendations,
            "assessment_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool("perform_sensitivity_analysis")
def perform_sensitivity_analysis(
    studies_data: Annotated[List[Dict[str, Any]], "Dados dos estudos"],
    meta_analysis_results: Annotated[Dict[str, Any], "Resultados da meta-análise original"],
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Realiza análise de sensibilidade removendo estudos um por vez.
    """
    
    try:
        if len(studies_data) < 4:  # Precisa de pelo menos 4 estudos
            return {
                "success": False,
                "error": "Estudos insuficientes para análise de sensibilidade",
                "studies_count": len(studies_data)
            }
        
        original_effect = meta_analysis_results["pooled_effect_size"]
        effect_type = meta_analysis_results["effect_type"]
        
        sensitivity_results = []
        
        # Para cada estudo, calcular meta-análise sem ele
        for i, excluded_study in enumerate(studies_data):
            remaining_studies = [s for j, s in enumerate(studies_data) if j != i]
            
            # Recalcular meta-análise
            subset_result = calculate_meta_analysis(
                studies_data=remaining_studies,
                effect_type=effect_type,
                state=state
            )
            
            if subset_result["success"]:
                effect_change = abs(subset_result["pooled_effect_size"] - original_effect)
                percent_change = (effect_change / abs(original_effect)) * 100 if original_effect != 0 else 0
                
                sensitivity_results.append({
                    "excluded_study": excluded_study.get("title", f"Study {i+1}"),
                    "new_effect_size": subset_result["pooled_effect_size"],
                    "effect_change": effect_change,
                    "percent_change": percent_change,
                    "new_i2": subset_result["heterogeneity_i2"],
                    "new_p_value": subset_result["p_value"]
                })
        
        # Identificar estudos influentes
        max_change = max(r["percent_change"] for r in sensitivity_results)
        influential_threshold = 10.0  # 10% de mudança
        
        influential_studies = [
            r for r in sensitivity_results 
            if r["percent_change"] > influential_threshold
        ]
        
        # Estatísticas da análise
        effect_changes = [r["effect_change"] for r in sensitivity_results]
        
        return {
            "success": True,
            "original_effect_size": original_effect,
            "sensitivity_results": sensitivity_results,
            "max_effect_change": max_change,
            "mean_effect_change": np.mean(effect_changes),
            "influential_studies": influential_studies,
            "is_robust": max_change < influential_threshold,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@tool("calculate_effect_sizes")
def calculate_effect_sizes(
    raw_data: Annotated[Dict[str, Any], "Dados brutos do estudo"],
    study_design: Annotated[str, "Tipo de estudo"] = "RCT",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Calcula effect sizes a partir de dados brutos quando disponível.
    """
    
    try:
        calculated_effects = {}
        
        # Verificar se há dados suficientes
        if "intervention_group" in raw_data and "control_group" in raw_data:
            intervention = raw_data["intervention_group"]
            control = raw_data["control_group"]
            
            # Odds Ratio (para dados categóricos)
            if all(k in intervention for k in ["events", "total"]) and \
               all(k in control for k in ["events", "total"]):
                
                a = intervention["events"]  # Eventos no grupo intervenção
                b = intervention["total"] - a  # Não-eventos no grupo intervenção
                c = control["events"]  # Eventos no grupo controle
                d = control["total"] - c  # Não-eventos no grupo controle
                
                if b > 0 and d > 0:  # Evitar divisão por zero
                    odds_ratio = (a * d) / (b * c) if c > 0 and b > 0 else None
                    if odds_ratio:
                        calculated_effects["odds_ratio"] = odds_ratio
                        calculated_effects["log_odds_ratio"] = np.log(odds_ratio)
                        
                        # Erro padrão do log OR
                        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(x > 0 for x in [a,b,c,d]) else None
                        if se_log_or:
                            calculated_effects["se_log_odds_ratio"] = se_log_or
            
            # Risk Ratio
            if all(k in intervention for k in ["events", "total"]) and \
               all(k in control for k in ["events", "total"]):
                
                risk_intervention = intervention["events"] / intervention["total"]
                risk_control = control["events"] / control["total"]
                
                if risk_control > 0:
                    risk_ratio = risk_intervention / risk_control
                    calculated_effects["relative_risk"] = risk_ratio
                    calculated_effects["log_relative_risk"] = np.log(risk_ratio)
            
            # Mean Difference (para dados contínuos)
            if all(k in intervention for k in ["mean", "sd", "n"]) and \
               all(k in control for k in ["mean", "sd", "n"]):
                
                mean_diff = intervention["mean"] - control["mean"]
                calculated_effects["mean_difference"] = mean_diff
                
                # Standardized Mean Difference (Cohen's d)
                pooled_sd = np.sqrt(
                    ((intervention["n"] - 1) * intervention["sd"]**2 + 
                     (control["n"] - 1) * control["sd"]**2) /
                    (intervention["n"] + control["n"] - 2)
                )
                
                if pooled_sd > 0:
                    cohens_d = mean_diff / pooled_sd
                    calculated_effects["standardized_mean_difference"] = cohens_d
        
        return {
            "success": True,
            "calculated_effects": calculated_effects,
            "study_design": study_design,
            "calculation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "study_design": study_design
        }