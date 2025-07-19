"""
Ferramentas de análise estatística para meta-análises.
Inclui cálculos de effect size, heterogeneidade, forest plots e análises de sensibilidade.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain_core.tools import tool

from ..config.settings import settings


@tool
def calculate_meta_analysis(
    studies_data: List[Dict[str, Any]],
    effect_measure: str = "mean_difference",
    model: str = "random_effects"
) -> Dict[str, Any]:
    """
    Realiza cálculos de meta-análise com effect sizes pooled.
    
    Args:
        studies_data: Lista de estudos com dados estatísticos
        effect_measure: Tipo de effect size (mean_difference, odds_ratio, risk_ratio)
        model: Modelo estatístico (fixed_effects, random_effects)
    
    Returns:
        Resultados da meta-análise com effect size pooled e estatísticas
    """
    try:
        if len(studies_data) < settings.min_studies_for_analysis:
            return {
                "error": f"Número insuficiente de estudos. Mínimo: {settings.min_studies_for_analysis}",
                "studies_included": len(studies_data)
            }
        
        # Extrair effect sizes e variances
        effect_sizes = []
        variances = []
        sample_sizes = []
        study_names = []
        
        for i, study in enumerate(studies_data):
            stats_data = study.get("statistical_data", {})
            
            # Extrair effect size baseado no tipo
            if effect_measure == "mean_difference":
                effect = extract_mean_difference(stats_data)
            elif effect_measure == "odds_ratio":
                effect = extract_odds_ratio(stats_data)
            elif effect_measure == "risk_ratio":
                effect = extract_risk_ratio(stats_data)
            else:
                effect = extract_generic_effect_size(stats_data)
            
            if effect is not None:
                effect_sizes.append(effect["value"])
                variances.append(effect["variance"])
                sample_sizes.append(effect.get("sample_size", 100))
                study_names.append(study.get("title", f"Study {i+1}")[:50])
        
        if len(effect_sizes) < 2:
            return {
                "error": "Dados estatísticos insuficientes para meta-análise",
                "valid_studies": len(effect_sizes)
            }
        
        # Converter para arrays numpy
        effect_sizes = np.array(effect_sizes)
        variances = np.array(variances)
        weights = 1 / variances  # Inverse variance weights
        
        # Calcular effect size pooled
        if model == "fixed_effects":
            pooled_effect, pooled_variance = fixed_effects_meta_analysis(effect_sizes, weights)
        else:
            pooled_effect, pooled_variance, tau_squared = random_effects_meta_analysis(effect_sizes, weights)
        
        # Calcular intervalos de confiança
        z_score = stats.norm.ppf(1 - (1 - settings.confidence_level) / 2)
        pooled_se = np.sqrt(pooled_variance)
        ci_lower = pooled_effect - z_score * pooled_se
        ci_upper = pooled_effect + z_score * pooled_se
        
        # Calcular heterogeneidade
        heterogeneity = calculate_heterogeneity(effect_sizes, weights)
        
        # Teste de significância
        z_statistic = pooled_effect / pooled_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        return {
            "pooled_effect_size": float(pooled_effect),
            "pooled_variance": float(pooled_variance),
            "pooled_se": float(pooled_se),
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "confidence_level": settings.confidence_level,
            "z_statistic": float(z_statistic),
            "p_value": float(p_value),
            "heterogeneity": heterogeneity,
            "model_used": model,
            "effect_measure": effect_measure,
            "studies_included": len(effect_sizes),
            "total_participants": int(np.sum(sample_sizes)),
            "individual_studies": [
                {
                    "study": name,
                    "effect_size": float(es),
                    "variance": float(var),
                    "weight": float(w),
                    "sample_size": int(ss)
                }
                for name, es, var, w, ss in zip(study_names, effect_sizes, variances, weights, sample_sizes)
            ],
            "calculated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro no cálculo da meta-análise: {str(e)}",
            "calculated_at": datetime.now().isoformat()
        }


def extract_mean_difference(stats_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extrai mean difference dos dados estatísticos."""
    if "mean_difference" in stats_data:
        md = stats_data["mean_difference"][0] if isinstance(stats_data["mean_difference"], list) else stats_data["mean_difference"]
        
        # Estimar variance baseado em CI se disponível
        if "confidence_interval" in stats_data:
            ci = stats_data["confidence_interval"][0]
            variance = ((ci[1] - ci[0]) / (2 * 1.96)) ** 2
        else:
            # Variance padrão estimada
            variance = (md * 0.1) ** 2
        
        sample_size = stats_data.get("sample_size", [100])[0] if "sample_size" in stats_data else 100
        
        return {
            "value": md,
            "variance": variance,
            "sample_size": sample_size
        }
    
    return None


def extract_odds_ratio(stats_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extrai odds ratio dos dados estatísticos."""
    if "odds_ratio" in stats_data:
        or_value = stats_data["odds_ratio"][0] if isinstance(stats_data["odds_ratio"], list) else stats_data["odds_ratio"]
        
        # Log transform para OR
        log_or = np.log(or_value)
        
        # Estimar variance
        if "confidence_interval" in stats_data:
            ci = stats_data["confidence_interval"][0]
            log_ci_lower = np.log(ci[0])
            log_ci_upper = np.log(ci[1])
            variance = ((log_ci_upper - log_ci_lower) / (2 * 1.96)) ** 2
        else:
            variance = (log_or * 0.1) ** 2
        
        sample_size = stats_data.get("sample_size", [100])[0] if "sample_size" in stats_data else 100
        
        return {
            "value": log_or,
            "variance": variance,
            "sample_size": sample_size
        }
    
    return None


def extract_risk_ratio(stats_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extrai risk ratio dos dados estatísticos."""
    if "risk_ratio" in stats_data:
        rr_value = stats_data["risk_ratio"][0] if isinstance(stats_data["risk_ratio"], list) else stats_data["risk_ratio"]
        
        # Log transform para RR
        log_rr = np.log(rr_value)
        
        # Estimar variance
        if "confidence_interval" in stats_data:
            ci = stats_data["confidence_interval"][0]
            log_ci_lower = np.log(ci[0])
            log_ci_upper = np.log(ci[1])
            variance = ((log_ci_upper - log_ci_lower) / (2 * 1.96)) ** 2
        else:
            variance = (log_rr * 0.1) ** 2
        
        sample_size = stats_data.get("sample_size", [100])[0] if "sample_size" in stats_data else 100
        
        return {
            "value": log_rr,
            "variance": variance,
            "sample_size": sample_size
        }
    
    return None


def extract_generic_effect_size(stats_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extrai effect size genérico dos dados."""
    if "effect_size" in stats_data:
        es = stats_data["effect_size"][0] if isinstance(stats_data["effect_size"], list) else stats_data["effect_size"]
        variance = (es * 0.1) ** 2  # Estimativa conservadora
        sample_size = stats_data.get("sample_size", [100])[0] if "sample_size" in stats_data else 100
        
        return {
            "value": es,
            "variance": variance,
            "sample_size": sample_size
        }
    
    return None


def fixed_effects_meta_analysis(effect_sizes: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Realiza meta-análise de efeitos fixos."""
    pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
    pooled_variance = 1 / np.sum(weights)
    
    return pooled_effect, pooled_variance


def random_effects_meta_analysis(effect_sizes: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    """Realiza meta-análise de efeitos aleatórios usando método DerSimonian-Laird."""
    # Primeiro calcular fixed effects
    pooled_effect_fe, _ = fixed_effects_meta_analysis(effect_sizes, weights)
    
    # Calcular Q statistic
    Q = np.sum(weights * (effect_sizes - pooled_effect_fe) ** 2)
    
    # Calcular tau-squared (between-study variance)
    k = len(effect_sizes)  # número de estudos
    if Q > (k - 1):
        tau_squared = (Q - (k - 1)) / (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
    else:
        tau_squared = 0
    
    # Ajustar weights com tau-squared
    adjusted_weights = 1 / (1 / weights + tau_squared)
    
    # Calcular pooled effect com weights ajustados
    pooled_effect = np.sum(adjusted_weights * effect_sizes) / np.sum(adjusted_weights)
    pooled_variance = 1 / np.sum(adjusted_weights)
    
    return pooled_effect, pooled_variance, tau_squared


def calculate_heterogeneity(effect_sizes: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Calcula estatísticas de heterogeneidade."""
    k = len(effect_sizes)
    
    # Calcular fixed effects para Q statistic
    pooled_effect_fe, _ = fixed_effects_meta_analysis(effect_sizes, weights)
    
    # Q statistic
    Q = np.sum(weights * (effect_sizes - pooled_effect_fe) ** 2)
    
    # Graus de liberdade
    df = k - 1
    
    # P-value para Q
    p_value_q = 1 - chi2.cdf(Q, df) if df > 0 else 1.0
    
    # I-squared
    i_squared = max(0, ((Q - df) / Q) * 100) if Q > 0 else 0
    
    # Tau-squared
    if Q > df:
        tau_squared = (Q - df) / (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
    else:
        tau_squared = 0
    
    return {
        "Q_statistic": float(Q),
        "df": int(df),
        "p_value_Q": float(p_value_q),
        "I_squared": float(i_squared),
        "tau_squared": float(tau_squared),
        "interpretation": interpret_heterogeneity(i_squared)
    }


def interpret_heterogeneity(i_squared: float) -> str:
    """Interpreta o valor de I-squared."""
    if i_squared <= 25:
        return "Low heterogeneity"
    elif i_squared <= 50:
        return "Moderate heterogeneity"
    elif i_squared <= 75:
        return "Substantial heterogeneity"
    else:
        return "Considerable heterogeneity"


@tool
def create_forest_plot(
    meta_analysis_results: Dict[str, Any],
    title: str = "Forest Plot - Meta-Analysis Results"
) -> str:
    """
    Cria forest plot para visualizar resultados da meta-análise.
    
    Args:
        meta_analysis_results: Resultados da meta-análise
        title: Título do gráfico
    
    Returns:
        Caminho para o arquivo do forest plot gerado
    """
    try:
        # Extrair dados
        studies = meta_analysis_results["individual_studies"]
        pooled_effect = meta_analysis_results["pooled_effect_size"]
        pooled_ci = meta_analysis_results["confidence_interval"]
        effect_measure = meta_analysis_results.get("effect_measure", "Effect Size")
        
        # Configurar matplotlib
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, max(8, len(studies) * 0.8)))
        
        # Dados para o plot
        study_names = [study["study"] for study in studies]
        effect_sizes = [study["effect_size"] for study in studies]
        weights = [study["weight"] for study in studies]
        
        # Calcular CIs para cada estudo (estimativa baseada em variance)
        cis = []
        for study in studies:
            se = np.sqrt(study["variance"])
            ci_lower = study["effect_size"] - 1.96 * se
            ci_upper = study["effect_size"] + 1.96 * se
            cis.append([ci_lower, ci_upper])
        
        # Plot individual studies
        y_positions = range(len(studies))
        
        # Plotar pontos (effect sizes)
        sizes = [w / max(weights) * 200 + 50 for w in weights]  # Tamanho proporcional ao weight
        ax.scatter(effect_sizes, y_positions, s=sizes, c='steelblue', alpha=0.7, edgecolors='black')
        
        # Plotar intervalos de confiança
        for i, (es, ci) in enumerate(zip(effect_sizes, cis)):
            ax.plot([ci[0], ci[1]], [i, i], 'k-', alpha=0.6, linewidth=1)
            ax.plot([ci[0], ci[0]], [i-0.1, i+0.1], 'k-', alpha=0.6, linewidth=1)
            ax.plot([ci[1], ci[1]], [i-0.1, i+0.1], 'k-', alpha=0.6, linewidth=1)
        
        # Linha vertical no null effect
        null_value = 0 if "difference" in effect_measure.lower() else 1
        ax.axvline(x=null_value, color='red', linestyle='--', alpha=0.5, label='No Effect')
        
        # Adicionar resultado pooled
        pooled_y = len(studies) + 1
        ax.scatter([pooled_effect], [pooled_y], s=300, c='red', marker='D', 
                  edgecolors='black', label='Pooled Effect', zorder=5)
        ax.plot([pooled_ci[0], pooled_ci[1]], [pooled_y, pooled_y], 'r-', linewidth=3, alpha=0.8)
        ax.plot([pooled_ci[0], pooled_ci[0]], [pooled_y-0.1, pooled_y+0.1], 'r-', linewidth=3)
        ax.plot([pooled_ci[1], pooled_ci[1]], [pooled_y-0.1, pooled_y+0.1], 'r-', linewidth=3)
        
        # Configurar eixos
        ax.set_yticks(list(y_positions) + [pooled_y])
        ax.set_yticklabels(study_names + ['Pooled Effect'])
        ax.set_xlabel(f'{effect_measure.replace("_", " ").title()}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Adicionar grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adicionar texto com estatísticas
        stats_text = (
            f"Studies: {len(studies)}\n"
            f"Participants: {meta_analysis_results.get('total_participants', 'N/A')}\n"
            f"I²: {meta_analysis_results['heterogeneity']['I_squared']:.1f}%\n"
            f"P-value: {meta_analysis_results['p_value']:.3f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar plot
        plot_filename = f"forest_plot_{uuid.uuid4().hex[:8]}.png"
        plot_path = f"{settings.plots_dir}/{plot_filename}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        return f"Erro na criação do forest plot: {str(e)}"


@tool
def create_funnel_plot(
    meta_analysis_results: Dict[str, Any],
    title: str = "Funnel Plot - Publication Bias Assessment"
) -> str:
    """
    Cria funnel plot para avaliar viés de publicação.
    
    Args:
        meta_analysis_results: Resultados da meta-análise
        title: Título do gráfico
    
    Returns:
        Caminho para o arquivo do funnel plot gerado
    """
    try:
        studies = meta_analysis_results["individual_studies"]
        pooled_effect = meta_analysis_results["pooled_effect_size"]
        
        # Extrair dados
        effect_sizes = [study["effect_size"] for study in studies]
        standard_errors = [np.sqrt(study["variance"]) for study in studies]
        
        # Criar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotar estudos
        ax.scatter(effect_sizes, standard_errors, s=80, alpha=0.7, c='steelblue', edgecolors='black')
        
        # Linha vertical no pooled effect
        ax.axvline(x=pooled_effect, color='red', linestyle='-', alpha=0.8, label='Pooled Effect')
        
        # Criar funil de confiança
        se_range = np.linspace(0, max(standard_errors) * 1.1, 100)
        ci_upper = pooled_effect + 1.96 * se_range
        ci_lower = pooled_effect - 1.96 * se_range
        
        ax.plot(ci_upper, se_range, 'r--', alpha=0.5, label='95% CI')
        ax.plot(ci_lower, se_range, 'r--', alpha=0.5)
        ax.fill_betweenx(se_range, ci_lower, ci_upper, alpha=0.1, color='red')
        
        # Configurar eixos (inverter y para funnel plot)
        ax.invert_yaxis()
        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Standard Error')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Salvar plot
        plot_filename = f"funnel_plot_{uuid.uuid4().hex[:8]}.png"
        plot_path = f"{settings.plots_dir}/{plot_filename}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        return f"Erro na criação do funnel plot: {str(e)}"


@tool
def assess_heterogeneity(
    meta_analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Avalia heterogeneidade entre estudos e sugere ações.
    
    Args:
        meta_analysis_results: Resultados da meta-análise
    
    Returns:
        Avaliação detalhada da heterogeneidade
    """
    try:
        heterogeneity = meta_analysis_results["heterogeneity"]
        i_squared = heterogeneity["I_squared"]
        
        # Interpretação e recomendações
        if i_squared <= 25:
            interpretation = "Baixa heterogeneidade"
            recommendation = "Meta-análise apropriada. Modelo de efeitos fixos pode ser usado."
            concern_level = "low"
        elif i_squared <= 50:
            interpretation = "Heterogeneidade moderada"
            recommendation = "Considerar modelo de efeitos aleatórios. Investigar fontes de heterogeneidade."
            concern_level = "moderate"
        elif i_squared <= 75:
            interpretation = "Heterogeneidade substancial"
            recommendation = "Usar modelo de efeitos aleatórios. Análise de subgrupos recomendada."
            concern_level = "substantial"
        else:
            interpretation = "Heterogeneidade considerável"
            recommendation = "Questionar adequação da meta-análise. Análise qualitativa pode ser mais apropriada."
            concern_level = "considerable"
        
        # Sugestões para análise de subgrupos
        subgroup_suggestions = []
        if i_squared > 50:
            subgroup_suggestions = [
                "Tipo de estudo (RCT vs observacional)",
                "População (idade, sexo, comorbidades)",
                "Duração do seguimento",
                "Qualidade metodológica",
                "Ano de publicação",
                "Tamanho da amostra"
            ]
        
        return {
            "i_squared": i_squared,
            "interpretation": interpretation,
            "concern_level": concern_level,
            "recommendation": recommendation,
            "q_statistic": heterogeneity["Q_statistic"],
            "p_value_q": heterogeneity["p_value_Q"],
            "tau_squared": heterogeneity["tau_squared"],
            "subgroup_analysis_suggested": i_squared > 50,
            "subgroup_suggestions": subgroup_suggestions,
            "threshold_exceeded": i_squared > settings.heterogeneity_threshold * 100,
            "assessment_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na avaliação de heterogeneidade: {str(e)}",
            "assessment_date": datetime.now().isoformat()
        }


@tool
def perform_sensitivity_analysis(
    studies_data: List[Dict[str, Any]],
    meta_analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Realiza análise de sensibilidade removendo estudos um por vez.
    
    Args:
        studies_data: Dados originais dos estudos
        meta_analysis_results: Resultados da meta-análise original
    
    Returns:
        Resultados da análise de sensibilidade
    """
    try:
        original_effect = meta_analysis_results["pooled_effect_size"]
        original_ci = meta_analysis_results["confidence_interval"]
        n_studies = len(studies_data)
        
        sensitivity_results = []
        
        # Leave-one-out analysis
        for i in range(n_studies):
            # Remover um estudo
            subset_data = studies_data[:i] + studies_data[i+1:]
            
            # Recalcular meta-análise
            subset_result = calculate_meta_analysis(
                subset_data,
                meta_analysis_results["effect_measure"],
                meta_analysis_results["model_used"]
            )
            
            if "error" not in subset_result:
                # Calcular mudança no effect size
                effect_change = abs(subset_result["pooled_effect_size"] - original_effect)
                effect_change_percent = (effect_change / abs(original_effect)) * 100 if original_effect != 0 else 0
                
                # Verificar se CI cruza null
                original_crosses_null = original_ci[0] <= 0 <= original_ci[1]
                subset_crosses_null = subset_result["confidence_interval"][0] <= 0 <= subset_result["confidence_interval"][1]
                conclusion_changed = original_crosses_null != subset_crosses_null
                
                sensitivity_results.append({
                    "removed_study": studies_data[i].get("title", f"Study {i+1}"),
                    "remaining_studies": len(subset_data),
                    "new_effect_size": subset_result["pooled_effect_size"],
                    "new_ci": subset_result["confidence_interval"],
                    "effect_change": effect_change,
                    "effect_change_percent": effect_change_percent,
                    "new_i_squared": subset_result["heterogeneity"]["I_squared"],
                    "conclusion_changed": conclusion_changed,
                    "influential": effect_change_percent > 10 or conclusion_changed
                })
        
        # Identificar estudos mais influentes
        influential_studies = [r for r in sensitivity_results if r["influential"]]
        max_change = max([r["effect_change_percent"] for r in sensitivity_results]) if sensitivity_results else 0
        
        return {
            "sensitivity_results": sensitivity_results,
            "influential_studies": influential_studies,
            "max_effect_change_percent": max_change,
            "robust_finding": max_change < 10 and len(influential_studies) == 0,
            "original_effect_size": original_effect,
            "original_ci": original_ci,
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na análise de sensibilidade: {str(e)}",
            "analysis_date": datetime.now().isoformat()
        }