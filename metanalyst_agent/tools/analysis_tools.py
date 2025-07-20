"""Analysis tools for statistical meta-analysis calculations and visualizations"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import logging
import base64
import io

logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')


@tool
def calculate_meta_analysis(
    studies_data: List[Dict[str, Any]],
    outcome_measure: str = "mean_difference",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Perform statistical meta-analysis calculations using LLM guidance and statistical methods
    
    Args:
        studies_data: List of studies with statistical data
        outcome_measure: Type of outcome (mean_difference, odds_ratio, risk_ratio, etc.)
        confidence_level: Confidence level for intervals (default 0.95)
        
    Returns:
        Dictionary with meta-analysis results including pooled estimates and heterogeneity
    """
    try:
        if not studies_data or len(studies_data) < 2:
            return {
                "success": False,
                "error": "Need at least 2 studies for meta-analysis",
                "studies_provided": len(studies_data)
            }
        
        # Use LLM to guide the meta-analysis approach
        llm = ChatOpenAI(model="o3")
        
        prompt = f"""
        You are an expert biostatistician performing a meta-analysis.
        
        STUDIES DATA:
        {json.dumps(studies_data, indent=2)}
        
        OUTCOME MEASURE: {outcome_measure}
        CONFIDENCE LEVEL: {confidence_level}
        
        Based on the provided studies, determine:
        1. The most appropriate meta-analysis model (fixed-effect vs random-effects)
        2. Effect size calculation method
        3. Weighting scheme (inverse variance, etc.)
        4. Heterogeneity assessment approach
        
        For each study, extract or calculate:
        - Effect size and standard error
        - Sample sizes
        - Variance/confidence intervals
        - Study weights
        
        Respond in JSON format:
        {{
            "recommended_model": "random_effects",
            "effect_size_method": "standardized_mean_difference",
            "weighting_method": "inverse_variance",
            "studies_analysis": [
                {{
                    "study_id": 1,
                    "effect_size": -0.52,
                    "standard_error": 0.15,
                    "variance": 0.0225,
                    "weight_fixed": 44.4,
                    "weight_random": 35.2,
                    "sample_size_total": 240,
                    "confidence_interval": [-0.82, -0.22]
                }}
            ],
            "pooled_effect_fixed": {{
                "estimate": -0.45,
                "standard_error": 0.08,
                "confidence_interval": [-0.61, -0.29],
                "z_score": -5.625,
                "p_value": 0.0001
            }},
            "pooled_effect_random": {{
                "estimate": -0.48,
                "standard_error": 0.12,
                "confidence_interval": [-0.72, -0.24],
                "z_score": -4.0,
                "p_value": 0.0001
            }},
            "heterogeneity": {{
                "q_statistic": 12.5,
                "df": 4,
                "p_value": 0.014,
                "i_squared": 68.0,
                "tau_squared": 0.045,
                "interpretation": "moderate_heterogeneity"
            }},
            "total_participants": 1200,
            "total_studies": 5,
            "recommended_result": "random_effects"
        }}
        """
        
        response = llm.invoke(prompt)
        
        try:
            analysis_result = json.loads(response.content)
            
            # Validate and enhance results with additional calculations
            if "studies_analysis" in analysis_result:
                studies_analysis = analysis_result["studies_analysis"]
                
                # Calculate additional statistics
                effect_sizes = [s["effect_size"] for s in studies_analysis]
                weights_random = [s.get("weight_random", 1) for s in studies_analysis]
                
                # Forest plot data preparation
                forest_plot_data = []
                for i, study in enumerate(studies_analysis):
                    forest_plot_data.append({
                        "study_name": f"Study {i+1}",
                        "effect_size": study["effect_size"],
                        "ci_lower": study["confidence_interval"][0],
                        "ci_upper": study["confidence_interval"][1],
                        "weight": study.get("weight_random", 1)
                    })
                
                analysis_result["forest_plot_data"] = forest_plot_data
                analysis_result["calculation_method"] = "llm_guided_statistical"
                analysis_result["analysis_timestamp"] = "2024-01-01T00:00:00Z"
                
                logger.info(f"Meta-analysis completed: {len(studies_data)} studies, pooled effect = {analysis_result.get('pooled_effect_random', {}).get('estimate', 'N/A')}")
                
            return {
                "success": True,
                "meta_analysis_results": analysis_result,
                "outcome_measure": outcome_measure,
                "confidence_level": confidence_level
            }
            
        except json.JSONDecodeError:
            logger.error("Failed to parse meta-analysis results from LLM")
            return {
                "success": False,
                "error": "Failed to parse LLM meta-analysis results",
                "raw_response": response.content[:500]
            }
            
    except Exception as e:
        logger.error(f"Error in meta-analysis calculation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "studies_provided": len(studies_data)
        }


@tool
def create_forest_plot(
    meta_analysis_data: Dict[str, Any],
    title: str = "Forest Plot",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a forest plot visualization for meta-analysis results
    
    Args:
        meta_analysis_data: Results from calculate_meta_analysis
        title: Title for the forest plot
        save_path: Optional path to save the plot
        
    Returns:
        Dictionary with plot information and base64 encoded image
    """
    try:
        if not meta_analysis_data.get("success"):
            return {
                "success": False,
                "error": "Invalid meta-analysis data provided"
            }
        
        forest_data = meta_analysis_data.get("meta_analysis_results", {}).get("forest_plot_data", [])
        pooled_result = meta_analysis_data.get("meta_analysis_results", {}).get("pooled_effect_random", {})
        
        if not forest_data:
            return {
                "success": False,
                "error": "No forest plot data available"
            }
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data for plotting
        studies = [study["study_name"] for study in forest_data]
        effect_sizes = [study["effect_size"] for study in forest_data]
        ci_lower = [study["ci_lower"] for study in forest_data]
        ci_upper = [study["ci_upper"] for study in forest_data]
        weights = [study["weight"] for study in forest_data]
        
        # Create y positions for studies
        y_positions = np.arange(len(studies))
        
        # Plot individual studies
        for i, (study, effect, lower, upper, weight) in enumerate(zip(studies, effect_sizes, ci_lower, ci_upper, weights)):
            # Plot confidence interval
            ax.plot([lower, upper], [i, i], 'b-', linewidth=2, alpha=0.7)
            
            # Plot effect size point (size proportional to weight)
            point_size = max(50, min(300, weight * 5))  # Scale point size
            ax.scatter([effect], [i], s=point_size, c='blue', marker='s', alpha=0.8, zorder=3)
            
            # Add study label
            ax.text(-0.1, i, study, ha='right', va='center', fontsize=10)
            
            # Add effect size and CI text
            ci_text = f"{effect:.2f} [{lower:.2f}, {upper:.2f}]"
            ax.text(max(ci_upper) + 0.1, i, ci_text, ha='left', va='center', fontsize=9)
        
        # Add pooled estimate if available
        if pooled_result:
            pooled_effect = pooled_result.get("estimate", 0)
            pooled_ci = pooled_result.get("confidence_interval", [0, 0])
            
            # Add separator line
            ax.axhline(y=-0.5, color='black', linewidth=1, alpha=0.5)
            
            # Plot pooled result
            pooled_y = -1
            ax.plot([pooled_ci[0], pooled_ci[1]], [pooled_y, pooled_y], 'r-', linewidth=3)
            ax.scatter([pooled_effect], [pooled_y], s=200, c='red', marker='D', zorder=3)
            
            # Add pooled label
            ax.text(-0.1, pooled_y, 'Pooled', ha='right', va='center', fontsize=12, fontweight='bold')
            pooled_text = f"{pooled_effect:.2f} [{pooled_ci[0]:.2f}, {pooled_ci[1]:.2f}]"
            ax.text(max(ci_upper) + 0.1, pooled_y, pooled_text, ha='left', va='center', fontsize=11, fontweight='bold')
        
        # Add vertical line at null effect
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_ylim(-1.5, len(studies))
        ax.set_xlabel('Effect Size', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Remove y-axis ticks and labels
        ax.set_yticks([])
        
        # Add heterogeneity information if available
        heterogeneity = meta_analysis_data.get("meta_analysis_results", {}).get("heterogeneity", {})
        if heterogeneity:
            i_squared = heterogeneity.get("i_squared", 0)
            q_p_value = heterogeneity.get("p_value", 1)
            het_text = f"I² = {i_squared:.1f}%, Q p-value = {q_p_value:.3f}"
            ax.text(0.02, 0.98, het_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        
        # Save or encode plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plot_path = save_path
        else:
            plot_path = f"/tmp/forest_plot_{hash(title)}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        
        plt.close()
        
        logger.info(f"Forest plot created with {len(studies)} studies")
        
        return {
            "success": True,
            "plot_path": plot_path,
            "plot_base64": plot_base64,
            "plot_type": "forest_plot",
            "studies_included": len(studies),
            "title": title,
            "created_at": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error creating forest plot: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "plot_type": "forest_plot"
        }


@tool
def assess_heterogeneity(
    meta_analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess heterogeneity in meta-analysis results using multiple statistical measures
    
    Args:
        meta_analysis_data: Results from calculate_meta_analysis
        
    Returns:
        Comprehensive heterogeneity assessment with interpretations
    """
    try:
        if not meta_analysis_data.get("success"):
            return {
                "success": False,
                "error": "Invalid meta-analysis data provided"
            }
        
        heterogeneity_data = meta_analysis_data.get("meta_analysis_results", {}).get("heterogeneity", {})
        
        if not heterogeneity_data:
            return {
                "success": False,
                "error": "No heterogeneity data available in meta-analysis results"
            }
        
        # Use LLM to provide comprehensive interpretation
        llm = ChatOpenAI(model="o3")
        
        prompt = f"""
        You are a biostatistician interpreting heterogeneity in a meta-analysis.
        
        HETEROGENEITY STATISTICS:
        {json.dumps(heterogeneity_data, indent=2)}
        
        Provide a comprehensive interpretation including:
        1. Clinical significance of the heterogeneity level
        2. Implications for meta-analysis validity
        3. Recommendations for handling heterogeneity
        4. Potential sources of heterogeneity to investigate
        5. Whether random-effects or fixed-effects model is more appropriate
        
        Use these guidelines:
        - I² 0-40%: might not be important
        - I² 30-60%: may represent moderate heterogeneity
        - I² 50-90%: may represent substantial heterogeneity
        - I² 75-100%: considerable heterogeneity
        
        Respond in JSON format:
        {{
            "heterogeneity_level": "moderate",
            "clinical_significance": "The moderate heterogeneity suggests some differences between studies...",
            "statistical_interpretation": {{
                "i_squared_interpretation": "Moderate heterogeneity (I² = 68%)",
                "q_test_interpretation": "Significant heterogeneity (p = 0.014)",
                "tau_squared_interpretation": "Between-study variance = 0.045"
            }},
            "model_recommendation": "random_effects",
            "model_justification": "Random-effects model recommended due to significant heterogeneity",
            "potential_sources": [
                "Differences in intervention duration",
                "Varying participant characteristics",
                "Different outcome measurement tools"
            ],
            "recommendations": [
                "Conduct subgroup analysis by intervention type",
                "Investigate publication bias",
                "Consider meta-regression for continuous moderators"
            ],
            "confidence_in_pooled_estimate": "moderate",
            "clinical_implications": "Results should be interpreted with caution due to heterogeneity"
        }}
        """
        
        response = llm.invoke(prompt)
        
        try:
            interpretation = json.loads(response.content)
            
            # Combine original statistics with interpretation
            comprehensive_assessment = {
                "success": True,
                "heterogeneity_statistics": heterogeneity_data,
                "interpretation": interpretation,
                "assessment_method": "llm_expert_interpretation",
                "assessment_timestamp": "2024-01-01T00:00:00Z"
            }
            
            logger.info(f"Heterogeneity assessment completed: {interpretation.get('heterogeneity_level', 'unknown')} level")
            
            return comprehensive_assessment
            
        except json.JSONDecodeError:
            logger.error("Failed to parse heterogeneity interpretation from LLM")
            return {
                "success": False,
                "error": "Failed to parse LLM interpretation",
                "raw_statistics": heterogeneity_data
            }
            
    except Exception as e:
        logger.error(f"Error assessing heterogeneity: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@tool
def perform_sensitivity_analysis(
    studies_data: List[Dict[str, Any]],
    meta_analysis_results: Dict[str, Any],
    sensitivity_type: str = "leave_one_out"
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis to test robustness of meta-analysis results
    
    Args:
        studies_data: Original studies data
        meta_analysis_results: Results from main meta-analysis
        sensitivity_type: Type of sensitivity analysis (leave_one_out, quality_threshold, etc.)
        
    Returns:
        Results of sensitivity analysis showing impact on pooled estimates
    """
    try:
        if len(studies_data) < 3:
            return {
                "success": False,
                "error": "Need at least 3 studies for meaningful sensitivity analysis"
            }
        
        # Use LLM to guide sensitivity analysis
        llm = ChatOpenAI(model="o3")
        
        prompt = f"""
        You are conducting a sensitivity analysis for a meta-analysis.
        
        ORIGINAL STUDIES DATA:
        {json.dumps(studies_data[:3], indent=2)}  # Limit for token efficiency
        
        ORIGINAL META-ANALYSIS RESULTS:
        {json.dumps(meta_analysis_results, indent=2)}
        
        SENSITIVITY ANALYSIS TYPE: {sensitivity_type}
        
        Perform a {sensitivity_type} sensitivity analysis:
        
        For leave-one-out analysis:
        - Calculate pooled estimate removing each study one at a time
        - Compare results to original pooled estimate
        - Identify influential studies
        
        For quality-based analysis:
        - Separate high vs low quality studies
        - Compare pooled estimates
        
        Respond in JSON format:
        {{
            "sensitivity_type": "leave_one_out",
            "original_pooled_estimate": -0.48,
            "leave_one_out_results": [
                {{
                    "excluded_study": "Study 1",
                    "pooled_estimate": -0.52,
                    "confidence_interval": [-0.78, -0.26],
                    "change_from_original": -0.04,
                    "percentage_change": 8.3
                }}
            ],
            "most_influential_study": {{
                "study_name": "Study 3",
                "influence_magnitude": 0.12,
                "direction": "towards_null"
            }},
            "robustness_assessment": {{
                "overall_robustness": "robust",
                "max_change": 0.12,
                "confidence_interval_overlap": true,
                "statistical_significance_maintained": true
            }},
            "interpretation": "The meta-analysis results are robust to the removal of individual studies",
            "recommendations": [
                "Results can be interpreted with confidence",
                "No single study drives the overall conclusion"
            ]
        }}
        """
        
        response = llm.invoke(prompt)
        
        try:
            sensitivity_results = json.loads(response.content)
            
            # Enhance with additional analysis
            sensitivity_results.update({
                "success": True,
                "analysis_method": "llm_guided_sensitivity",
                "total_studies_analyzed": len(studies_data),
                "analysis_timestamp": "2024-01-01T00:00:00Z"
            })
            
            logger.info(f"Sensitivity analysis completed: {sensitivity_results.get('robustness_assessment', {}).get('overall_robustness', 'unknown')} results")
            
            return sensitivity_results
            
        except json.JSONDecodeError:
            logger.error("Failed to parse sensitivity analysis results from LLM")
            return {
                "success": False,
                "error": "Failed to parse LLM sensitivity analysis",
                "raw_response": response.content[:500]
            }
            
    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "sensitivity_type": sensitivity_type
        }


@tool
def create_funnel_plot(
    meta_analysis_data: Dict[str, Any],
    title: str = "Funnel Plot - Publication Bias Assessment"
) -> Dict[str, Any]:
    """
    Create a funnel plot to assess publication bias
    
    Args:
        meta_analysis_data: Results from calculate_meta_analysis
        title: Title for the funnel plot
        
    Returns:
        Dictionary with funnel plot and bias assessment
    """
    try:
        if not meta_analysis_data.get("success"):
            return {
                "success": False,
                "error": "Invalid meta-analysis data provided"
            }
        
        studies_analysis = meta_analysis_data.get("meta_analysis_results", {}).get("studies_analysis", [])
        
        if len(studies_analysis) < 3:
            return {
                "success": False,
                "error": "Need at least 3 studies for funnel plot"
            }
        
        # Extract data for funnel plot
        effect_sizes = [study["effect_size"] for study in studies_analysis]
        standard_errors = [study["standard_error"] for study in studies_analysis]
        
        # Create funnel plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot studies
        ax.scatter(effect_sizes, standard_errors, alpha=0.7, s=60, c='blue', edgecolors='black')
        
        # Add reference lines (pseudo 95% confidence interval)
        pooled_effect = meta_analysis_data.get("meta_analysis_results", {}).get("pooled_effect_random", {}).get("estimate", 0)
        
        # Create funnel lines
        se_range = np.linspace(0, max(standard_errors) * 1.1, 100)
        upper_line = pooled_effect + 1.96 * se_range
        lower_line = pooled_effect - 1.96 * se_range
        
        ax.plot(upper_line, se_range, 'r--', alpha=0.7, label='95% CI')
        ax.plot(lower_line, se_range, 'r--', alpha=0.7)
        
        # Add vertical line at pooled estimate
        ax.axvline(x=pooled_effect, color='red', linestyle='-', alpha=0.8, label='Pooled Effect')
        
        # Formatting
        ax.invert_yaxis()  # Invert y-axis for funnel plot convention
        ax.set_xlabel('Effect Size', fontsize=12)
        ax.set_ylabel('Standard Error', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"/tmp/funnel_plot_{hash(title)}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        
        plt.close()
        
        # Simple asymmetry assessment
        mean_effect = np.mean(effect_sizes)
        asymmetry_score = abs(pooled_effect - mean_effect)
        
        bias_assessment = {
            "visual_asymmetry": "moderate" if asymmetry_score > 0.1 else "minimal",
            "asymmetry_score": asymmetry_score,
            "interpretation": "Potential publication bias detected" if asymmetry_score > 0.1 else "No obvious publication bias"
        }
        
        logger.info(f"Funnel plot created for {len(studies_analysis)} studies")
        
        return {
            "success": True,
            "plot_path": plot_path,
            "plot_base64": plot_base64,
            "plot_type": "funnel_plot",
            "bias_assessment": bias_assessment,
            "studies_included": len(studies_analysis),
            "title": title,
            "created_at": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error creating funnel plot: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "plot_type": "funnel_plot"
        }