"""
Writing tools for generating structured meta-analysis reports.
AI-first approach using LLMs for intelligent report generation and citation formatting.
"""

import json
import uuid
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

from ..config.settings import settings


# Initialize LLM
llm = ChatOpenAI(**settings.get_openai_config())


@tool
def generate_report_section_with_llm(
    section_type: str,
    content_data: Dict[str, Any],
    pico: Dict[str, str],
    writing_style: str = "academic"
) -> str:
    """
    Generate a specific section of the meta-analysis report using LLM.
    
    Args:
        section_type: Type of section (abstract, introduction, methods, results, discussion, conclusion)
        content_data: Data to include in the section
        pico: PICO framework for context
        writing_style: Writing style (academic, clinical, accessible)
    
    Returns:
        Generated section content in HTML format
    """
    
    section_prompts = {
        "abstract": """
        Write a structured abstract for this meta-analysis following PRISMA guidelines:
        
        PICO Context:
        Population: {population}
        Intervention: {intervention}
        Comparison: {comparison}
        Outcome: {outcome}
        
        Data: {content_data}
        
        Structure the abstract with:
        1. Background & Objectives
        2. Methods (search strategy, inclusion criteria, statistical analysis)
        3. Results (studies included, participants, main findings)
        4. Conclusions (clinical implications, limitations)
        
        Write in {style} style. Return HTML with appropriate headings.
        """,
        
        "introduction": """
        Write a comprehensive introduction for this meta-analysis:
        
        PICO Context:
        Population: {population}
        Intervention: {intervention}
        Comparison: {comparison}
        Outcome: {outcome}
        
        Background Data: {content_data}
        
        Include:
        1. Clinical background and rationale
        2. Current state of evidence
        3. Gaps in knowledge
        4. Objectives and research questions
        5. PICO framework explanation
        
        Write in {style} style. Return HTML with proper structure.
        """,
        
        "methods": """
        Write a detailed methods section for this meta-analysis:
        
        PICO Framework: {pico}
        Methods Data: {content_data}
        
        Include:
        1. Protocol and registration
        2. Search strategy and databases
        3. Inclusion and exclusion criteria
        4. Study selection process
        5. Data extraction methods
        6. Quality assessment approach
        7. Statistical analysis methods
        8. Heterogeneity assessment
        
        Follow PRISMA reporting guidelines. Write in {style} style.
        Return HTML with clear subsections.
        """,
        
        "results": """
        Write a comprehensive results section for this meta-analysis:
        
        PICO Context: {pico}
        Results Data: {content_data}
        
        Include:
        1. Study selection flowchart description
        2. Study characteristics summary
        3. Quality assessment results
        4. Quantitative synthesis results
        5. Heterogeneity analysis
        6. Subgroup analyses (if applicable)
        7. Publication bias assessment
        
        Present results clearly with reference to figures/tables.
        Write in {style} style. Return HTML with proper structure.
        """,
        
        "discussion": """
        Write a thorough discussion section for this meta-analysis:
        
        PICO Context: {pico}
        Results Summary: {content_data}
        
        Include:
        1. Summary of main findings
        2. Clinical interpretation and significance
        3. Comparison with previous reviews
        4. Strengths and limitations
        5. Heterogeneity interpretation
        6. Clinical implications
        7. Future research directions
        
        Write in {style} style with balanced interpretation.
        Return HTML with clear structure.
        """,
        
        "conclusion": """
        Write a concise conclusion section for this meta-analysis:
        
        PICO Context: {pico}
        Key Findings: {content_data}
        
        Include:
        1. Clear statement of findings
        2. Clinical recommendations
        3. Certainty of evidence
        4. Key limitations
        5. Practice implications
        
        Keep concise but comprehensive. Write in {style} style.
        Return HTML format.
        """
    }
    
    if section_type not in section_prompts:
        return f"<p>Error: Unknown section type '{section_type}'</p>"
    
    try:
        prompt = ChatPromptTemplate.from_template(section_prompts[section_type])
        
        response = llm.invoke(prompt.format(
            population=pico.get("P", ""),
            intervention=pico.get("I", ""),
            comparison=pico.get("C", ""),
            outcome=pico.get("O", ""),
            pico=json.dumps(pico, indent=2),
            content_data=json.dumps(content_data, indent=2)[:3000],  # Limit content
            style=writing_style
        ))
        
        return response.content
        
    except Exception as e:
        return f"<p>Error generating {section_type} section: {str(e)}</p>"


@tool
def format_citations_with_llm(
    citations: List[Dict[str, str]],
    citation_style: str = "vancouver"
) -> str:
    """
    Format citations using LLM with specified citation style.
    
    Args:
        citations: List of citation dictionaries
        citation_style: Citation style (vancouver, apa, harvard, chicago)
    
    Returns:
        Formatted citations in HTML
    """
    
    prompt = ChatPromptTemplate.from_template("""
    Format these citations in {style} style:
    
    Citations: {citations}
    
    Requirements:
    - Use proper {style} formatting
    - Number citations sequentially
    - Include all available information
    - Format as HTML ordered list
    - Handle missing information appropriately
    
    Return only the formatted HTML citation list.
    """)
    
    try:
        response = llm.invoke(prompt.format(
            citations=json.dumps(citations, indent=2),
            style=citation_style
        ))
        
        return response.content
        
    except Exception as e:
        return f"<p>Error formatting citations: {str(e)}</p>"


@tool
def create_complete_report(
    meta_analysis_data: Dict[str, Any],
    pico: Dict[str, str],
    title: Optional[str] = None,
    writing_style: str = "academic"
) -> str:
    """
    Create a complete meta-analysis report with all sections.
    
    Args:
        meta_analysis_data: Complete meta-analysis data
        pico: PICO framework
        title: Report title (auto-generated if not provided)
        writing_style: Writing style for the report
    
    Returns:
        Complete HTML report
    """
    
    try:
        # Generate title if not provided
        if not title:
            title = generate_title_with_llm(pico)
        
        # Extract data for different sections
        abstract_data = {
            "studies_included": len(meta_analysis_data.get("processed_articles", [])),
            "participants": meta_analysis_data.get("statistical_analysis", {}).get("total_participants", 0),
            "main_findings": meta_analysis_data.get("statistical_analysis", {}),
            "quality_assessment": meta_analysis_data.get("quality_assessments", {})
        }
        
        methods_data = {
            "search_strategy": meta_analysis_data.get("search_queries", []),
            "inclusion_criteria": meta_analysis_data.get("inclusion_criteria", []),
            "exclusion_criteria": meta_analysis_data.get("exclusion_criteria", []),
            "databases_searched": meta_analysis_data.get("search_domains", []),
            "analysis_methods": "Random-effects meta-analysis using inverse variance weighting"
        }
        
        results_data = {
            "study_selection": {
                "total_found": len(meta_analysis_data.get("candidate_urls", [])),
                "included": len(meta_analysis_data.get("processed_articles", [])),
                "excluded": len(meta_analysis_data.get("failed_urls", []))
            },
            "statistical_analysis": meta_analysis_data.get("statistical_analysis", {}),
            "heterogeneity": meta_analysis_data.get("heterogeneity_analysis", {}),
            "quality_scores": meta_analysis_data.get("quality_assessments", {})
        }
        
        discussion_data = {
            "main_findings": meta_analysis_data.get("statistical_analysis", {}),
            "quality_evidence": meta_analysis_data.get("quality_assessments", {}),
            "limitations": extract_limitations(meta_analysis_data),
            "clinical_implications": meta_analysis_data.get("clinical_interpretation", {})
        }
        
        # Generate each section
        sections = {}
        section_types = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        
        for section in section_types:
            if section == "abstract":
                data = abstract_data
            elif section == "methods":
                data = methods_data
            elif section == "results":
                data = results_data
            elif section in ["discussion", "conclusion"]:
                data = discussion_data
            else:
                data = meta_analysis_data
            
            sections[section] = generate_report_section_with_llm(
                section_type=section,
                content_data=data,
                pico=pico,
                writing_style=writing_style
            )
        
        # Format citations
        citations = meta_analysis_data.get("citations", [])
        formatted_citations = format_citations_with_llm(citations) if citations else "<p>No citations available</p>"
        
        # Create complete HTML report
        html_report = create_html_template(
            title=title,
            sections=sections,
            citations=formatted_citations,
            meta_analysis_data=meta_analysis_data,
            pico=pico
        )
        
        return html_report
        
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>Failed to create report: {str(e)}</p></body></html>"


@tool
def generate_title_with_llm(pico: Dict[str, str]) -> str:
    """
    Generate an appropriate title for the meta-analysis using LLM.
    
    Args:
        pico: PICO framework
    
    Returns:
        Generated title
    """
    
    prompt = ChatPromptTemplate.from_template("""
    Generate a clear, informative title for a systematic review and meta-analysis based on this PICO:
    
    Population: {population}
    Intervention: {intervention}
    Comparison: {comparison}
    Outcome: {outcome}
    
    Requirements:
    - Include key elements from PICO
    - Specify it's a systematic review and meta-analysis
    - Be specific but concise
    - Follow academic conventions
    
    Return only the title, no other text.
    """)
    
    try:
        response = llm.invoke(prompt.format(
            population=pico.get("P", ""),
            intervention=pico.get("I", ""),
            comparison=pico.get("C", ""),
            outcome=pico.get("O", "")
        ))
        
        return response.content.strip()
        
    except Exception as e:
        return f"Meta-analysis of {pico.get('I', 'intervention')} versus {pico.get('C', 'comparison')} for {pico.get('O', 'outcome')}"


def create_html_template(
    title: str,
    sections: Dict[str, str],
    citations: str,
    meta_analysis_data: Dict[str, Any],
    pico: Dict[str, str]
) -> str:
    """Create complete HTML template for the report."""
    
    # Get current date
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Extract key statistics
    stats = meta_analysis_data.get("statistical_analysis", {})
    total_studies = stats.get("total_studies", 0)
    total_participants = stats.get("total_participants", 0)
    effect_size = stats.get("pooled_effect", {}).get("effect_size", 0)
    
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
                background-color: #ffffff;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 20px;
            }}
            
            h1 {{
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            
            .meta-info {{
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 20px;
            }}
            
            .summary-box {{
                background-color: #ecf0f1;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            
            .pico-box {{
                background-color: #e8f5e8;
                border: 1px solid #27ae60;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            
            h2 {{
                color: #2c3e50;
                font-size: 20px;
                margin-top: 30px;
                margin-bottom: 15px;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            
            h3 {{
                color: #34495e;
                font-size: 16px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            
            .section {{
                margin-bottom: 30px;
            }}
            
            .statistics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            
            .stat-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #e9ecef;
            }}
            
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                display: block;
            }}
            
            .stat-label {{
                font-size: 14px;
                color: #6c757d;
                margin-top: 5px;
            }}
            
            .citation-section {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
            }}
            
            ol {{
                padding-left: 20px;
            }}
            
            li {{
                margin-bottom: 8px;
            }}
            
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #bdc3c7;
                text-align: center;
                color: #7f8c8d;
                font-size: 12px;
            }}
            
            @media print {{
                body {{ margin: 0; padding: 15px; }}
                .header {{ page-break-after: avoid; }}
                h2 {{ page-break-after: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <div class="meta-info">
                Systematic Review and Meta-Analysis<br>
                Generated on {current_date}<br>
                Metanalyst-Agent v1.0
            </div>
        </div>
        
        <div class="summary-box">
            <h3>Key Findings Summary</h3>
            <div class="statistics">
                <div class="stat-card">
                    <span class="stat-number">{total_studies}</span>
                    <div class="stat-label">Studies Included</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{total_participants:,}</span>
                    <div class="stat-label">Total Participants</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{effect_size:.3f}</span>
                    <div class="stat-label">Pooled Effect Size</div>
                </div>
            </div>
        </div>
        
        <div class="pico-box">
            <h3>PICO Framework</h3>
            <p><strong>Population:</strong> {pico.get('P', 'Not specified')}</p>
            <p><strong>Intervention:</strong> {pico.get('I', 'Not specified')}</p>
            <p><strong>Comparison:</strong> {pico.get('C', 'Not specified')}</p>
            <p><strong>Outcome:</strong> {pico.get('O', 'Not specified')}</p>
        </div>
        
        <div class="section">
            <h2>Abstract</h2>
            {sections.get('abstract', '<p>Abstract not available</p>')}
        </div>
        
        <div class="section">
            <h2>Introduction</h2>
            {sections.get('introduction', '<p>Introduction not available</p>')}
        </div>
        
        <div class="section">
            <h2>Methods</h2>
            {sections.get('methods', '<p>Methods not available</p>')}
        </div>
        
        <div class="section">
            <h2>Results</h2>
            {sections.get('results', '<p>Results not available</p>')}
        </div>
        
        <div class="section">
            <h2>Discussion</h2>
            {sections.get('discussion', '<p>Discussion not available</p>')}
        </div>
        
        <div class="section">
            <h2>Conclusion</h2>
            {sections.get('conclusion', '<p>Conclusion not available</p>')}
        </div>
        
        <div class="citation-section">
            <h2>References</h2>
            {citations}
        </div>
        
        <div class="footer">
            Generated by Metanalyst-Agent | Nobrega Medtech | {current_date}
        </div>
    </body>
    </html>
    """
    
    return html_template


def extract_limitations(meta_analysis_data: Dict[str, Any]) -> List[str]:
    """Extract limitations from meta-analysis data."""
    
    limitations = []
    
    # Check heterogeneity
    heterogeneity = meta_analysis_data.get("heterogeneity_analysis", {})
    i_squared = heterogeneity.get("i_squared", 0)
    
    if i_squared > 50:
        limitations.append(f"Substantial heterogeneity observed (I² = {i_squared}%)")
    
    # Check sample size
    total_studies = meta_analysis_data.get("statistical_analysis", {}).get("total_studies", 0)
    if total_studies < 5:
        limitations.append(f"Small number of included studies (n = {total_studies})")
    
    # Check quality
    quality_scores = meta_analysis_data.get("quality_assessments", {})
    if quality_scores:
        mean_quality = quality_scores.get("mean_quality_score", 0)
        if mean_quality < 6:
            limitations.append("Some included studies had methodological limitations")
    
    # Check failed extractions
    failed_count = len(meta_analysis_data.get("failed_urls", []))
    total_found = len(meta_analysis_data.get("candidate_urls", []))
    
    if failed_count > 0 and total_found > 0:
        failure_rate = failed_count / total_found
        if failure_rate > 0.2:  # More than 20% failures
            limitations.append("Some relevant studies could not be processed")
    
    return limitations if limitations else ["No major limitations identified"]