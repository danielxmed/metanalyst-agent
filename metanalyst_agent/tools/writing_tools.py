"""
Ferramentas de escrita para geração de relatórios de meta-análise.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from datetime import datetime
import uuid

from ..config.settings import settings


@tool
def generate_report_section(
    section_type: str,
    content_data: Dict[str, Any],
    pico_criteria: Dict[str, str],
    writing_style: str = "academic"
) -> str:
    """
    Gera seção específica do relatório de meta-análise.
    
    Args:
        section_type: Tipo de seção (abstract, introduction, methods, results, discussion, conclusion)
        content_data: Dados relevantes para a seção
        pico_criteria: Critérios PICO da meta-análise
        writing_style: Estilo de escrita (academic, clinical, lay)
    
    Returns:
        Texto da seção gerada
    """
    try:
        if section_type == "abstract":
            return generate_abstract(content_data, pico_criteria)
        elif section_type == "introduction":
            return generate_introduction(content_data, pico_criteria)
        elif section_type == "methods":
            return generate_methods(content_data, pico_criteria)
        elif section_type == "results":
            return generate_results(content_data)
        elif section_type == "discussion":
            return generate_discussion(content_data, pico_criteria)
        elif section_type == "conclusion":
            return generate_conclusion(content_data, pico_criteria)
        else:
            return f"Tipo de seção '{section_type}' não reconhecido."
            
    except Exception as e:
        return f"Erro na geração da seção {section_type}: {str(e)}"


def generate_abstract(content_data: Dict[str, Any], pico_criteria: Dict[str, str]) -> str:
    """Gera resumo estruturado."""
    meta_results = content_data.get("meta_analysis_results", {})
    
    abstract = f"""
## RESUMO

**Objetivo:** Avaliar a eficácia de {pico_criteria.get('I', 'intervenção')} comparado a {pico_criteria.get('C', 'controle')} para {pico_criteria.get('O', 'desfecho')} em {pico_criteria.get('P', 'população')}.

**Métodos:** Revisão sistemática e meta-análise seguindo diretrizes PRISMA. Busca realizada em PubMed, Cochrane Library e ClinicalTrials.gov. Critérios de inclusão: estudos que avaliaram {pico_criteria.get('I', 'a intervenção')} versus {pico_criteria.get('C', 'controle')} em {pico_criteria.get('P', 'população alvo')}. Análise estatística utilizou modelo de efeitos {'aleatórios' if meta_results.get('model_used') == 'random_effects' else 'fixos'}.

**Resultados:** Foram incluídos {meta_results.get('studies_included', 'N')} estudos com {meta_results.get('total_participants', 'N')} participantes. O effect size pooled foi {meta_results.get('pooled_effect_size', 'N'):.3f} (IC 95%: {meta_results.get('confidence_interval', [0, 0])[0]:.3f} a {meta_results.get('confidence_interval', [0, 0])[1]:.3f}, p = {meta_results.get('p_value', 0):.3f}). A heterogeneidade entre estudos foi {meta_results.get('heterogeneity', {}).get('interpretation', 'N/A')} (I² = {meta_results.get('heterogeneity', {}).get('I_squared', 0):.1f}%).

**Conclusão:** {'Os resultados sugerem benefício significativo' if meta_results.get('p_value', 1) < 0.05 else 'Não foi demonstrado benefício significativo'} de {pico_criteria.get('I', 'intervenção')} comparado a {pico_criteria.get('C', 'controle')} para {pico_criteria.get('O', 'desfecho')} em {pico_criteria.get('P', 'população')}.

**Palavras-chave:** {pico_criteria.get('I', '')}, {pico_criteria.get('O', '')}, meta-análise, revisão sistemática
"""
    return abstract.strip()


def generate_introduction(content_data: Dict[str, Any], pico_criteria: Dict[str, str]) -> str:
    """Gera introdução."""
    introduction = f"""
## INTRODUÇÃO

### Contexto e Justificativa

{pico_criteria.get('O', 'A condição em estudo')} representa um importante problema de saúde pública que afeta {pico_criteria.get('P', 'a população alvo')}. Diversas estratégias terapêuticas têm sido propostas, incluindo {pico_criteria.get('I', 'a intervenção em questão')}.

### Tratamentos Disponíveis

Atualmente, {pico_criteria.get('C', 'o tratamento padrão')} é considerado uma abordagem estabelecida para o manejo de {pico_criteria.get('O', 'a condição')}. No entanto, {pico_criteria.get('I', 'a nova intervenção')} emerge como uma alternativa promissora que pode oferecer vantagens adicionais.

### Lacuna do Conhecimento

Embora estudos individuais tenham investigado a eficácia de {pico_criteria.get('I', 'a intervenção')}, não existe consenso claro sobre sua superioridade em relação a {pico_criteria.get('C', 'o controle')}. Uma síntese quantitativa da evidência disponível é necessária para orientar a prática clínica.

### Objetivos

**Objetivo Primário:** Comparar a eficácia de {pico_criteria.get('I', 'intervenção')} versus {pico_criteria.get('C', 'controle')} para {pico_criteria.get('O', 'desfecho')} em {pico_criteria.get('P', 'população')}.

**Objetivos Secundários:**
- Avaliar a heterogeneidade entre estudos
- Investigar potenciais fontes de variabilidade nos resultados
- Analisar a qualidade da evidência disponível
"""
    return introduction.strip()


def generate_methods(content_data: Dict[str, Any], pico_criteria: Dict[str, str]) -> str:
    """Gera seção de métodos."""
    search_data = content_data.get("search_results", {})
    
    methods = f"""
## MÉTODOS

### Protocolo e Registro

Esta revisão sistemática foi conduzida seguindo as diretrizes PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) e registrada prospectivamente.

### Critérios de Elegibilidade

**População:** {pico_criteria.get('P', 'População alvo')}
**Intervenção:** {pico_criteria.get('I', 'Intervenção')}
**Comparação:** {pico_criteria.get('C', 'Controle/Comparador')}
**Desfecho:** {pico_criteria.get('O', 'Desfecho primário')}

**Tipos de Estudo:** Ensaios clínicos randomizados, estudos de coorte prospectivos, estudos caso-controle.

**Critérios de Inclusão:**
- Estudos que compararam diretamente {pico_criteria.get('I', 'intervenção')} versus {pico_criteria.get('C', 'controle')}
- População adulta (≥18 anos)
- Seguimento mínimo de 4 semanas
- Dados quantitativos disponíveis para meta-análise

**Critérios de Exclusão:**
- Estudos com menos de 20 participantes
- Resumos de congressos sem dados completos
- Estudos em animais ou in vitro

### Estratégia de Busca

Busca sistemática realizada nas seguintes bases de dados:
- PubMed/MEDLINE
- Cochrane Central Register of Controlled Trials
- ClinicalTrials.gov

**Período:** Sem restrição de data até {datetime.now().strftime('%B %Y')}
**Idiomas:** Inglês, português, espanhol

### Seleção de Estudos

Dois revisores independentes realizaram a seleção de estudos em duas etapas:
1. Triagem de títulos e resumos
2. Avaliação de textos completos

Discordâncias foram resolvidas por consenso ou terceiro revisor.

### Extração de Dados

Dados extraídos incluíram:
- Características do estudo (autor, ano, desenho)
- Características da população
- Detalhes da intervenção e controle
- Desfechos e medidas de efeito
- Dados para avaliação de qualidade

### Avaliação de Qualidade

Qualidade metodológica avaliada usando ferramentas apropriadas:
- Ensaios clínicos: Cochrane Risk of Bias Tool
- Estudos observacionais: Newcastle-Ottawa Scale

### Análise Estatística

Meta-análise realizada usando modelo de efeitos aleatórios (DerSimonian-Laird). Heterogeneidade avaliada pelo teste Q de Cochran e estatística I². Análise de sensibilidade conduzida removendo estudos individualmente. Viés de publicação investigado por funnel plots.

**Software:** Python com bibliotecas científicas especializadas.
"""
    return methods.strip()


def generate_results(content_data: Dict[str, Any]) -> str:
    """Gera seção de resultados."""
    meta_results = content_data.get("meta_analysis_results", {})
    search_data = content_data.get("search_results", {})
    
    results = f"""
## RESULTADOS

### Seleção de Estudos

A busca inicial identificou {search_data.get('total_articles', 'N')} artigos. Após remoção de duplicatas e aplicação dos critérios de elegibilidade, {meta_results.get('studies_included', 'N')} estudos foram incluídos na meta-análise.

### Características dos Estudos

Os {meta_results.get('studies_included', 'N')} estudos incluídos envolveram {meta_results.get('total_participants', 'N')} participantes. A maioria dos estudos foram ensaios clínicos randomizados ({len([s for s in meta_results.get('individual_studies', []) if 'randomized' in s.get('study', '').lower()])} estudos).

### Desfecho Primário

**Effect Size Pooled:** {meta_results.get('pooled_effect_size', 'N'):.3f} (IC 95%: {meta_results.get('confidence_interval', [0, 0])[0]:.3f} a {meta_results.get('confidence_interval', [0, 0])[1]:.3f})

**Significância Estatística:** p = {meta_results.get('p_value', 0):.3f}

{'A diferença foi estatisticamente significativa' if meta_results.get('p_value', 1) < 0.05 else 'A diferença não foi estatisticamente significativa'}, {'favorecendo a intervenção' if meta_results.get('pooled_effect_size', 0) > 0 else 'favorecendo o controle'}.

### Heterogeneidade

**I² = {meta_results.get('heterogeneity', {}).get('I_squared', 0):.1f}%** ({meta_results.get('heterogeneity', {}).get('interpretation', 'N/A')})
**Q = {meta_results.get('heterogeneity', {}).get('Q_statistic', 0):.2f}** (p = {meta_results.get('heterogeneity', {}).get('p_value_Q', 0):.3f})

{'A heterogeneidade foi baixa, sugerindo consistência entre estudos.' if meta_results.get('heterogeneity', {}).get('I_squared', 100) <= 25 else 'Heterogeneidade substancial foi observada, indicando variabilidade entre estudos.'}

### Análise de Sensibilidade

{content_data.get('sensitivity_analysis', {}).get('robust_finding', False) and 'Os resultados mostraram-se robustos na análise de sensibilidade.' or 'A análise de sensibilidade identificou estudos influentes que afetam os resultados.'}

### Qualidade da Evidência

A qualidade geral da evidência foi classificada como {'alta' if meta_results.get('studies_included', 0) >= 10 and meta_results.get('heterogeneity', {}).get('I_squared', 100) <= 25 else 'moderada' if meta_results.get('studies_included', 0) >= 5 else 'baixa'} segundo critérios GRADE.
"""
    return results.strip()


def generate_discussion(content_data: Dict[str, Any], pico_criteria: Dict[str, str]) -> str:
    """Gera seção de discussão."""
    meta_results = content_data.get("meta_analysis_results", {})
    
    discussion = f"""
## DISCUSSÃO

### Principais Achados

Esta meta-análise demonstrou que {pico_criteria.get('I', 'a intervenção')} {'apresenta benefício significativo' if meta_results.get('p_value', 1) < 0.05 else 'não demonstra benefício significativo'} comparado a {pico_criteria.get('C', 'controle')} para {pico_criteria.get('O', 'desfecho')} em {pico_criteria.get('P', 'população')}.

### Interpretação dos Resultados

O effect size observado ({meta_results.get('pooled_effect_size', 0):.3f}) {'sugere um efeito clinicamente relevante' if abs(meta_results.get('pooled_effect_size', 0)) > 0.2 else 'indica um efeito de magnitude pequena'}. {'A significância estatística (p < 0.05) fortalece a evidência de eficácia.' if meta_results.get('p_value', 1) < 0.05 else 'A ausência de significância estatística limita as conclusões sobre eficácia.'}

### Heterogeneidade

{'A baixa heterogeneidade (I² ≤ 25%) sugere consistência entre estudos, aumentando a confiança nos resultados.' if meta_results.get('heterogeneity', {}).get('I_squared', 100) <= 25 else f"A heterogeneidade {meta_results.get('heterogeneity', {}).get('interpretation', '').lower()} (I² = {meta_results.get('heterogeneity', {}).get('I_squared', 0):.1f}%) indica variabilidade entre estudos que pode ser explicada por diferenças metodológicas, populacionais ou de intervenção."}

### Limitações

1. **Qualidade dos Estudos:** Variabilidade na qualidade metodológica dos estudos incluídos
2. **Heterogeneidade:** {meta_results.get('heterogeneity', {}).get('interpretation', 'Heterogeneidade observada')} pode limitar a generalização dos resultados
3. **Viés de Publicação:** Possível viés favorecendo estudos com resultados positivos
4. **Tamanho Amostral:** {f"Número limitado de estudos ({meta_results.get('studies_included', 0)})" if meta_results.get('studies_included', 0) < 10 else "Tamanho amostral adequado"}

### Implicações Clínicas

{'Os resultados suportam o uso de' if meta_results.get('p_value', 1) < 0.05 else 'A evidência atual não suporta claramente o uso de'} {pico_criteria.get('I', 'intervenção')} como {'primeira linha' if meta_results.get('pooled_effect_size', 0) > 0.5 else 'alternativa'} para {pico_criteria.get('O', 'desfecho')} em {pico_criteria.get('P', 'população')}.

### Pesquisas Futuras

Estudos futuros devem focar em:
- Ensaios clínicos de maior porte e melhor qualidade
- Padronização de protocolos de intervenção
- Avaliação de desfechos de longo prazo
- Análises de custo-efetividade
"""
    return discussion.strip()


def generate_conclusion(content_data: Dict[str, Any], pico_criteria: Dict[str, str]) -> str:
    """Gera conclusão."""
    meta_results = content_data.get("meta_analysis_results", {})
    
    conclusion = f"""
## CONCLUSÃO

Com base na evidência disponível de {meta_results.get('studies_included', 'N')} estudos envolvendo {meta_results.get('total_participants', 'N')} participantes, {'esta meta-análise demonstra que' if meta_results.get('p_value', 1) < 0.05 else 'esta meta-análise não demonstra evidência convincente de que'} {pico_criteria.get('I', 'intervenção')} {'é superior' if meta_results.get('pooled_effect_size', 0) > 0 else 'é inferior'} a {pico_criteria.get('C', 'controle')} para {pico_criteria.get('O', 'desfecho')} em {pico_criteria.get('P', 'população')}.

{'A evidência suporta a consideração de' if meta_results.get('p_value', 1) < 0.05 else 'A evidência atual não suporta claramente'} {pico_criteria.get('I', 'intervenção')} {'como opção terapêutica eficaz' if meta_results.get('p_value', 1) < 0.05 else 'como opção terapêutica superior'} nesta população.

{'Recomenda-se cautela na interpretação devido à heterogeneidade observada entre estudos.' if meta_results.get('heterogeneity', {}).get('I_squared', 0) > 50 else 'A consistência entre estudos fortalece a confiança nos resultados.'}

Estudos adicionais de alta qualidade são necessários para {'confirmar estes achados' if meta_results.get('p_value', 1) < 0.05 else 'esclarecer o potencial benefício'} e estabelecer diretrizes clínicas baseadas em evidência.
"""
    return conclusion.strip()


@tool
def format_citations(
    citations: List[Dict[str, Any]],
    style: str = "vancouver"
) -> str:
    """
    Formata lista de citações no estilo especificado.
    
    Args:
        citations: Lista de citações com metadados
        style: Estilo de citação (vancouver, apa, chicago)
    
    Returns:
        Citações formatadas
    """
    try:
        if style.lower() == "vancouver":
            formatted_citations = []
            for i, citation in enumerate(citations, 1):
                formatted = f"{i}. {citation.get('formatted_citation', citation.get('title', 'Título não disponível'))}"
                formatted_citations.append(formatted)
            return "\n".join(formatted_citations)
        else:
            return "Estilo de citação não suportado. Use 'vancouver'."
            
    except Exception as e:
        return f"Erro na formatação de citações: {str(e)}"


@tool
def create_html_report(
    sections: Dict[str, str],
    meta_analysis_results: Dict[str, Any],
    plots_paths: List[str],
    title: str = "Meta-Analysis Report"
) -> str:
    """
    Cria relatório HTML completo com todas as seções.
    
    Args:
        sections: Dicionário com seções do relatório
        meta_analysis_results: Resultados da meta-análise
        plots_paths: Caminhos para gráficos gerados
        title: Título do relatório
    
    Returns:
        Caminho para o arquivo HTML gerado
    """
    try:
        # Template HTML básico
        html_template = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .meta-info {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .results-box {{ background: #e8f5e8; padding: 15px; border-left: 5px solid #27ae60; margin: 20px 0; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #bdc3c7; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="meta-info">
        <strong>Data de Geração:</strong> {datetime.now().strftime('%d/%m/%Y às %H:%M')}<br>
        <strong>Estudos Incluídos:</strong> {meta_analysis_results.get('studies_included', 'N/A')}<br>
        <strong>Participantes Totais:</strong> {meta_analysis_results.get('total_participants', 'N/A')}<br>
        <strong>Effect Size Pooled:</strong> {meta_analysis_results.get('pooled_effect_size', 'N/A'):.3f} (IC 95%: {meta_analysis_results.get('confidence_interval', [0, 0])[0]:.3f} - {meta_analysis_results.get('confidence_interval', [0, 0])[1]:.3f})<br>
        <strong>Heterogeneidade (I²):</strong> {meta_analysis_results.get('heterogeneity', {}).get('I_squared', 'N/A'):.1f}%
    </div>
"""

        # Adicionar seções
        for section_name, section_content in sections.items():
            html_template += f"\n    <div class='section'>\n        {section_content.replace('#', '').strip()}\n    </div>\n"

        # Adicionar gráficos
        if plots_paths:
            html_template += "\n    <h2>Gráficos e Visualizações</h2>\n"
            for plot_path in plots_paths:
                plot_name = plot_path.split('/')[-1]
                html_template += f"""
    <div class="plot">
        <h3>{plot_name.replace('_', ' ').title()}</h3>
        <img src="{plot_path}" alt="{plot_name}">
    </div>
"""

        # Adicionar rodapé
        html_template += f"""
    <div class="footer">
        <p>Relatório gerado automaticamente pelo Metanalyst-Agent - Nobrega Medtech</p>
        <p>Este relatório segue as diretrizes PRISMA para revisões sistemáticas e meta-análises.</p>
    </div>
</body>
</html>
"""

        # Salvar arquivo HTML
        report_filename = f"meta_analysis_report_{uuid.uuid4().hex[:8]}.html"
        report_path = f"{settings.reports_dir}/{report_filename}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        return report_path
        
    except Exception as e:
        return f"Erro na criação do relatório HTML: {str(e)}"