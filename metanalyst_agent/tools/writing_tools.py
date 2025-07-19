"""
Ferramentas de escrita e geração de relatórios seguindo diretrizes PRISMA.
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

# LLM para escrita científica
writing_llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.2,  # Mais criativo para escrita
    api_key=config.openai_api_key
)

@tool("generate_report_section")
def generate_report_section(
    section_type: Annotated[str, "Tipo de seção (abstract, introduction, methods, results, discussion, conclusion)"],
    content_data: Annotated[Dict[str, Any], "Dados e informações para gerar a seção"],
    pico: Annotated[Dict[str, str], "Estrutura PICO para contexto"],
    writing_style: Annotated[str, "Estilo de escrita (academic, clinical, concise)"] = "academic",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> Dict[str, Any]:
    """
    Gera seção específica do relatório de meta-análise seguindo diretrizes PRISMA.
    """
    
    section_prompts = {
        "abstract": """Você é um especialista em redação científica médica.

TAREFA: Escrever um abstract estruturado para meta-análise seguindo PRISMA.

ESTRUTURA DO ABSTRACT:
1. Background (2-3 frases): contexto e justificativa
2. Methods (3-4 frases): estratégia de busca, critérios, análise
3. Results (4-5 frases): estudos incluídos, características, resultados principais
4. Conclusions (2-3 frases): interpretação, implicações clínicas

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

DADOS DISPONÍVEIS:
{content_data}

DIRETRIZES:
- Máximo 250 palavras
- Linguagem clara e objetiva
- Incluir números específicos
- Mencionar limitações se relevantes

FORMATO: HTML com tags <p> para cada seção.""",

        "introduction": """Você é um especialista em redação científica médica.

TAREFA: Escrever introdução para meta-análise seguindo diretrizes acadêmicas.

ESTRUTURA DA INTRODUÇÃO:
1. Contexto clínico e epidemiológico
2. Revisão da literatura existente
3. Lacunas do conhecimento
4. Justificativa para meta-análise
5. Objetivos específicos (PICO)

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

DADOS DISPONÍVEIS:
{content_data}

DIRETRIZES:
- 3-4 parágrafos
- Linguagem científica apropriada
- Citar necessidade de evidências
- Finalizar com objetivos claros

FORMATO: HTML com tags <p> para parágrafos.""",

        "methods": """Você é um especialista em metodologia de meta-análises.

TAREFA: Escrever seção de métodos seguindo PRISMA e Cochrane guidelines.

ESTRUTURA DOS MÉTODOS:
1. Protocol and registration
2. Eligibility criteria (PICO)
3. Information sources
4. Search strategy
5. Study selection
6. Data collection process
7. Data items
8. Risk of bias assessment
9. Summary measures
10. Statistical analysis

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

DADOS DISPONÍVEIS:
{content_data}

DIRETRIZES:
- Detalhar cada etapa metodológica
- Mencionar softwares utilizados
- Incluir critérios de inclusão/exclusão
- Descrever análise estatística

FORMATO: HTML com subtítulos <h3> e parágrafos <p>.""",

        "results": """Você é um especialista em apresentação de resultados de meta-análises.

TAREFA: Escrever seção de resultados seguindo PRISMA flow.

ESTRUTURA DOS RESULTADOS:
1. Study selection (PRISMA flow)
2. Study characteristics
3. Risk of bias assessment
4. Results of individual studies
5. Summary of evidence (forest plots)
6. Additional analyses

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

DADOS DISPONÍVEIS:
{content_data}

DIRETRIZES:
- Apresentar números e estatísticas
- Referenciar figuras e tabelas
- Incluir intervalos de confiança
- Mencionar heterogeneidade

FORMATO: HTML com subtítulos <h3> e parágrafos <p>.""",

        "discussion": """Você é um especialista em discussão científica médica.

TAREFA: Escrever discussão interpretando resultados da meta-análise.

ESTRUTURA DA DISCUSSÃO:
1. Summary of main findings
2. Comparison with previous reviews
3. Clinical implications
4. Strengths and limitations
5. Future research directions

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

DADOS DISPONÍVEIS:
{content_data}

DIRETRIZES:
- Interpretar significado clínico
- Discutir limitações honestamente
- Contextualizar com literatura
- Sugerir pesquisas futuras

FORMATO: HTML com parágrafos <p>.""",

        "conclusion": """Você é um especialista em conclusões científicas médicas.

TAREFA: Escrever conclusão concisa da meta-análise.

ESTRUTURA DA CONCLUSÃO:
1. Resposta à pergunta PICO
2. Força da evidência
3. Recomendações clínicas
4. Implicações para prática

CONTEXTO PICO:
Population: {population}
Intervention: {intervention}
Comparison: {comparison}
Outcome: {outcome}

DADOS DISPONÍVEIS:
{content_data}

DIRETRIZES:
- Máximo 200 palavras
- Linguagem clara e objetiva
- Evitar over-interpretation
- Dar direções práticas

FORMATO: HTML com parágrafos <p>."""
    }
    
    try:
        prompt_template = section_prompts.get(section_type, section_prompts["abstract"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "Gere a seção solicitada com base nos dados fornecidos.")
        ])
        
        # Preparar dados do conteúdo
        content_summary = json.dumps(content_data, indent=2)[:8000]  # Limitar tamanho
        
        chain = prompt | writing_llm
        response = chain.invoke({
            "population": pico.get("P", ""),
            "intervention": pico.get("I", ""),
            "comparison": pico.get("C", ""),
            "outcome": pico.get("O", ""),
            "content_data": content_summary
        })
        
        return {
            "success": True,
            "section_type": section_type,
            "content": response.content,
            "word_count": len(response.content.split()),
            "writing_style": writing_style,
            "generation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "section_type": section_type
        }

@tool("format_html_report")
def format_html_report(
    sections: Annotated[Dict[str, str], "Dicionário com seções do relatório"],
    meta_analysis_data: Annotated[Dict[str, Any], "Dados gerais da meta-análise"],
    figures: Annotated[List[str], "Caminhos das figuras geradas"] = None,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> str:
    """
    Formata relatório completo em HTML com estilo científico apropriado.
    """
    
    try:
        figures = figures or []
        
        # CSS para estilo científico
        css_style = """
        <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 20px;
        }
        .abstract {
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
        }
        .keywords {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #ddd;
        }
        .citation {
            font-size: 0.9em;
            color: #666;
        }
        .meta-info {
            background-color: #f8f9fa;
            padding: 15px;
            border: 1px solid #ddd;
            margin: 20px 0;
        }
        </style>
        """
        
        # Título e metadados
        title = meta_analysis_data.get("research_question", "Meta-análise Sistemática")
        authors = "Metanalyst-Agent System"
        date = datetime.now().strftime("%d de %B de %Y")
        
        # Construir HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {css_style}
        </head>
        <body>
            <h1>{title}</h1>
            
            <div class="meta-info">
                <p><strong>Autores:</strong> {authors}</p>
                <p><strong>Data:</strong> {date}</p>
                <p><strong>Tipo de estudo:</strong> Meta-análise sistemática</p>
                <p><strong>ID da análise:</strong> {meta_analysis_data.get('meta_analysis_id', 'N/A')}</p>
            </div>
        """
        
        # Adicionar seções na ordem apropriada
        section_order = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        section_titles = {
            "abstract": "Resumo",
            "introduction": "Introdução", 
            "methods": "Métodos",
            "results": "Resultados",
            "discussion": "Discussão",
            "conclusion": "Conclusão"
        }
        
        for section in section_order:
            if section in sections and sections[section]:
                title_section = section_titles.get(section, section.title())
                
                if section == "abstract":
                    html_content += f"""
                    <div class="abstract">
                        <h2>{title_section}</h2>
                        {sections[section]}
                    </div>
                    """
                else:
                    html_content += f"""
                    <h2>{title_section}</h2>
                    {sections[section]}
                    """
        
        # Adicionar figuras se disponíveis
        if figures:
            html_content += """
            <h2>Figuras</h2>
            """
            for i, figure_path in enumerate(figures, 1):
                html_content += f"""
                <div class="figure">
                    <img src="{figure_path}" alt="Figura {i}" style="max-width: 100%;">
                    <p><strong>Figura {i}:</strong> Análise estatística gerada</p>
                </div>
                """
        
        # Adicionar informações técnicas
        if meta_analysis_data.get("citations"):
            html_content += """
            <h2>Referências</h2>
            <ol>
            """
            for citation in meta_analysis_data["citations"]:
                html_content += f"<li class='citation'>{citation}</li>"
            html_content += "</ol>"
        
        # Fechar HTML
        html_content += """
            </body>
            </html>
        """
        
        return html_content
        
    except Exception as e:
        return f"<html><body><h1>Erro na formatação</h1><p>{str(e)}</p></body></html>"

@tool("create_executive_summary")
def create_executive_summary(
    full_report_data: Annotated[Dict[str, Any], "Dados completos do relatório"],
    target_audience: Annotated[str, "Público-alvo (clinicians, researchers, patients)"] = "clinicians",
    max_words: Annotated[int, "Número máximo de palavras"] = 500,
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> str:
    """
    Cria sumário executivo adaptado ao público-alvo.
    """
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em comunicação científica médica.

TAREFA: Criar sumário executivo conciso e impactante da meta-análise.

PÚBLICO-ALVO: {target_audience}

ADAPTAÇÕES POR PÚBLICO:
- clinicians: foco em aplicabilidade clínica, recomendações práticas
- researchers: foco em metodologia, lacunas, pesquisas futuras  
- patients: linguagem simples, benefícios/riscos, recomendações práticas

ESTRUTURA DO SUMÁRIO:
1. Questão de pesquisa (1 frase)
2. Métodos principais (2-3 frases)
3. Resultados-chave (3-4 frases)
4. Conclusões práticas (2-3 frases)
5. Recomendações específicas (bullet points)

LIMITE: {max_words} palavras

DIRETRIZES:
- Linguagem clara e direta
- Números específicos quando relevantes
- Evitar jargões desnecessários
- Incluir limitações importantes
- Dar recomendações acionáveis

FORMATO: HTML simples com parágrafos e listas."""),
        ("human", "Dados do relatório:\n\n{report_data}")
    ])
    
    try:
        chain = summary_prompt | writing_llm
        response = chain.invoke({
            "target_audience": target_audience,
            "max_words": max_words,
            "report_data": json.dumps(full_report_data, indent=2)[:10000]
        })
        
        return response.content
        
    except Exception as e:
        return f"<p>Erro na geração do sumário executivo: {str(e)}</p>"

@tool("compile_citations")
def compile_citations(
    articles_data: Annotated[List[Dict[str, Any]], "Lista de artigos processados"],
    citation_style: Annotated[str, "Estilo de citação (vancouver, apa, ama)"] = "vancouver",
    state: Annotated[MetaAnalysisState, InjectedState] = None
) -> List[str]:
    """
    Compila citações formatadas de todos os artigos incluídos.
    """
    
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um especialista em citações acadêmicas médicas.

TAREFA: Gerar citações no estilo {citation_style} para todos os artigos fornecidos.

ESTILO {citation_style.upper()}:
- Formatação precisa conforme diretrizes
- Ordem alfabética por primeiro autor
- Numeração sequencial
- Abreviações padronizadas de revistas

IMPORTANTE: 
- Retorne apenas as citações formatadas
- Uma citação por linha
- Numeração consecutiva
- Verificar completude dos dados"""),
        ("human", "Artigos para citar:\n\n{articles_data}")
    ])
    
    try:
        chain = citation_prompt | writing_llm
        response = chain.invoke({
            "citation_style": citation_style,
            "articles_data": json.dumps(articles_data, indent=2)[:12000]
        })
        
        # Processar resposta em lista
        citations = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):  # Ignorar comentários
                citations.append(line)
        
        return citations
        
    except Exception as e:
        # Fallback: citações básicas
        citations = []
        for i, article in enumerate(articles_data, 1):
            title = article.get("title", "Título não disponível")
            authors = article.get("authors", ["Autor não disponível"])
            year = article.get("year", "Ano não disponível")
            
            if isinstance(authors, list):
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
            else:
                author_str = str(authors)
            
            citation = f"{i}. {author_str}. {title}. {year}."
            citations.append(citation)
        
        return citations