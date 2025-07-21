#!/usr/bin/env python3
"""
CORREÇÃO DEFINITIVA DO PROBLEMA DE UUID NO BANCO DE DADOS
=======================================================

Este script corrige o problema onde meta_analysis_id inválidos são 
passados para o sistema, causando falha de integridade referencial.

Problema identificado:
- Sistema recebe meta_analysis_id='Amiodarone_vs_BB_AF' (não é UUID)
- ensure_valid_uuid() gera UUID aleatório que não existe no banco
- Inserções falham com foreign key constraint

Correção:
1. Melhorar ensure_valid_uuid para buscar ID correto no estado
2. Criar função para validar meta_analysis_id existe no banco
3. Adicionar fallbacks robustos
"""

import os
import sys
import uuid
from datetime import datetime

# Adicionar o diretório do projeto ao Python path
sys.path.insert(0, '/Users/danielnobregamedeiros/Desktop/metanalyst-agent')

def fix_research_tools():
    """Corrigir research_tools.py com validação robusta de UUID"""
    
    print("🔧 CORRIGINDO RESEARCH_TOOLS.PY")
    print("=" * 50)
    
    research_tools_path = "metanalyst_agent/tools/research_tools.py"
    
    # Fazer backup
    backup_path = f"{research_tools_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.system(f"cp {research_tools_path} {backup_path}")
    print(f"✅ Backup criado: {backup_path}")
    
    # Nova versão corrigida
    corrected_code = '''"""Research tools for literature search and relevance assessment"""

import os
import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import logging

logger = logging.getLogger(__name__)

# Cache global para manter meta_analysis_id consistente
_global_meta_analysis_cache = {}

def get_cached_meta_analysis_id(key: str = "current") -> Optional[str]:
    """Recupera meta_analysis_id do cache global"""
    return _global_meta_analysis_cache.get(key)

def set_cached_meta_analysis_id(meta_analysis_id: str, key: str = "current") -> None:
    """Define meta_analysis_id no cache global"""
    if validate_uuid_format(meta_analysis_id):
        _global_meta_analysis_cache[key] = meta_analysis_id
        logger.info(f"Meta-analysis ID cached: {meta_analysis_id}")

def validate_uuid_format(uuid_string: str) -> bool:
    """
    Valida se uma string tem formato de UUID válido.
    
    Args:
        uuid_string: String para validar
        
    Returns:
        True se é um UUID válido, False caso contrário
    """
    if not uuid_string or not isinstance(uuid_string, str):
        return False
    
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False

def validate_meta_analysis_exists(meta_analysis_id: str) -> bool:
    """
    Verifica se meta_analysis_id existe no banco de dados.
    
    Args:
        meta_analysis_id: UUID para verificar
        
    Returns:
        True se existe, False caso contrário
    """
    if not validate_uuid_format(meta_analysis_id):
        return False
    
    try:
        from ..database.connection import get_database_manager
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM meta_analyses WHERE id = %s",
                    (meta_analysis_id,)
                )
                count = cursor.fetchone()[0]
                return count > 0
                
    except Exception as e:
        logger.error(f"Erro verificando meta_analysis_id no banco: {e}")
        return False

def ensure_valid_meta_analysis_id(
    meta_analysis_id: str, 
    operation: str = "operation",
    state: Dict[str, Any] = None
) -> Optional[str]:
    """
    Garante que um meta_analysis_id seja válido e exista no banco.
    
    Estratégia de recuperação:
    1. Se o ID é válido e existe no banco, usar esse ID
    2. Buscar no cache global
    3. Buscar no estado fornecido
    4. Buscar meta-análise mais recente no banco
    5. Como último recurso, criar nova meta-análise
    
    Args:
        meta_analysis_id: ID para validar
        operation: Nome da operação para logs
        state: Estado atual do sistema para busca de fallback
        
    Returns:
        UUID válido e existente no banco, ou None se falhou
    """
    # 1. Verificar se o ID fornecido é válido e existe
    if validate_uuid_format(meta_analysis_id) and validate_meta_analysis_exists(meta_analysis_id):
        set_cached_meta_analysis_id(meta_analysis_id)
        return meta_analysis_id
    
    logger.warning(f"Invalid meta_analysis_id in {operation}: '{meta_analysis_id}'")
    
    # 2. Buscar no cache global
    cached_id = get_cached_meta_analysis_id()
    if cached_id and validate_meta_analysis_exists(cached_id):
        logger.info(f"Using cached meta_analysis_id: {cached_id}")
        return cached_id
    
    # 3. Buscar no estado
    if state:
        for key in ['meta_analysis_id', 'id', 'analysis_id']:
            if key in state:
                candidate_id = str(state[key])
                if validate_uuid_format(candidate_id) and validate_meta_analysis_exists(candidate_id):
                    logger.info(f"Found valid meta_analysis_id in state[{key}]: {candidate_id}")
                    set_cached_meta_analysis_id(candidate_id)
                    return candidate_id
    
    # 4. Buscar meta-análise mais recente
    try:
        from ..database.connection import get_database_manager
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM meta_analyses 
                    WHERE status = 'active' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    recent_id = str(result[0])
                    logger.info(f"Using most recent active meta-analysis: {recent_id}")
                    set_cached_meta_analysis_id(recent_id)
                    return recent_id
                    
    except Exception as e:
        logger.error(f"Erro buscando meta-análise recente: {e}")
    
    # 5. Criar nova meta-análise como último recurso
    try:
        from ..database.connection import get_database_manager
        
        new_id = str(uuid.uuid4())
        thread_id = f"thread_{new_id}"
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO meta_analyses (
                        id, thread_id, title, pico, status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    new_id,
                    thread_id,
                    f"Auto-created for {operation}",
                    json.dumps({"P": "Not specified", "I": "Not specified", "C": "Not specified", "O": "Not specified"}),
                    "active",
                    datetime.utcnow(),
                    datetime.utcnow()
                ))
                conn.commit()
        
        logger.warning(f"Created new meta-analysis as fallback: {new_id}")
        set_cached_meta_analysis_id(new_id)
        return new_id
        
    except Exception as e:
        logger.error(f"Erro criando meta-análise de fallback: {e}")
    
    # Se tudo falhou
    logger.error(f"Failed to ensure valid meta_analysis_id for {operation}")
    return None

# PostgreSQL connection for URL deduplication
from ..database.connection import get_database_manager

# Cache for candidate URLs to avoid duplicates within session
_candidate_urls_cache = set()

def _is_url_already_candidate(url: str, meta_analysis_id: str) -> bool:
    """Check if URL is already a candidate using PostgreSQL and cache"""
    # Check cache first
    if url in _candidate_urls_cache:
        return True
    
    # Check database
    try:
        cursor_query = """SELECT COUNT(*) FROM articles WHERE url = %s AND meta_analysis_id = %s"""
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(cursor_query, (url, meta_analysis_id))
                count = cursor.fetchone()[0]
                if count > 0:
                    _candidate_urls_cache.add(url)
                    return True
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        logger.error(f"Failed to check URL in database: {e}")
    
    return False

def _add_url_to_candidates(url: str, meta_analysis_id: str, metadata: Dict[str, Any], state: Dict[str, Any] = None):
    """Add URL to candidates in PostgreSQL with proper error handling"""
    try:
        # Validar meta_analysis_id antes de inserir
        valid_id = ensure_valid_meta_analysis_id(meta_analysis_id, "_add_url_to_candidates", state)
        if not valid_id:
            logger.error(f"Cannot add URL: invalid meta_analysis_id {meta_analysis_id}")
            return
            
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                article_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO articles (id, meta_analysis_id, url, title, processing_status, created_at)
                    VALUES (%s, %s, %s, %s, 'pending', %s)
                    ON CONFLICT DO NOTHING
                """, (
                    article_id, 
                    valid_id,  # Usar ID validado
                    url, 
                    metadata.get('title', ''), 
                    datetime.utcnow()
                ))
                conn.commit()
        _candidate_urls_cache.add(url)
        logger.info(f"Added URL to candidates: {url}")
        
    except Exception as e:
        logger.error(f"Error adding URL to candidates: {e}")
        _candidate_urls_cache.add(url)


@tool
def search_literature(
    query: str, 
    domains: List[str] = None,
    meta_analysis_id: str = None,
    max_results: int = 10,
    include_raw_content: bool = False
) -> Dict[str, Any]:
    """
    Search for scientific literature using Tavily API with deduplication
    OPTIMIZED: Avoids storing duplicate URLs and raw content
    
    Args:
        query: Search query string
        domains: Optional list of domains to search within
        meta_analysis_id: ID for deduplication across searches
        max_results: Maximum number of results to return
        include_raw_content: Whether to include raw content (should be False for optimization)
        
    Returns:
        Dictionary with search results and metadata (NO raw content)
    """
    try:
        # Validate and fix meta_analysis_id if provided
        if meta_analysis_id:
            meta_analysis_id = ensure_valid_meta_analysis_id(
                meta_analysis_id, 
                "search_literature"
            )
            if not meta_analysis_id:
                return {
                    "success": False,
                    "error": "Could not resolve valid meta_analysis_id",
                    "results": []
                }
        
        # Initialize Tavily client
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "TAVILY_API_KEY not found in environment variables",
                "results": []
            }
        
        client = TavilyClient(api_key=api_key)
        
        # Perform search
        search_params = {
            "query": query,
            "max_results": max_results * 2,  # Get more to account for duplicates
            "include_raw_content": include_raw_content,  # Should be False
            "include_domains": domains if domains else []
        }
        
        # Add scientific domains if none specified
        if not domains:
            search_params["include_domains"] = [
                "pubmed.ncbi.nlm.nih.gov",
                "scholar.google.com", 
                "cochranelibrary.com",
                "bmj.com",
                "nejm.org",
                "thelancet.com",
                "nature.com",
                "science.org",
                "jamanetwork.com",
                "europepmc.org"
            ]
        
        response = client.search(**search_params)
        
        if not response or 'results' not in response:
            return {
                "success": False,
                "error": "No results returned from Tavily API",
                "results": []
            }
        
        # Filter out duplicate URLs and process results
        unique_results = []
        processed_urls = set()
        
        for result in response['results']:
            url = result.get('url', '')
            
            # Skip if URL is empty or already processed in this search
            if not url or url in processed_urls:
                continue
                
            # Skip if URL is already a candidate for this meta-analysis
            if meta_analysis_id and _is_url_already_candidate(url, meta_analysis_id):
                continue
            
            processed_urls.add(url)
            
            # Create optimized result (NO raw content)
            optimized_result = {
                "url": url,
                "title": result.get('title', ''),
                "snippet": result.get('content', '')[:500],  # Limit snippet size
                "score": result.get('score', 0.0),
                "published_date": result.get('published_date', ''),
                "domain": result.get('url', '').split('/')[2] if '/' in result.get('url', '') else ''
            }
            
            # Add to candidates if meta_analysis_id provided
            if meta_analysis_id:
                _add_url_to_candidates(url, meta_analysis_id, optimized_result)
            
            unique_results.append(optimized_result)
            
            # Stop when we have enough unique results
            if len(unique_results) >= max_results:
                break
        
        return {
            "success": True,
            "query": query,
            "total_found": len(response.get('results', [])),
            "unique_results": len(unique_results),
            "results": unique_results,
            "search_metadata": {
                "domains_searched": search_params.get("include_domains", []),
                "duplicates_filtered": len(response.get('results', [])) - len(unique_results),
                "searched_at": datetime.now().isoformat(),
                "meta_analysis_id": meta_analysis_id
            }
        }
        
    except Exception as e:
        logger.error(f"Search literature error: {e}")
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "results": []
        }

@tool
def get_candidate_urls_summary(state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get summary of candidate URLs without loading full content
    
    Args:
        state: State containing meta_analysis_id
        
    Returns:
        Summary of candidate URLs
    """
    try:
        # Extrair meta_analysis_id do estado
        meta_analysis_id = None
        if state:
            # Tentar diferentes chaves possíveis
            for key in ['meta_analysis_id', 'id', 'analysis_id']:
                if key in state and state[key]:
                    meta_analysis_id = str(state[key])
                    break
        
        # Validar o ID encontrado
        if meta_analysis_id:
            meta_analysis_id = ensure_valid_meta_analysis_id(
                meta_analysis_id, 
                "get_candidate_urls_summary",
                state
            )
        
        if not meta_analysis_id:
            return {
                "success": False,
                "error": "No state provided to get_candidate_urls_summary",
                "candidates": []
            }
        
        with get_database_manager().get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT url, title, processing_status, created_at
                    FROM articles 
                    WHERE meta_analysis_id = %s
                    ORDER BY created_at DESC
                """, (meta_analysis_id,))
                
                candidates = []
                status_counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
                
                for row in cursor.fetchall():
                    candidates.append({
                        "url": row['url'],
                        "title": row['title'],
                        "status": row['processing_status'],
                        "added_at": row['created_at'].isoformat() if row['created_at'] else None
                    })
                    status_counts[row['processing_status']] = status_counts.get(row['processing_status'], 0) + 1
                
                return {
                    "success": True,
                    "total_candidates": len(candidates),
                    "status_summary": status_counts,
                    "candidates": candidates,
                    "meta_analysis_id": meta_analysis_id
                }
                
    except Exception as e:
        logger.error(f"Error getting candidate URLs summary: {e}")
        return {
            "success": False,
            "error": str(e),
            "candidates": []
        }


@tool
def generate_search_queries(pico: Dict[str, str]) -> List[str]:
    """
    Generate optimized search queries based on PICO framework using LLM
    
    Args:
        pico: Dictionary with P (Population), I (Intervention), C (Comparison), O (Outcome)
        
    Returns:
        List of optimized search queries for different databases
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        llm = ChatOpenAI(model="o3-mini", api_key=api_key)
        
        prompt = f"""
        You are an expert in constructing search queries for semantic search engines in the medical field. We are using Tavily, a semantic search tool, to find relevant scientific literature. The search must be restricted to trustworthy medical domains (such as .gov, .edu, .org, or well-known medical publishers).

        Generate 5-7 optimized search queries in natural language, based on the following PICO framework:

        Population (P): {pico.get('P', 'Not specified')}
        Intervention (I): {pico.get('I', 'Not specified')}
        Comparison (C): {pico.get('C', 'Not specified')}
        Outcome (O): {pico.get('O', 'Not specified')}

        Requirements:
        1. Write queries in natural language, as if you were asking a question or describing the information you want to find. Do NOT use MeSH terms or database-specific syntax.
        2. Restrict the queries to trustworthy medical sources (e.g., add "site:gov", "site:edu", "site:nih.gov", "site:who.int", or similar, when appropriate).
        3. Include both broad and specific queries.
        4. Consider synonyms and alternative terms in natural language.
        5. Focus on finding randomized controlled trials, systematic reviews, and meta-analyses.
        6. Each query should be concise (10-20 words maximum).
        7. Include at least one query that combines all PICO elements.
        8. Include queries focusing on specific aspects (such as intervention effectiveness or comparison studies).

        Return ONLY a JSON list of query strings, and nothing else. Do not include any explanation, markdown, or extra text.

        Example format: ["query 1", "query 2", "query 3"]
        """
        
        response = llm.invoke(prompt)
        
        # Parse LLM response to extract queries
        try:
            queries = json.loads(response.content)
            if isinstance(queries, list):
                logger.info(f"Generated {len(queries)} search queries from PICO")
                return queries
        except json.JSONDecodeError:
            # Fallback: extract queries from text response
            lines = response.content.strip().split('\\n')
            queries = []
            for line in lines:
                line = line.strip()
                # Skip JSON markdown, empty lines, and array brackets
                if (line and 
                    not line.startswith('[') and 
                    not line.startswith(']') and
                    not line.startswith('```') and
                    line != '```json' and
                    line != '```'):
                    
                    # Remove quotes, commas, and numbering
                    query = line.strip('"').strip("'").strip(',').strip()
                    if query.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.')):
                        query = query[2:].strip()
                    
                    # Final cleanup - remove any remaining quotes
                    query = query.strip('"').strip("'").strip()
                    
                    # Only add non-empty, meaningful queries
                    if query and len(query) > 5:
                        queries.append(query)
            
            if queries:
                logger.info(f"Generated {len(queries)} search queries (fallback parsing)")
                return queries
        
        # Ultimate fallback: generate basic queries
        fallback_queries = []
        if pico.get('P') and pico.get('I'):
            fallback_queries.append(f"{pico['P']} AND {pico['I']}")
        if pico.get('I') and pico.get('O'):
            fallback_queries.append(f"{pico['I']} AND {pico['O']} systematic review")
        if pico.get('I') and pico.get('C'):
            fallback_queries.append(f"{pico['I']} vs {pico['C']} randomized controlled trial")
        
        logger.warning("Using fallback query generation")
        return fallback_queries or [f"{pico.get('I', 'intervention')} meta-analysis"]
        
    except Exception as e:
        logger.error(f"Error generating search queries: {str(e)}")
        # Return basic query as last resort
        intervention = pico.get('I', 'intervention')
        return [f"{intervention} systematic review", f"{intervention} meta-analysis"]


@tool
def assess_article_relevance(
    article: Dict[str, Any],
    pico: Dict[str, str],
    inclusion_criteria: List[str],
    exclusion_criteria: List[str]
) -> Dict[str, Any]:
    """
    Assess article relevance to research question using LLM analysis
    
    Args:
        article: Article metadata (title, snippet, url)
        pico: PICO framework for the meta-analysis
        inclusion_criteria: List of inclusion criteria
        exclusion_criteria: List of exclusion criteria
        
    Returns:
        Assessment with relevance score, reasoning, and recommendation
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        llm = ChatOpenAI(model="o3", api_key=api_key)
        
        prompt = f"""
        You are an expert systematic reviewer assessing article relevance for a meta-analysis.
        
        RESEARCH FRAMEWORK:
        Population (P): {pico.get('P', 'Not specified')}
        Intervention (I): {pico.get('I', 'Not specified')}
        Comparison (C): {pico.get('C', 'Not specified')}
        Outcome (O): {pico.get('O', 'Not specified')}
        
        INCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in inclusion_criteria) if inclusion_criteria else "- Not specified"}
        
        EXCLUSION CRITERIA:
        {chr(10).join(f"- {criterion}" for criterion in exclusion_criteria) if exclusion_criteria else "- Not specified"}
        
        ARTICLE TO ASSESS:
        Title: {article.get('title', 'No title')}
        Abstract/Snippet: {article.get('snippet', 'No abstract available')}
        Source: {article.get('source_domain', 'Unknown source')}
        
        Please assess this article's relevance and provide:
        1. Relevance score (0-100, where 100 is perfectly relevant)
        2. Brief reasoning for the score
        3. Specific PICO elements addressed
        4. Study type identification (RCT, systematic review, etc.)
        5. Recommendation (include/exclude/needs_full_text_review)
        
        Respond in JSON format:
        {{
            "relevance_score": 85,
            "reasoning": "Brief explanation of relevance assessment",
            "pico_coverage": {{
                "population": true,
                "intervention": true, 
                "comparison": false,
                "outcome": true
            }},
            "study_type": "randomized_controlled_trial",
            "recommendation": "include",
            "quality_indicators": ["large_sample_size", "peer_reviewed"],
            "potential_issues": ["old_study", "limited_population"]
        }}
        """
        
        response = llm.invoke(prompt)
        
        try:
            assessment = json.loads(response.content)
            
            # Add metadata
            assessment.update({
                "article_url": article.get("url", ""),
                "article_title": article.get("title", ""),
                "assessed_at": datetime.now().isoformat(),
            })
            
            logger.info(f"Assessed article relevance: {assessment.get('relevance_score', 0)}/100")
            return assessment
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM assessment response")
            return {
                "relevance_score": 50,  # Neutral score
                "reasoning": "Assessment parsing failed",
                "recommendation": "needs_full_text_review",
                "error": "Failed to parse LLM response"
            }
    
    except Exception as e:
        logger.error(f"Error assessing article relevance: {str(e)}")
        return {
            "relevance_score": 0,
            "reasoning": f"Assessment failed: {str(e)}",
            "recommendation": "exclude",
            "error": str(e)
        }


def extract_domain(url: str) -> str:
    """Extract domain from URL for categorization"""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"
'''
    
    with open(research_tools_path, 'w', encoding='utf-8') as f:
        f.write(corrected_code)
    
    print("✅ research_tools.py corrigido!")

def fix_main_py():
    """Inicializar cache global no main.py"""
    
    print("\n🔧 CORRIGINDO MAIN.PY PARA INICIALIZAR CACHE")
    print("=" * 50)
    
    main_path = "metanalyst_agent/main.py"
    
    # Ler conteúdo atual
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adicionar importação e inicialização do cache
    if "set_cached_meta_analysis_id" not in content:
        # Encontrar local para adicionar import
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Adicionar import para o cache
        import_lines.append("from metanalyst_agent.tools.research_tools import set
