"""
Ferramentas para processamento de artigos científicos.
Implementa extração de dados estruturados e vetorização usando OpenAI.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
import hashlib
import uuid

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import Config
from src.models.schemas import (
    ExtractedStudy, StudyCharacteristics, OutcomeData, 
    QualityAssessment, VectorChunk, StudyType, OutcomeType,
    Citation
)

logger = logging.getLogger(__name__)


class ProcessingTools:
    """Ferramentas para processamento de artigos científicos."""
    
    def __init__(self):
        """Inicializa ferramentas de processamento."""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,
            api_key=Config.OPENAI_API_KEY
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.processing_config = Config.get_processing_config()
    
    def extract_structured_data(
        self, 
        content: Dict[str, Any], 
        pico: Dict[str, str]
    ) -> Optional[ExtractedStudy]:
        """
        Extrai dados estruturados de um artigo científico.
        
        Args:
            content: Conteúdo extraído do artigo
            pico: Estrutura PICO para contexto
            
        Returns:
            Dados estruturados extraídos ou None
        """
        try:
            logger.info(f"Extraindo dados estruturados de: {content['url']}")
            
            # Preparar prompts para extração
            extraction_prompt = self._create_extraction_prompt()
            
            # Preparar contexto
            article_text = content.get("content", "")
            sections = content.get("sections", {})
            
            context = {
                "title": content.get("title", ""),
                "content": article_text[:8000],  # Limitar para não exceder tokens
                "sections": sections,
                "pico": pico
            }
            
            # Extrair características do estudo
            characteristics = self._extract_study_characteristics(context)
            if not characteristics:
                logger.warning("Falha ao extrair características do estudo")
                return None
            
            # Extrair dados de desfechos
            outcomes = self._extract_outcome_data(context, characteristics)
            
            # Avaliar qualidade do estudo
            quality_assessment = self._assess_study_quality(context, characteristics)
            
            # Extrair findings e limitações
            key_findings = self._extract_key_findings(context)
            limitations = self._extract_limitations(context)
            
            # Calcular score de confiança
            confidence_score = self._calculate_confidence_score(
                characteristics, outcomes, quality_assessment
            )
            
            # Criar objeto ExtractedStudy
            extracted_study = ExtractedStudy(
                url=content["url"],
                characteristics=characteristics,
                outcomes=outcomes,
                quality_assessment=quality_assessment,
                full_text_available=bool(sections),
                abstract=sections.get("abstract", article_text[:500]),
                key_findings=key_findings,
                limitations=limitations,
                confidence_score=confidence_score
            )
            
            logger.info(f"Dados extraídos com sucesso. Confiança: {confidence_score:.2f}")
            return extracted_study
            
        except Exception as e:
            logger.error(f"Erro na extração de dados: {e}")
            return None
    
    def create_vector_chunks(
        self, 
        extracted_study: ExtractedStudy,
        content: Dict[str, Any]
    ) -> List[VectorChunk]:
        """
        Cria chunks vetorizados de um estudo.
        
        Args:
            extracted_study: Dados estruturados do estudo
            content: Conteúdo original
            
        Returns:
            Lista de chunks vetorizados
        """
        try:
            logger.info(f"Criando chunks para: {extracted_study.url}")
            
            # Preparar texto para chunking
            full_text = content.get("content", "")
            sections = content.get("sections", {})
            
            # Criar chunks por seção se disponível
            chunks = []
            if sections:
                for section_name, section_text in sections.items():
                    section_chunks = self._create_section_chunks(
                        section_text, section_name, extracted_study
                    )
                    chunks.extend(section_chunks)
            else:
                # Criar chunks do texto completo
                text_chunks = self.text_splitter.split_text(full_text)
                for i, chunk_text in enumerate(text_chunks):
                    chunk = self._create_vector_chunk(
                        chunk_text, "full_text", i, extracted_study
                    )
                    chunks.append(chunk)
            
            # Gerar embeddings em batch
            chunks_with_embeddings = self._generate_embeddings_batch(chunks)
            
            logger.info(f"Criados {len(chunks_with_embeddings)} chunks vetorizados")
            return chunks_with_embeddings
            
        except Exception as e:
            logger.error(f"Erro na criação de chunks: {e}")
            return []
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """Cria prompt para extração de dados estruturados."""
        template = """
        Você é um especialista em revisão sistemática e meta-análise. 
        Analise o artigo científico fornecido e extraia os dados estruturados solicitados.
        
        CONTEXTO PICO:
        População: {pico[population]}
        Intervenção: {pico[intervention]}
        Comparação: {pico[comparison]}
        Desfecho: {pico[outcome]}
        
        ARTIGO:
        Título: {title}
        Conteúdo: {content}
        
        INSTRUÇÕES:
        1. Extraia apenas informações explicitamente mencionadas no texto
        2. Se alguma informação não estiver disponível, indique como "não informado"
        3. Para dados numéricos, extraia valores exatos quando possível
        4. Identifique o tipo de estudo baseado na metodologia descrita
        5. Avalie a qualidade metodológica do estudo
        
        Responda em formato JSON estruturado.
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _extract_study_characteristics(
        self, 
        context: Dict[str, Any]
    ) -> Optional[StudyCharacteristics]:
        """Extrai características básicas do estudo."""
        try:
            prompt = f"""
            Extraia as características básicas deste estudo científico:
            
            Título: {context['title']}
            Conteúdo: {context['content'][:3000]}
            
            Responda em JSON com esta estrutura:
            {{
                "title": "título do estudo",
                "authors": ["autor1", "autor2"],
                "journal": "nome do journal",
                "year": 2023,
                "doi": "10.xxxx/xxxx",
                "pmid": "12345678",
                "study_type": "randomized_controlled_trial",
                "sample_size": 100,
                "population_description": "descrição da população",
                "intervention_group": {{"description": "...", "n": 50}},
                "control_group": {{"description": "...", "n": 50}},
                "primary_outcome": "desfecho primário",
                "secondary_outcomes": ["desfecho1", "desfecho2"],
                "follow_up_duration": "6 meses"
            }}
            
            Tipos de estudo válidos: {[e.value for e in StudyType]}
            """
            
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)
            
            # Validar e criar objeto
            return StudyCharacteristics(**data)
            
        except Exception as e:
            logger.error(f"Erro ao extrair características: {e}")
            return None
    
    def _extract_outcome_data(
        self, 
        context: Dict[str, Any],
        characteristics: StudyCharacteristics
    ) -> List[OutcomeData]:
        """Extrai dados de desfechos do estudo."""
        try:
            prompt = f"""
            Extraia os dados de desfechos deste estudo:
            
            Estudo: {characteristics.title}
            Conteúdo: {context['content'][:3000]}
            Desfecho primário: {characteristics.primary_outcome}
            
            Para cada desfecho, extraia os dados numéricos disponíveis.
            
            Responda em JSON com array de objetos:
            [
                {{
                    "outcome_name": "nome do desfecho",
                    "outcome_type": "binary|continuous|time_to_event|ordinal",
                    "intervention_events": 25,
                    "intervention_total": 100,
                    "control_events": 15,
                    "control_total": 100,
                    "effect_measure": "OR",
                    "effect_size": 1.67,
                    "confidence_interval_lower": 0.89,
                    "confidence_interval_upper": 3.12,
                    "p_value": 0.045,
                    "units": "mg/dL",
                    "notes": "observações adicionais"
                }}
            ]
            
            Tipos de desfecho válidos: {[e.value for e in OutcomeType]}
            """
            
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)
            
            outcomes = []
            for outcome_data in data:
                try:
                    outcome = OutcomeData(**outcome_data)
                    outcomes.append(outcome)
                except Exception as e:
                    logger.warning(f"Erro ao processar desfecho: {e}")
                    continue
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Erro ao extrair desfechos: {e}")
            return []
    
    def _assess_study_quality(
        self, 
        context: Dict[str, Any],
        characteristics: StudyCharacteristics
    ) -> QualityAssessment:
        """Avalia a qualidade metodológica do estudo."""
        try:
            prompt = f"""
            Avalie a qualidade metodológica deste estudo:
            
            Tipo: {characteristics.study_type}
            Título: {characteristics.title}
            Conteúdo: {context['content'][:3000]}
            
            Avalie cada domínio como: "low", "high", ou "unclear"
            
            Responda em JSON:
            {{
                "randomization": "low|high|unclear",
                "allocation_concealment": "low|high|unclear",
                "blinding_participants": "low|high|unclear",
                "blinding_outcome": "low|high|unclear",
                "incomplete_data": "low|high|unclear",
                "selective_reporting": "low|high|unclear",
                "other_bias": "low|high|unclear",
                "overall_quality": "low|moderate|high",
                "quality_score": 7.5,
                "notes": "observações sobre a qualidade"
            }}
            """
            
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)
            
            return QualityAssessment(
                study_id=str(uuid.uuid4()),
                **data
            )
            
        except Exception as e:
            logger.error(f"Erro na avaliação de qualidade: {e}")
            # Retornar avaliação padrão em caso de erro
            return QualityAssessment(
                study_id=str(uuid.uuid4()),
                overall_quality="unclear",
                quality_score=5.0,
                notes="Avaliação automática falhou"
            )
    
    def _extract_key_findings(self, context: Dict[str, Any]) -> List[str]:
        """Extrai principais achados do estudo."""
        try:
            prompt = f"""
            Extraia os principais achados deste estudo em português:
            
            Conteúdo: {context['content'][:2000]}
            
            Liste os 3-5 principais achados de forma concisa.
            Responda como array JSON de strings.
            """
            
            response = self.llm.invoke(prompt)
            findings = json.loads(response.content)
            
            return findings if isinstance(findings, list) else []
            
        except Exception as e:
            logger.error(f"Erro ao extrair achados: {e}")
            return []
    
    def _extract_limitations(self, context: Dict[str, Any]) -> List[str]:
        """Extrai limitações do estudo."""
        try:
            prompt = f"""
            Extraia as limitações mencionadas neste estudo:
            
            Conteúdo: {context['content'][-2000:]}  # Últimos 2000 chars (discussão)
            
            Liste as limitações identificadas pelos autores.
            Responda como array JSON de strings.
            """
            
            response = self.llm.invoke(prompt)
            limitations = json.loads(response.content)
            
            return limitations if isinstance(limitations, list) else []
            
        except Exception as e:
            logger.error(f"Erro ao extrair limitações: {e}")
            return []
    
    def _calculate_confidence_score(
        self,
        characteristics: StudyCharacteristics,
        outcomes: List[OutcomeData],
        quality: QualityAssessment
    ) -> float:
        """Calcula score de confiança na extração."""
        score = 0.0
        
        # Pontuação por completude dos dados
        if characteristics.sample_size > 0:
            score += 0.2
        if outcomes:
            score += 0.3
        if quality.quality_score > 5:
            score += 0.2
        if characteristics.study_type != StudyType.OTHER:
            score += 0.2
        if characteristics.doi:
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_section_chunks(
        self,
        section_text: str,
        section_name: str,
        study: ExtractedStudy
    ) -> List[VectorChunk]:
        """Cria chunks de uma seção específica."""
        chunks = []
        text_chunks = self.text_splitter.split_text(section_text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = self._create_vector_chunk(
                chunk_text, section_name, i, study
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_vector_chunk(
        self,
        text: str,
        section: str,
        index: int,
        study: ExtractedStudy
    ) -> VectorChunk:
        """Cria um chunk vetorizado."""
        chunk_id = hashlib.md5(
            f"{study.url}_{section}_{index}".encode()
        ).hexdigest()
        
        return VectorChunk(
            chunk_id=chunk_id,
            study_id=study.url,
            content=text,
            embedding=[],  # Será preenchido depois
            section=section,
            start_char=index * Config.CHUNK_SIZE,
            end_char=(index + 1) * Config.CHUNK_SIZE,
            study_title=study.characteristics.title,
            study_authors=study.characteristics.authors,
            study_year=study.characteristics.year
        )
    
    def _generate_embeddings_batch(
        self, 
        chunks: List[VectorChunk]
    ) -> List[VectorChunk]:
        """Gera embeddings em batch para os chunks."""
        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            return chunks


# Ferramentas para uso no LangGraph
@tool
def extract_study_data(
    content: Dict[str, Any], 
    pico: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Extrai dados estruturados de um artigo científico.
    
    Args:
        content: Conteúdo extraído do artigo
        pico: Estrutura PICO
        
    Returns:
        Dados estruturados ou None
    """
    processor = ProcessingTools()
    extracted = processor.extract_structured_data(content, pico)
    return extracted.dict() if extracted else None


@tool
def create_study_chunks(
    extracted_study: Dict[str, Any],
    content: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Cria chunks vetorizados de um estudo.
    
    Args:
        extracted_study: Dados estruturados do estudo
        content: Conteúdo original
        
    Returns:
        Lista de chunks vetorizados
    """
    processor = ProcessingTools()
    study = ExtractedStudy(**extracted_study)
    chunks = processor.create_vector_chunks(study, content)
    return [chunk.dict() for chunk in chunks]


@tool
def generate_citation(study_data: Dict[str, Any]) -> str:
    """
    Gera citação em formato Vancouver.
    
    Args:
        study_data: Dados do estudo
        
    Returns:
        Citação formatada
    """
    try:
        characteristics = study_data.get("characteristics", {})
        
        citation = Citation(
            study_id=study_data.get("url", ""),
            authors=characteristics.get("authors", []),
            title=characteristics.get("title", ""),
            journal=characteristics.get("journal", ""),
            year=characteristics.get("year", 2023),
            doi=characteristics.get("doi")
        )
        
        return citation.to_vancouver()
        
    except Exception as e:
        logger.error(f"Erro ao gerar citação: {e}")
        return "Citação não disponível"