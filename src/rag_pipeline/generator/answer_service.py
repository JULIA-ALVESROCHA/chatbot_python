from typing import List, Dict, Any, Optional
import logging
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


from .templates import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    FALLBACK_RESPONSE,
)
from src.app.core.config import settings

logger = logging.getLogger("bgo_chatbot.generator")


class AnswerService:
    """
    Service responsible for generating the final answer
    based on reranked documents and a user question.

    This class contains NO prompt text definitions.
    All prompt content is delegated to templates.py.
    """

    def __init__(self, model_name: str = None, temperature: float = None):
        self.llm = ChatOpenAI(
            model=model_name or getattr(settings, "generation_model", "gpt-4o-mini"),
            temperature=(
                temperature
                if temperature is not None
                else getattr(settings, "generation_temperature", 0.1)
            ),
            request_timeout=30,
            max_tokens=300,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", ANSWER_TEMPLATE),
            ]
        )

    async def generate_answer(
        self,
        question: str,
        documents: List[Document],
        language: str = "pt-BR",
        chat_history: str = ""
    ) -> Dict[str, Any]:
        """
        Generates a grounded answer using only the provided documents.
        
        Returns:
            Dict with keys:
                - answer: str - The generated answer text WITH citations
                - sources: List[Dict] - Source metadata for reference
        """

        if not documents:
            logger.warning("No documents provided for answer generation")
            return {
                "answer": FALLBACK_RESPONSE,
                "sources": []
            }

        try:
            # Build context with source labels for the LLM
            context = self._build_context_with_labels(documents)
            
            logger.debug("Generating answer with %d documents", len(documents))
            
            response = await self.llm.ainvoke(
                self.prompt.format_messages(
                    question=question,
                    context=context,
                    language=language,
                    chat_history=chat_history or "(sem histórico)",
                )
            )

            answer_text = response.content
            
            if not answer_text or not answer_text.strip():
                logger.warning("LLM returned empty response, using fallback")
                answer_text = FALLBACK_RESPONSE

            # Abstenção/pedido de reformulação não deve exibir fontes:
            # citações sob "não sei" passam falsa autoridade ao usuário.
            if self._is_abstention(answer_text):
                sources = []
            else:
                sources = self._extract_sources(documents)

            # Ensure answer has citations in the correct format
            answer_with_citations = self._ensure_citations(answer_text, sources)
            
            logger.info("Successfully generated answer (len=%d chars) with %d sources", 
                       len(answer_with_citations), len(sources))
            
            return {
                "answer": answer_with_citations,
                "sources": sources
            }

        except Exception as e:
            logger.exception("Error during answer generation: %s", e)
            # Re-raise to allow pipeline retry logic to handle it
            raise

    @staticmethod
    def _is_abstention(text: str) -> bool:
        """Detecta respostas de abstenção/reformulação (sem conteúdo factual)."""
        t = (text or "").lower()
        patterns = [
            "não encontrei", "nao encontrei",
            "não está no regulamento", "nao esta no regulamento",
            "não consta", "nao consta",
            "não está clara", "nao esta clara",
            "reformul",                      # reformule / reformular
            "pergunta está incompleta", "pergunta esta incompleta",
            "não fornece informações suficientes",
            "nao fornece informacoes suficientes",
            "não é mencionad", "nao e mencionad",
            "not mentioned", "could not find", "i did not find",
            "please rephrase",
        ]
        return any(p in t for p in patterns)

    @staticmethod
    def _detect_phase(text: str) -> str:
        """Heurística: a qual fase da OBG o trecho se refere.
        Usada para rotular o contexto e impedir que o gerador aplique
        regra de uma fase à outra (ex.: consulta permitida online != presencial)."""
        t = (text or "").lower()
        has_presencial = "presencial" in t
        has_online = bool(re.search(r"\bonline\b|fases?\s+on-?line", t))
        if has_presencial and has_online:
            return "menciona fase online E fase presencial — atenção ao trecho exato"
        if has_presencial:
            return "fase presencial"
        if has_online:
            return "fases online"
        return "fase não especificada no trecho"

    @staticmethod
    def _build_context_with_labels(documents: List[Document]) -> str:
        """
        Builds the textual context with source labels for citation.
        Each chunk is labeled with its source AND the phase it refers to.
        """
        context_parts = []

        for idx, doc in enumerate(documents, 1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source", "Regulamento")
            page = metadata.get("page", "?")

            source_clean = source_name.replace(".pdf", "").replace(".txt", "")

            phase = AnswerService._detect_phase(doc.page_content)
            label = f"[Fonte {idx}: {source_clean}-pag{page} | aplica-se a: {phase}]"
            context_parts.append(f"{label}\n{doc.page_content}")

        return "\n\n".join(context_parts)

    @staticmethod
    def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
        sources = []
        seen_sources = set()

        for doc in documents:
            metadata = doc.metadata or {}

            source_key = (
                metadata.get("source", ""),
                metadata.get("page"),
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            source_name = metadata.get("source", "Regulamento")
            source_clean = source_name.replace(".pdf", "").replace(".txt", "")
            source_clean = re.sub(r'^(?:data[/\\]raw[/\\]|data[/\\])', '', source_clean)
            source_clean = source_clean.strip()

            page = metadata.get("page")

           
            item_match = re.search(r"\b\d+(?:\.\d+){1,3}\b", doc.page_content)
            item = item_match.group(0) if item_match else None

            citation_parts = [source_clean]

            if item:
                citation_parts.append(f"item {item}")

            if page is not None:
                citation_parts.append(f"pag {page}")

            citation = " — ".join(citation_parts)

            sources.append({
                "title": source_clean,
                "page": page,
                "item": item,  
                "url": metadata.get("url"),
                "chunk_id": metadata.get("chunk_id"),
                "citation": citation
            })

        return sources[:3]

    @staticmethod
    def _ensure_citations(answer_text: str, sources: List[Dict[str, Any]]) -> str:
        """
        Ensures the answer has properly formatted citations with correct spacing.
        """
        
        clean_answer = re.sub(r"^Resposta:\s*", "", answer_text, flags=re.IGNORECASE)

      
        citation_patterns = [
            r"você pode encontrar.*$",
            r"encontre mais em.*$",
            r"consulte:.*$",
            r"fonte:.*$",
            r"fontes:.*$",
            r"referência:.*$",
        ]
        
        for pattern in citation_patterns:
            # We use DOTALL or just ensure we catch the end of the string
            clean_answer = re.sub(pattern, "", clean_answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
       
        clean_answer = clean_answer.strip()

        # As fontes vão SOMENTE no campo estruturado "sources" da resposta;
        # o frontend as renderiza como badges. Não anexar texto de citação aqui.
        return clean_answer


# Module-level singleton instance for backward compatibility
_service_instance: Optional[AnswerService] = None


def _get_service() -> AnswerService:
    """Get or create the singleton AnswerService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AnswerService()
    return _service_instance


async def generate_answer(
    question: str,
    documents: List[Document],
    language: str = "pt-BR"
) -> Dict[str, Any]:
    """
    Module-level function wrapper for AnswerService.generate_answer().
    
    This function maintains backward compatibility with pipeline.py
    which imports generate_answer as a standalone function.
    
    Args:
        question: The user's question
        documents: List of Document objects to use as context
        language: Language for the response (default: pt-BR)
        
    Returns:
        Dict with keys:
            - answer: str - The generated answer text WITH citations
            - sources: List[Dict] - Source metadata for reference
    """
    service = _get_service()
    return await service.generate_answer(question, documents, language)