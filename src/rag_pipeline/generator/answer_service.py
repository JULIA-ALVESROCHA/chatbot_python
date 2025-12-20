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

logger = logging.getLogger("bgo_chatbot.generator")


class AnswerService:
    """
    Service responsible for generating the final answer
    based on reranked documents and a user question.

    This class contains NO prompt text definitions.
    All prompt content is delegated to templates.py.
    """

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,  # 0.3 for balanced creativity/consistency
            request_timeout=30,  # 30 second timeout
            max_tokens=300,  # Limit response length to force conciseness
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
        language: str
    ) -> str:
        
        system_prompt = SYSTEM_PROMPT.format(language=language)

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
                )
            )

            answer_text = response.content
            
            if not answer_text or not answer_text.strip():
                logger.warning("LLM returned empty response, using fallback")
                answer_text = FALLBACK_RESPONSE

            # Extract sources from documents
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
    def _build_context_with_labels(documents: List[Document]) -> str:
        """
        Builds the textual context with source labels for citation.
        Each chunk is labeled with its source for easier reference.
        """
        context_parts = []
        
        for idx, doc in enumerate(documents, 1):
            metadata = doc.metadata or {}
            source_name = metadata.get("source", "Regulamento")
            page = metadata.get("page", "?")
            
            # Clean source name (remove extension if present)
            source_clean = source_name.replace(".pdf", "").replace(".txt", "")
            
            # Format: [Source: regulamento-pag8]
            label = f"[Fonte {idx}: {source_clean}-pag{page}]"
            context_parts.append(f"{label}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)

    @staticmethod
    def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extracts source metadata from documents for citation.
        
        Returns a list of dicts with title, page, url, etc.
        """
        sources = []
        seen_sources = set()  # Deduplicate sources
        
        for doc in documents:
            metadata = doc.metadata or {}
            
            # Create a unique key for this source
            source_key = (
                metadata.get("source", ""),
                metadata.get("page"),
            )
            
            # Skip duplicates
            if source_key in seen_sources:
                continue
            
            seen_sources.add(source_key)
            
            source_name = metadata.get("source", "Regulamento")
            
            # Clean source name: remove path, extension, and extra spaces
            source_clean = source_name.replace(".pdf", "").replace(".txt", "")
            # Remove common path prefixes
            source_clean = re.sub(r'^(?:data[/\\]raw[/\\]|data[/\\])', '', source_clean)
            source_clean = source_clean.strip()
            
            page = metadata.get("page")
            
            sources.append({
                "title": source_clean,
                "page": page,
                "url": metadata.get("url"),
                "chunk_id": metadata.get("chunk_id"),
                # Formatted citation for display
                "citation": f"{source_clean}- pag {page}" if page is not None else source_clean
            })
        
        return sources

    @staticmethod
    def _ensure_citations(answer_text: str, sources: List[Dict[str, Any]]) -> str:
        """
        Ensures the answer has properly formatted citations with correct spacing.
        """
        # 1. Strip the "Resposta:" prefix if the LLM included it (common with your template)
        clean_answer = re.sub(r"^Resposta:\s*", "", answer_text, flags=re.IGNORECASE)

        # 2. Remove any citations/trailers the LLM might have added
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
        
        # 3. CRITICAL: Clean up whitespace at the end of the LLM's text
        clean_answer = clean_answer.strip()

        if not sources:
            return clean_answer

        citation_lines = [
            f"- {src['citation']}"
            for src in sources
            if src.get("citation")
        ]

        if not citation_lines:
            return clean_answer

        # 4. Use double \n\n to ensure a clear vertical break
        return (
            clean_answer
            + "\n\nVocê pode encontrar mais em:\n"
            + "\n".join(citation_lines)
        )

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
) -> Dict[str, Any]:
    """
    Module-level function wrapper for AnswerService.generate_answer().
    
    This function maintains backward compatibility with pipeline.py
    which imports generate_answer as a standalone function.
    
    Args:
        question: The user's question
        documents: List of Document objects to use as context
        
    Returns:
        Dict with keys:
            - answer: str - The generated answer text WITH citations
            - sources: List[Dict] - Source metadata for reference
    """
    service = _get_service()
    return await service.generate_answer(question, documents)