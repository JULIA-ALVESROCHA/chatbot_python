from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .templates import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    FALLBACK_RESPONSE,
)


class AnswerService:
    """
    Service responsible for generating the final answer
    based on reranked documents and a user question.

    This class contains NO prompt text definitions.
    All prompt content is delegated to templates.py.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # deterministic, as expected for QA
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
    ) -> str:
        """
        Generates a grounded answer using only the provided documents.
        """

        if not documents:
            return FALLBACK_RESPONSE

        context = self._build_context(documents)

        response = await self.llm.ainvoke(
            self.prompt.format_messages(
                question=question,
                context=context,
            )
        )

        return response.content

    @staticmethod
    def _build_context(documents: List[Document]) -> str:
        """
        Builds the textual context passed to the LLM
        by concatenating reranked documents.
        """

        return "\n\n".join(
            f"- {doc.page_content}" for doc in documents
        )


# Module-level singleton instance for backward compatibility
# This allows pipeline.py to import generate_answer as a function
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
) -> str:
    """
    Module-level function wrapper for AnswerService.generate_answer().
    
    This function maintains backward compatibility with pipeline.py
    which imports generate_answer as a standalone function.
    
    Args:
        question: The user's question
        documents: List of Document objects to use as context
        
    Returns:
        Generated answer string
    """
    service = _get_service()
    return await service.generate_answer(question, documents)