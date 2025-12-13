import asyncio
from typing import List
from langchain_core.documents import Document
from .model_wrapper import CrossEncoderReranker

_reranker = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


async def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """
    Reranks documents using cross-encoder model.
    Wraps synchronous rerank call in async executor to avoid blocking event loop.
    """
    reranker = get_reranker()
    # Run blocking synchronous call in thread pool to avoid blocking event loop
    return await asyncio.to_thread(reranker.rerank, query, documents, top_k)
