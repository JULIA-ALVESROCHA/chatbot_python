# src/rag_pipeline/pipeline.py
from typing import Any, Dict, List, Optional
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from src.rag_pipeline.rewrite.rewrite_service import rewrite_query
from src.rag_pipeline.retrieval.vectorstore import get_retriever
from src.rag_pipeline.generator.answer_service import generate_answer
from src.app.core.config import settings

logger = logging.getLogger("bgo_chatbot.pipeline")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
async def process_query(
    question: str,
    language: str = "pt",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orquestra o pipeline RAG:
      1) rewrite -> torna a pergunta autossuficiente
      2) retrieve -> recupera documentos do vectorstore (FAISS)
      3) (opcional) rerank -> reordena top-N
      4) generate -> gera resposta final a partir dos docs

    Retorna um dicionário { "answer": str, "sources": list }.
    Tenacity retry decorates a função inteira — se uma exceção ocorrer,
    a chamada será re-tentada até o limite configurado.
    """
    logger.info("Processing query (session=%s): %s", session_id, question)

    # 1) REWRITE
    try:
        rewritten = await rewrite_query(question, chat_history="")
        logger.debug("Rewritten query: %s", rewritten)
    except Exception as e:
        logger.exception("Error during rewrite_query: %s", e)
        # lançar para que tenacity re-tente
        raise

    # 2) RETRIEVE
    try:
        # usa max_retrieve configurado no settings
        k = getattr(settings, "max_retrieve", 6)
        retriever = get_retriever(k=k)
        # get_relevant_documents é o método padrão de Retriever do LangChain
        docs = retriever.get_relevant_documents(rewritten)
        logger.info("Retrieved %d docs for query", len(docs))
    except RuntimeError as e:
        # Vectorstore não inicializado -> retornar mensagem amigável (não re-try)
        logger.error("Vectorstore not initialized: %s", e)
        return {
            "answer": "O sistema de busca ainda não está pronto. Por favor, tente novamente mais tarde.",
            "sources": []
        }
    except Exception as e:
        logger.exception("Error during retrieval: %s", e)
        # lançar para que tenacity re-tente
        raise

    if not docs:
        logger.info("No documents retrieved for query.")
        return {"answer": "Não encontrei nada no regulamento relacionado à sua pergunta. Pode reformular?", "sources": []}

    # 3) OPTIONAL RERANK - placeholder (implemente se tiver reranker)
    # try:
    #     if settings.use_reranker:
    #         docs = rerank(docs, rewritten)
    # except Exception as e:
    #     logger.exception("Reranker failed: %s", e)
    #     # continue with un-reranked docs

    # 4) GENERATE
    try:
        result = await generate_answer(rewritten, docs)
    except Exception as e:
        logger.exception("Error during generate_answer: %s", e)
        # lançar para que tenacity re-tente (ou mude aqui para fallback)
        raise

    # Normalize result to expected shape
    if isinstance(result, dict):
        answer = result.get("answer") or result.get("text") or str(result)
        sources = result.get("sources") or result.get("references") or []
    else:
        # if generate_answer returned a plain string
        answer = str(result)
        sources = []

    # Ensure types
    if not isinstance(sources, list):
        logger.warning("Sources is not a list; converting to list.")
        sources = [sources]

    logger.info("Returning answer (len=%d chars) and %d sources", len(answer or ""), len(sources))
    return {"answer": answer, "sources": sources}
