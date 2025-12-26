from typing import Any, Dict, List, Optional
import logging
import re
import os

from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.language import detect_language
from src.rag_pipeline.reranker.reranker import rerank_documents
from src.rag_pipeline.rewrite.rewrite_service import rewrite_query
# ✅ IMPORTANTE: Usamos apenas o get_retriever do seu vectorstore.py
from src.rag_pipeline.retrieval.vectorstore import get_retriever
from src.rag_pipeline.generator.answer_service import AnswerService
from src.app.core.config import settings
from src.infra.cache import get_history, add_to_history

logger = logging.getLogger("bgo_chatbot.pipeline")
answer_service = AnswerService()

# 

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
async def process_query(
    question: str,
    language: str = None, # Fixed type hint default
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orquestra o pipeline RAG:
      1) rewrite -> torna a pergunta autossuficiente
      2) retrieve -> recupera documentos do vectorstore (FAISS)
      3) (opcional) rerank -> reordena top-N
      4) generate -> gera resposta final a partir dos docs

    Retorna um dicionário { "answer": str, "sources": list }.
    """
    
    # --- 1. PREPARATION ---
    if not language or language == "auto":
        language = detect_language(question)

    logger.info("Processing query (session=%s, language=%s): %s", session_id, language, question)

    # Get chat history if session_id provided
    chat_history = ""
    if session_id:
        chat_history = get_history(session_id, max_turns=5)
        if chat_history:
            logger.debug("Retrieved chat history for session %s (%d chars)", session_id, len(chat_history))

    # --- 2. REWRITE ---
    try:
        rewritten = await rewrite_query(question, chat_history=chat_history)
        logger.debug("Rewritten query: %s", rewritten)
    except Exception as e:
        logger.exception("Error during rewrite_query: %s", e)
        # Se falhar o rewrite, usamos a pergunta original
        rewritten = question

    # --- 3. RETRIEVE ---
    try:
        k = getattr(settings, "max_retrieve", 6)
        
        # ✅ FIX: Usamos get_retriever que já trata o vectorstore global
        retriever = get_retriever(k=k)
        
        # Invoke retriever asynchronously
        docs = await retriever.ainvoke(rewritten)
        
        logger.info("Retrieved %d docs for query", len(docs))
        
    except RuntimeError as e:
        # Vectorstore não inicializado -> retornar mensagem amigável (não re-try)
        logger.error("Vectorstore not initialized: %s", e)
        return {
            "answer": "O sistema de busca ainda não está pronto. Aguarde alguns segundos e tente novamente.",
            "sources": []
        }
    except Exception as e:
        logger.exception("Error during retrieval: %s", e)
        raise

    if not docs:
        logger.info("No documents retrieved for query.")
        return {
            "answer": "Não encontrei nada no regulamento relacionado à sua pergunta. Pode reformular?",
            "sources": []
        }

    # --- 4. RERANK (Opcional) ---
    if getattr(settings, "use_reranker", False): # Safer getattr
        try:
            rerank_k = getattr(settings, "max_rerank", 4)
            docs = await rerank_documents(
                query=rewritten,
                documents=docs,
                top_k=rerank_k,
            )
            logger.info("Reranked to %d docs", len(docs))
        except Exception as e:
            logger.warning("Reranker failed, using original docs: %s", e)

    # --- 5. GENERATE (LLM Final) ---
    # answer_service já retorna {"answer": str, "sources": list}
    result = await answer_service.generate_answer(
        question=question,
        documents=docs,
        language=language
    )

    answer = result["answer"]
    sources = result["sources"]

    logger.info(
        "Returning answer (len=%d chars) and %d sources",
        len(answer),
        len(sources)
    )

    # Save to chat history if session_id provided
    if session_id:
        add_to_history(session_id, question, answer)
        logger.debug("Saved to chat history for session %s", session_id)

    return {
        "answer": answer,
        "sources": sources
    }