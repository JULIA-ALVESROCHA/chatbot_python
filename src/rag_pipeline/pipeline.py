from typing import Any, Dict, List, Optional
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from src.rag_pipeline.reranker.reranker import rerank_documents
from src.rag_pipeline.rewrite.rewrite_service import rewrite_query
from src.rag_pipeline.retrieval.vectorstore import get_retriever
from src.rag_pipeline.generator.answer_service import generate_answer
from src.app.core.config import settings
from src.infra.cache import get_history, add_to_history

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

    # 0) Get chat history if session_id provided
    chat_history = ""
    if session_id:
        chat_history = get_history(session_id, max_turns=5)
        if chat_history:
            logger.debug("Retrieved chat history for session %s (%d chars)", session_id, len(chat_history))

    # 1) REWRITE
    try:
        rewritten = await rewrite_query(question, chat_history=chat_history)
        logger.debug("Rewritten query: %s", rewritten)
    except Exception as e:
        logger.exception("Error during rewrite_query: %s", e)
        raise

    # 2) RETRIEVE
    try:
        k = getattr(settings, "max_retrieve", 6)
        retriever = get_retriever(k=k)
        docs = await retriever.ainvoke(rewritten)
        
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
        raise

    if not docs:
        logger.info("No documents retrieved for query.")
        return {
            "answer": "Não encontrei nada no regulamento relacionado à sua pergunta. Pode reformular?",
            "sources": []
        }

    # 3) RERANK (opcional)
    if settings.use_reranker:
        try:
            rerank_k = settings.max_rerank
            docs = await rerank_documents(
                query=rewritten,
                documents=docs,
                top_k=rerank_k,
            )
            logger.info("Reranked to %d docs", len(docs))
        except Exception as e:
            logger.warning("Reranker failed, using original docs: %s", e)

    # 4) GENERATE (LLM FINAL)
    try:
        # generate_answer now returns Dict[str, Any] with "answer" and "sources"
        result = await generate_answer(rewritten, docs)
        
        # Validate result structure
        if not isinstance(result, dict):
            logger.error("generate_answer returned non-dict: %s", type(result))
            raise ValueError("Invalid response from generate_answer")
        
        if "answer" not in result:
            logger.error("generate_answer missing 'answer' key")
            raise ValueError("Missing 'answer' in response")
        
    except Exception as e:
        logger.exception("Error during generate_answer: %s", e)
        raise

    # Extract answer and sources
    answer = result["answer"]
    sources = result.get("sources", [])

    logger.info("Returning answer (len=%d chars) and %d sources", len(answer or ""), len(sources))
    
    # Save to chat history if session_id provided
    if session_id:
        add_to_history(session_id, question, answer)
        logger.debug("Saved to chat history for session %s", session_id)
    
    return {
        "answer": answer,
        "sources": sources
    }