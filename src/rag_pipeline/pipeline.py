from typing import Any, Dict, List, Optional
import logging
import re
import os

from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.language import detect_language
from src.rag_pipeline.reranker.reranker import rerank_documents
from src.rag_pipeline.rewrite.rewrite_service import rewrite_query
from src.rag_pipeline.retrieval.vectorstore import get_retriever
from src.rag_pipeline.generator.answer_service import AnswerService
from src.app.core.config import settings
from src.infra.cache import get_history, add_to_history

logger = logging.getLogger("bgo_chatbot.pipeline")
answer_service = AnswerService()

# Perguntas sobre o próprio assistente (meta) não devem passar pelo RAG:
# o retrieval devolve trechos sobre como os ALUNOS embasam respostas,
# e o bot responde isso em vez de falar de si mesmo.
_META_PATTERNS = re.compile(
    r"(baseia (as )?suas respostas|em que (voc[eê]|vc) se baseia"
    r"|como (voc[eê]|vc) (funciona|responde|foi (feito|criado|treinado))"
    r"|quem (te|lhe) (criou|fez|desenvolveu|treinou)"
    r"|(voc[eê]|vc) (como|é um|eh um) (sistema|rob[oô]|ia|assistente|bot)"
    r"|o que (voc[eê]|vc) (é|eh|sabe fazer)"
    r"|what (do you base|are you based)|how (do you|were you) (work|made|trained)"
    r"|who (made|created|trained) you)",
    re.IGNORECASE,
)

_META_ANSWER_PT = (
    "Eu sou o GeoLUME, o assistente virtual da Olimpíada Brasileira de "
    "Geografia. Baseio minhas respostas exclusivamente nos documentos "
    "oficiais da OBG — regulamento, editais e materiais publicados pela "
    "organização. Se algo não estiver nesses documentos, eu digo que não "
    "encontrei, em vez de inventar."
)

_META_ANSWER_EN = (
    "I am GeoLUME, the virtual assistant of the Brazilian Geography "
    "Olympiad. My answers are based exclusively on the official OBG "
    "documents — the regulations, official notices and published "
    "materials. If something is not in those documents, I say I could "
    "not find it rather than guessing."
)

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

    # --- 1.5 META-QUESTIONS (sobre o assistente) — responder direto, sem RAG ---
    if _META_PATTERNS.search(question):
        meta_answer = _META_ANSWER_EN if language == "en" else _META_ANSWER_PT
        logger.info("Meta-question detected; answering directly without retrieval")
        if session_id:
            add_to_history(session_id, question, meta_answer)
        return {"answer": meta_answer, "sources": []}

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
    # IMPORTANTE: usamos a pergunta REESCRITA (autossuficiente) na geração.
    # Antes, o gerador recebia a pergunta crua ("e na fase presencial?") sem
    # histórico e respondia "pergunta incompleta" mesmo com retrieval correto.
    result = await answer_service.generate_answer(
        question=rewritten,
        documents=docs,
        language=language,
        chat_history=chat_history,
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