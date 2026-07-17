from typing import List, Optional, Any
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline.pipeline import process_query
from src.infra.cache import clear_history

logger = logging.getLogger("bgo_chatbot.api.chat")
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)  # limite prático
    language: str = "pt"
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None
    # ajuste os campos conforme o que seus docs/metadata realmente têm


class ChatResponse(BaseModel):
    answer: str
    sources: List[Any]  # pode trocar por List[SourceItem] se quiser validar estrutura


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Endpoint principal do chatbot.
    Recebe uma pergunta (req.question) -> executa o pipeline RAG -> devolve resposta e fontes.
    """
    try:
        # chama o pipeline (assíncrono)
        result = await process_query(
            req.question,
            language=req.language,
            session_id=req.session_id
        )

        # validação simples do resultado esperado
        if not isinstance(result, dict):
            logger.error("process_query retornou tipo inesperado: %s", type(result))
            raise HTTPException(status_code=500, detail="Erro interno: formato de resposta inválido.")

        answer = result.get("answer")
        sources = result.get("sources", [])

        if not answer:
            logger.error("process_query não retornou 'answer': %s", result)
            raise HTTPException(status_code=500, detail="Erro interno: resposta vazia do pipeline.")

        # garantir que sources é lista
        if not isinstance(sources, list):
            logger.warning("sources não é lista, convertendo para lista.")
            sources = [sources]

        return {"answer": answer, "sources": sources}

    except HTTPException:
        # re-raise, já está no formato certo
        raise
    except Exception as e:
        logger.exception("Erro no endpoint /chat: %s", e)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a requisição.")


@router.delete("/chat/history/{session_id}", tags=["chat"])
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a specific session.
    
    Args:
        session_id: Unique identifier for the conversation session
    """
    try:
        clear_history(session_id)
        return {"message": f"Chat history cleared for session {session_id}", "session_id": session_id}
    except Exception as e:
        logger.exception("Erro ao limpar histórico: %s", e)
        raise HTTPException(status_code=500, detail="Erro interno ao limpar histórico.")
