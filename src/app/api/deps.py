from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

# ajuste os imports conforme seu projeto
from src.rag_pipeline.vectorstore import get_vectorstore_instance
from src.core.cache import cache
from src.core.settings import settings


# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------

def get_settings():
    """
    Dependency para acessar settings globais.
    """
    return settings


# ------------------------------------------------------------------------------
# REQUEST / SESSION
# ------------------------------------------------------------------------------

def get_request_id(
    x_request_id: Optional[str] = Header(default=None),
) -> str:
    """
    Gera ou reaproveita um request-id.
    """
    return x_request_id or str(uuid.uuid4())


def get_session_id(request: Request) -> Optional[str]:
    """
    Extrai session_id do body se existir.
    """
    try:
        body = request.state.body
        return getattr(body, "session_id", None)
    except Exception:
        return None


# ------------------------------------------------------------------------------
# CACHE
# ------------------------------------------------------------------------------

def get_cache():
    """
    Dependency de cache (in-memory ou Redis).
    """
    if cache is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache not available",
        )
    return cache


# ------------------------------------------------------------------------------
# VECTORSTORE (FAISS)
# ------------------------------------------------------------------------------

def get_vectorstore():
    """
    Retorna o vectorstore já inicializado.
    """
    try:
        vectorstore = get_vectorstore_instance()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vectorstore not initialized",
        ) from exc

    if vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vectorstore not available",
        )

    return vectorstore


# ------------------------------------------------------------------------------
# OPENAI KEY
# ------------------------------------------------------------------------------

def ensure_openai_key():
    """
    Garante que a OPENAI_API_KEY existe.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY not configured",
        )
    return True


# ------------------------------------------------------------------------------
# COMPOSED DEPENDENCY (RAG)
# ------------------------------------------------------------------------------

def rag_dependencies(
    _openai=Depends(ensure_openai_key),
    vectorstore=Depends(get_vectorstore),
    cache=Depends(get_cache),
):
    """
    Dependency composta para endpoints RAG.
    """
    return {
        "vectorstore": vectorstore,
        "cache": cache,
    }
