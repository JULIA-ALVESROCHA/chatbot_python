# src/app/api/deps.py
from fastapi import Depends, HTTPException

from src.rag_pipeline.retrieval.vectorstore import get_retriever


def retriever_dep():
    """
    FastAPI dependency that returns a ready-to-use retriever.
    Ensures the FAISS vectorstore was initialized at startup.
    """
    try:
        return get_retriever()
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail="Sistema de busca ainda não está inicializado"
        ) from e
