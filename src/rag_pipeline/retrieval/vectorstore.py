from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# variável global que manterá o índice carregado
_vectorstore = None


def init_vectorstore(index_path: str):
    """
    Inicializa o vectorstore FAISS a partir do caminho salvo.
    Esse método deve ser chamado na startup do servidor FastAPI.
    """
    global _vectorstore

    index_dir = Path(index_path)

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index não encontrado em {index_path}. "
            f"Execute primeiro o script build_index.py."
        )

    # Cria embeddings (TEM que ser o mesmo modelo usado no build_index)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Carrega FAISS
    _vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True  # necessário para FAISS
    )

    return _vectorstore


def create_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Cria um novo vectorstore FAISS a partir de documentos e embeddings.
    
    Esta função é útil para testes e para criar índices do zero.
    Para produção, use init_vectorstore() para carregar um índice salvo.
    
    Args:
        documents: Lista de Document objects para indexar
        embeddings: Instância de Embeddings para gerar vetores
        
    Returns:
        FAISS: Vectorstore FAISS criado
        
    Example:
        >>> from src.rag_pipeline.retrieval.embeddings import get_embeddings
        >>> embeddings = get_embeddings()
        >>> vectorstore = create_vectorstore(chunks, embeddings)
    """
    if not documents:
        raise ValueError("Cannot create vectorstore from empty document list")
    
    if not embeddings:
        raise ValueError("Embeddings instance is required")
    
    print(f"[VECTORSTORE] Creating FAISS index from {len(documents)} documents")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("[VECTORSTORE] FAISS index created successfully")
    
    return vectorstore


def get_retriever(k: int = 6):
    """
    Retorna um retriever configurado com k documentos.
    """
    if _vectorstore is None:
        raise RuntimeError(
            "Vectorstore não inicializado. "
            "Chame init_vectorstore() na inicialização do servidor."
        )

    return _vectorstore.as_retriever(search_kwargs={"k": k})
