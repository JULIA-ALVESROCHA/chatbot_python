from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

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
