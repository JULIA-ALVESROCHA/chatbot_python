from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.app.core.config import settings
from src.rag_pipeline.retrieval.loader import load_documents as _load_documents
from src.rag_pipeline.retrieval.text_splitter import split_documents as _split_documents
from src.rag_pipeline.retrieval.manifest import write_manifest

DATA_RAW = Path("data/raw")
PROCESSED = Path("data/processed/faiss_index")
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_documents() -> List[Document]:
    # Delegates to the shared loader (PyMuPDF, page/source metadata
    # normalization) so build-time and query-time loading never diverge.
    docs = _load_documents(str(DATA_RAW))
    if docs:
        print(f"[LOADER] Sample metadata from first doc: {docs[0].metadata}")
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    # Item-aware splitting (src/rag_pipeline/retrieval/text_splitter.py):
    # keeps numbered regulation items whole and cleans PDF extraction noise
    # (ligatures, soft line breaks) before embedding. Also assigns
    # chunk_id/item/section/part metadata — no separate pass needed here.
    return _split_documents(documents)


def build_faiss(docs: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(PROCESSED))
    write_manifest(vectorstore, docs, str(PROCESSED)) 
    return vectorstore


def verify_index(vectorstore: FAISS):
    """Sanity check — confirms chunk_id and page are populated."""
    print("\n[VERIFY] Sampling 5 chunks from index:")
    results = vectorstore.similarity_search("Quem pode participar?", k=5)
    for r in results:
        print({
            "chunk_id": r.metadata.get("chunk_id"),
            "page":     r.metadata.get("page"),
            "item":     r.metadata.get("item"),
            "source":   r.metadata.get("source"),
        })
    
    # check for nulls
    import pickle
    with open(PROCESSED / "index.pkl", "rb") as f:
        docstore, _ = pickle.load(f)
    
    null_ids = [
        k for k, doc in docstore._dict.items()
        if not doc.metadata.get("chunk_id")
    ]
    null_pages = [
        k for k, doc in docstore._dict.items()
        if not doc.metadata.get("page")
    ]
    print(f"\n[VERIFY] Chunks with null chunk_id: {len(null_ids)}")
    print(f"[VERIFY] Chunks with null/zero page: {len(null_pages)}")
    if not null_ids and not null_pages:
        print("[VERIFY] ✔ Index is clean — ready for evaluation")


if __name__ == "__main__":
    print("1) Carregando documentos...")
    docs = load_documents()
    print(f"   Documentos carregados: {len(docs)}")

    print("2) Dividindo em chunks...")
    chunks = split_documents(docs)
    print(f"   Chunks criados: {len(chunks)}")
    print(f"   Sample chunk metadata: {chunks[0].metadata}")

    print("3) Criando índice FAISS...")
    vs = build_faiss(chunks)

    print("4) Verificando índice...")
    verify_index(vs)

    print("\n✔ INDEXAÇÃO FINALIZADA COM SUCESSO!")

