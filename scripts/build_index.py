from pathlib import Path
from typing import List
import re
from collections import defaultdict

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.app.core.config import settings

DATA_RAW = Path("data/raw")
PROCESSED = Path("data/processed/faiss_index")
PROCESSED.mkdir(parents=True, exist_ok=True)

ITEM_REGEX = re.compile(r"\b\d+\.\d+\.\d+\b")


def load_documents() -> List[Document]:
    docs = []
    for p in DATA_RAW.iterdir():
        if p.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(p))  # fixed: was PyPDFLoader
            loaded = loader.load()
            # PyMuPDF gives 0-indexed pages — convert to 1-indexed
            for doc in loaded:
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            docs.extend(loaded)
        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf8")
            docs.extend(loader.load())
    print(f"[LOADER] Sample metadata from first doc: {docs[0].metadata}")
    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(documents)

    # assign deterministic chunk_id and extract item number
    page_chunk_counter = defaultdict(int)

    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)

        slug = (
            Path(source).stem
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .lower()
        )

        key = f"{slug}_p{page}"
        page_chunk_counter[key] += 1
        idx = page_chunk_counter[key]

        doc.metadata["chunk_id"] = f"{slug}_p{page}_c{idx}"

        # extract regulation item number (e.g. 4.2.1)
        match = ITEM_REGEX.search(doc.page_content or "")
        doc.metadata["item"] = match.group() if match else None

    return split_docs


def build_faiss(docs: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(PROCESSED))
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