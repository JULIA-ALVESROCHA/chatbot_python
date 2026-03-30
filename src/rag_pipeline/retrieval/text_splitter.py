from collections import defaultdict
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.app.core.config import settings


def split_documents(
    documents: List[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[Document]:
    """
    Split documents into smaller chunks suitable for embeddings.

    Args:
        documents (List[Document]): Original loaded documents.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: Chunked documents with chunk_id and page metadata populated.
    """
    if not documents:
        print("[SPLITTER] No documents received. Returning empty list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    # --- chunk_id injection ---
    # Produces deterministic IDs like: regulamento_11obg_2026_p8_c3
    # Counter resets per (source, page) pair so IDs are stable across re-indexing
    page_chunk_counter = defaultdict(int)

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)

        # source is already a clean stem from loader.py (e.g. "regulamento_11obg_2026")
        # but sanitize just in case it comes with a path or extension
        slug = source.split("/")[-1].replace(".pdf", "").replace(" ", "_").lower()

        key = f"{slug}_p{page}"
        page_chunk_counter[key] += 1
        idx = page_chunk_counter[key]

        chunk.metadata["chunk_id"] = f"{slug}_p{page}_c{idx}"

    print(
        f"[SPLITTER] Split {len(documents)} documents into "
        f"{len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks