from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Split documents into smaller chunks suitable for embeddings.

    Args:
        documents (List[Document]): Original loaded documents.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: Chunked documents.
    """
    if not documents:
        print("[SPLITTER] No documents received. Returning empty list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"[SPLITTER] Split {len(documents)} documents into "
        f"{len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks
