from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def _load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF file and return LangChain Documents.
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        print(f"[LOADER] Loaded PDF: {pdf_path.name} ({len(documents)} pages)")
        return documents

    except Exception as e:
        print(f"[LOADER][ERROR] Failed to load PDF {pdf_path}: {e}")
        return []


def load_documents(path: str) -> List[Document]:
    """
    Load PDF documents from a file or directory.

    Args:
        path (str): Path to a PDF file or a directory containing PDFs.

    Returns:
        List[Document]: Loaded documents.
    """
    base_path = Path(path)

    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    all_documents: List[Document] = []

    # Case 1: single PDF file
    if base_path.is_file():
        if base_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {base_path.suffix}")

        all_documents.extend(_load_single_pdf(base_path))
        return all_documents

    # Case 2: directory with PDFs
    if base_path.is_dir():
        pdf_files = list(base_path.glob("*.pdf"))

        if not pdf_files:
            print(f"[LOADER] No PDF files found in directory: {base_path}")
            return []

        for pdf_file in pdf_files:
            docs = _load_single_pdf(pdf_file)
            all_documents.extend(docs)

        print(f"[LOADER] Total documents loaded: {len(all_documents)}")
        return all_documents

    # Fallback (should not happen)
    raise ValueError(f"Invalid path type: {path}")
