from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader  # ← changed
from langchain_core.documents import Document


def _load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF file and return LangChain Documents with page metadata.
    """
    try:
        loader = PyMuPDFLoader(str(pdf_path))  # ← changed
        documents = loader.load()

        # Normalize metadata for every page-document
        for doc in documents:
            # PyMuPDF returns 0-indexed pages — convert to 1-indexed for humans
            raw_page = doc.metadata.get("page", 0)
            doc.metadata["page"] = int(raw_page) + 1

            # Normalize source to just the filename stem (no path, no extension)
            # e.g. "/data/regulamento_11obg_2026.pdf" → "regulamento_11obg_2026"
            doc.metadata["source"] = pdf_path.stem

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
        List[Document]: Loaded documents with page and source metadata populated.
    """
    base_path = Path(path)
    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    all_documents: List[Document] = []

    if base_path.is_file():
        if base_path.suffix.lower() != ".pdf":
            raise ValueError(f"Unsupported file type: {base_path.suffix}")
        all_documents.extend(_load_single_pdf(base_path))
        return all_documents

    if base_path.is_dir():
        pdf_files = sorted(base_path.glob("*.pdf"))  # sorted for reproducibility
        if not pdf_files:
            print(f"[LOADER] No PDF files found in directory: {base_path}")
            return []
        for pdf_file in pdf_files:
            docs = _load_single_pdf(pdf_file)
            all_documents.extend(docs)
        print(f"[LOADER] Total documents loaded: {len(all_documents)}")
        return all_documents

    if __name__ == "__main__":
        docs = load_documents("path/to/your/pdfs")
        for doc in docs[:5]:
            print(doc.metadata)
        # Expected output:
        # {'source': 'regulamento_11obg_2026', 'page': 1, ...}
        # {'source': 'regulamento_11obg_2026', 'page': 2, ...}

    raise ValueError(f"Invalid path type: {path}")