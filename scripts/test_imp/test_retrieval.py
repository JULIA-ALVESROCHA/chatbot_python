import sys
from pathlib import Path

# -------------------------------------------------
# Force chatbot_python as project root (igual aos outros testes)
# -------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print("[TEST] Project root:", PROJECT_ROOT)

# -------------------------------------------------
# Imports do retrieval
# -------------------------------------------------
from src.rag_pipeline.retrieval.loader import load_documents
from src.rag_pipeline.retrieval.text_splitter import split_documents
from src.rag_pipeline.retrieval.embeddings import get_embeddings
from src.rag_pipeline.retrieval.vectorstore import create_vectorstore

# -------------------------------------------------
# Config
# -------------------------------------------------
# Use existing PDFs from data/raw directory
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Regulamento_2025 .pdf"
TOP_K = 3

# -------------------------------------------------
# Test
# -------------------------------------------------
def test_retrieval_pipeline():
    print("\n[1] Loading documents...")
    if not PDF_PATH.exists():
        print(f"[ERROR] PDF file not found: {PDF_PATH}")
        print("[INFO] Trying to load from data/raw directory instead...")
        # Fallback: try loading from directory
        pdf_dir = PROJECT_ROOT / "data" / "raw"
        if pdf_dir.exists():
            docs = load_documents(str(pdf_dir))
        else:
            raise FileNotFoundError(f"Neither {PDF_PATH} nor {pdf_dir} exists")
    else:
        docs = load_documents(str(PDF_PATH))  # Convert Path to string
    
    print(f"Loaded {len(docs)} documents")

    assert len(docs) > 0, "Loader retornou zero documentos"

    print("\n[2] Splitting documents...")
    chunks = split_documents(docs)
    print(f"Generated {len(chunks)} chunks")

    assert len(chunks) >= len(docs), "Splitter deveria gerar pelo menos o mesmo número de chunks que documentos"

    print("\n[3] Creating embeddings...")
    embeddings = get_embeddings()
    test_vec = embeddings.embed_query("teste")
    print("Embedding vector size:", len(test_vec))

    assert len(test_vec) > 0, "Embedding falhou"

    print("\n[4] Creating vectorstore (FAISS)...")
    vectorstore = create_vectorstore(chunks, embeddings)

    print("\n[5] Running similarity search...")
    query = "Quem pode participar da Olimpíada Brasileira de Geografia?"
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    print("\n[RESULTS]")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n{i}. Score: {score}")
        print("Text:", doc.page_content[:300])
        print("Metadata:", doc.metadata)

    assert len(results) > 0, "Nenhum resultado retornado pelo retrieval"

    print("\n[SUCCESS] Retrieval pipeline está funcionando corretamente!")
    return True

# -------------------------------------------------
if __name__ == "__main__":
    try:
        success = test_retrieval_pipeline()
        if success:
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
