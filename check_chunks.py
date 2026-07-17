import sys
sys.path.insert(0, ".")
from src.app.core.config import settings
from src.rag_pipeline.retrieval.vectorstore import init_vectorstore

vs = init_vectorstore(settings.faiss_index_path)
docs = vs.similarity_search("regulamento OBG", k=10)
for d in docs:
    print(d.metadata.get("chunk_id"), "|", d.metadata.get("source"))