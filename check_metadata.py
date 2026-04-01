import sys
sys.path.insert(0, ".")
from src.app.core.config import settings
from src.rag_pipeline.retrieval.vectorstore import init_vectorstore

vs = init_vectorstore(settings.faiss_index_path)
docs = vs.similarity_search("test", k=1)
print(docs[0].metadata)