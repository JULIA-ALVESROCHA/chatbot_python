# test_index.py
from src.rag_pipeline.retrieval.vectorstore import init_vectorstore
from src.app.core.config import settings

vs = init_vectorstore(settings.faiss_index_path)

sample = list(vs.docstore._dict.values())[:5]
for doc in sample:
    print(doc.metadata)