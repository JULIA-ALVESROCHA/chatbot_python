from src.app.core.config import settings
from src.rag_pipeline.retrieval import vectorstore as vs

vs.init_vectorstore(settings.faiss_index_path)
q = "Quantos alunos formam uma equipe?"
for th in (0.0, 0.30, 0.40, 0.45, 0.50, 0.60):
    settings.retrieval_score_threshold = th
    docs = vs.get_retriever(k=5).invoke(q)
    print(f"th={th}: {len(docs)} docs ->", [d.metadata.get("page") for d in docs])