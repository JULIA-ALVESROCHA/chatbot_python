from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-encoder reranker aligned with SDD.
    Jointly encodes (query, passage).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = []
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
