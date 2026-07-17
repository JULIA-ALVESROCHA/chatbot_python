from langchain_core.documents import Document
from src.rag_pipeline.reranker.model_wrapper import CrossEncoderReranker


def main():
    print("🔹 Initializing cross-encoder reranker...")
    reranker = CrossEncoderReranker()

    query = "Quem pode participar da Olimpíada Brasileira de Geografia?"

    docs = [
        Document(
            page_content=(
                "A Olimpíada Brasileira de Geografia é destinada a estudantes "
                "do ensino médio regularmente matriculados."
            ),
            metadata={
                "source": "regulamento.pdf",
                "section": "Elegibilidade",
            },
        ),
        Document(
            page_content=(
                "A fase final da competição ocorre presencialmente "
                "no mês de dezembro."
            ),
            metadata={
                "source": "regulamento.pdf",
                "section": "Cronograma",
            },
        ),
        Document(
            page_content=(
                "Podem participar estudantes do ensino médio de escolas "
                "públicas e privadas de todo o Brasil."
            ),
            metadata={
                "source": "regulamento.pdf",
                "section": "Participantes",
            },
        ),
    ]

    print("\n🔹 Running reranker...\n")
    ranked_docs = reranker.rerank(query, docs, top_k=3)

    print("✅ Rerank result:\n")
    for i, doc in enumerate(ranked_docs, start=1):
        score = doc.metadata.get("rerank_score")
        print(f"{i}. Score = {score:.4f}")
        print(f"   Text: {doc.page_content}")
        print(f"   Metadata: {doc.metadata}\n")


if __name__ == "__main__":
    main()
