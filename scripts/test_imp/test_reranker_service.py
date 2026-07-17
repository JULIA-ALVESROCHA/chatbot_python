import sys
from pathlib import Path
import asyncio

# --- garante que o src/ está no PYTHONPATH ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from langchain_core.documents import Document
from src.rag_pipeline.reranker.reranker import rerank_documents


async def main():
    print("🔹 Testing reranker service (async)...\n")

    query = "Quem pode participar da Olimpíada Brasileira de Geografia?"

    docs = [
        Document(
            page_content=(
                "A Olimpíada Brasileira de Geografia é destinada a estudantes "
                "do ensino médio regularmente matriculados."
            ),
            metadata={"id": "eligibilidade"},
        ),
        Document(
            page_content=(
                "A fase final da competição ocorre presencialmente "
                "no mês de dezembro."
            ),
            metadata={"id": "cronograma"},
        ),
        Document(
            page_content=(
                "Podem participar estudantes do ensino médio de escolas "
                "públicas e privadas de todo o Brasil."
            ),
            metadata={"id": "participantes"},
        ),
    ]

    ranked_docs = await rerank_documents(
        query=query,
        documents=docs,
        top_k=3,
    )

    print("✅ Rerank result:\n")
    for i, doc in enumerate(ranked_docs, start=1):
        score = doc.metadata.get("rerank_score")
        print(f"{i}. id={doc.metadata.get('id')} | score={score:.4f}")
        print(f"   {doc.page_content}\n")

    # --- validações simples (sanity checks) ---
    assert ranked_docs[0].metadata["id"] != "cronograma", (
        "❌ Documento irrelevante apareceu em primeiro lugar"
    )

    assert ranked_docs[-1].metadata["id"] == "cronograma", (
        "❌ Documento de cronograma deveria estar por último"
    )

    print("🎉 Reranker service test PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
