import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from src.app.core.config import settings

logger = logging.getLogger("bgo_chatbot.vectorstore")

# variável global que manterá o índice carregado
_vectorstore = None

# Índice BM25 (busca lexical) construído sobre os mesmos chunks do FAISS.
# Resgata documentos com termos exatos (INEP, calendário, senha) que os
# embeddings posicionam "longe" da pergunta — principal causa dos recall
# gaps nos documentos de suporte.
_bm25 = None
_bm25_docs: List[Document] = []


def _tokenize_pt(text: str) -> List[str]:
    """Tokenização simples para BM25: minúsculas, sem acentos, alfanumérico."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = t.encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", t)


def _build_bm25():
    global _bm25, _bm25_docs
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning(
            "rank_bm25 não instalado — busca híbrida desativada "
            "(pip install rank-bm25). Seguindo apenas com FAISS."
        )
        _bm25, _bm25_docs = None, []
        return

    _bm25_docs = list(_vectorstore.docstore._dict.values())
    corpus = [_tokenize_pt(d.page_content) for d in _bm25_docs]
    _bm25 = BM25Okapi(corpus)
    logger.info("BM25 construído sobre %d chunks", len(_bm25_docs))


def _is_support_doc(doc: Document) -> bool:
    """Documentos de suporte (senhas/acesso/dúvidas) embedam com scores
    sistematicamente mais baixos que o regulamento; recebem threshold próprio."""
    src = ((doc.metadata or {}).get("source") or "").lower()
    src = unicodedata.normalize("NFKD", src).encode("ascii", "ignore").decode()
    return any(w in src for w in ("senha", "acesso", "duvida", "procedimento", "suporte"))


def init_vectorstore(index_path: str):
    global _vectorstore

    index_dir = Path(index_path)

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index não encontrado em {index_path}. "
            f"Execute primeiro o script build_index.py."
        )

    # Cria embeddings (TEM que ser o mesmo modelo usado no build_index)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    # Carrega FAISS
    _vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True  # necessário para FAISS
    )

    _build_bm25()

    return _vectorstore


def create_vectorstore(
    documents: List[Document],
    embeddings: Embeddings,
) -> FAISS:
    """
    Cria um novo vectorstore FAISS a partir de documentos e embeddings.
    
    Esta função é útil para testes e para criar índices do zero.
    Para produção, use init_vectorstore() para carregar um índice salvo.
    
    Args:
        documents: Lista de Document objects para indexar
        embeddings: Instância de Embeddings para gerar vetores
        
    Returns:
        FAISS: Vectorstore FAISS criado
        
    Example:
        >>> from src.rag_pipeline.retrieval.embeddings import get_embeddings
        >>> embeddings = get_embeddings()
        >>> vectorstore = create_vectorstore(chunks, embeddings)
    """
    if not documents:
        raise ValueError("Cannot create vectorstore from empty document list")
    
    if not embeddings:
        raise ValueError("Embeddings instance is required")
    
    print(f"[VECTORSTORE] Creating FAISS index from {len(documents)} documents")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("[VECTORSTORE] FAISS index created successfully")
    
    return vectorstore


class _DiverseRetriever:
    """
    Retriever híbrido: FAISS (semântico) + BM25 (lexical), fundidos por
    Reciprocal Rank Fusion, com corte por score, threshold por família de
    documento e diversidade por página.

    Regras de aceitação de um candidato:
      - score semântico >= threshold da família do documento
        (regulamento: retrieval_score_threshold; docs de suporte:
        support_score_threshold, mais baixo), OU
      - está no top-N do BM25 com score lexical > 0 (match de termos exatos
        resgata o que o embedding perdeu: "INEP", "calendário", "senha").

    Ordenação final por RRF; diversidade: max_chunks_per_page por
    (fonte, página) + dedupe de conteúdo. Interface: invoke / ainvoke /
    get_relevant_documents.
    """

    RRF_K = 60  # constante padrão do Reciprocal Rank Fusion

    def __init__(self, vectorstore, k: int):
        self._vs = vectorstore
        self._k = k

    # ---------------- candidatos ----------------

    def _bm25_candidates(self, query: str):
        """[(doc, bm25_rank)] com score > 0, limitado a fetch_k."""
        if _bm25 is None:
            return []
        fetch_k = getattr(settings, "retrieval_fetch_k", 20)
        scores = _bm25.get_scores(_tokenize_pt(query))
        ranked = sorted(
            ((i, sc) for i, sc in enumerate(scores) if sc > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:fetch_k]
        return [(_bm25_docs[i], rank) for rank, (i, _sc) in enumerate(ranked, 1)]

    # ---------------- fusão + seleção ----------------

    def _fuse_and_select(self, sem_docs_scores, bm25_ranked):
        base_th = settings.retrieval_score_threshold
        support_th = getattr(settings, "support_score_threshold", 0.12)
        bm25_top_accept = getattr(settings, "bm25_top_accept", 5)
        per_page_cap = getattr(settings, "max_chunks_per_page", 2)

        def fp(doc):
            return hash(doc.page_content.strip())

        bm25_rank_by_fp = {fp(d): r for d, r in bm25_ranked}

        # pontuação RRF e elegibilidade
        candidates = {}  # fp -> {"doc", "rrf", "eligible"}
        for sem_rank, (doc, score) in enumerate(sem_docs_scores, 1):
            f = fp(doc)
            th = support_th if _is_support_doc(doc) else base_th
            entry = candidates.setdefault(
                f, {"doc": doc, "rrf": 0.0, "eligible": False}
            )
            entry["rrf"] += 1.0 / (self.RRF_K + sem_rank)
            if score >= th:
                entry["eligible"] = True

        for doc, b_rank in bm25_ranked:
            f = fp(doc)
            entry = candidates.setdefault(
                f, {"doc": doc, "rrf": 0.0, "eligible": False}
            )
            entry["rrf"] += 1.0 / (self.RRF_K + b_rank)
            if b_rank <= bm25_top_accept:
                entry["eligible"] = True

        ordered = sorted(
            (c for c in candidates.values() if c["eligible"]),
            key=lambda c: c["rrf"],
            reverse=True,
        )

        selected: List[Document] = []
        per_page_count = {}
        for c in ordered:
            doc = c["doc"]
            meta = doc.metadata or {}
            page_key = (meta.get("source"), meta.get("page"))
            if per_page_count.get(page_key, 0) >= per_page_cap:
                continue
            per_page_count[page_key] = per_page_count.get(page_key, 0) + 1
            selected.append(doc)
            if len(selected) >= self._k:
                break
        return selected

    # ---------------- interfaces ----------------

    def invoke(self, query: str) -> List[Document]:
        fetch_k = getattr(settings, "retrieval_fetch_k", 20)
        sem = self._vs.similarity_search_with_relevance_scores(query, k=fetch_k)
        return self._fuse_and_select(sem, self._bm25_candidates(query))

    async def ainvoke(self, query: str) -> List[Document]:
        fetch_k = getattr(settings, "retrieval_fetch_k", 20)
        sem = await self._vs.asimilarity_search_with_relevance_scores(
            query, k=fetch_k
        )
        return self._fuse_and_select(sem, self._bm25_candidates(query))

    # compatibilidade com código legado
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


def get_retriever(k: int = settings.max_retrieve):
    """
    Retorna um retriever configurado com k documentos.
    """
    if _vectorstore is None:
        raise RuntimeError(
            "Vectorstore não inicializado. "
            "Chame init_vectorstore() na inicialização do servidor."
        )
    return _DiverseRetriever(_vectorstore, k=k)