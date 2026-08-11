"""
src/rag_pipeline/reranker/model_wrapper.py

Four changes from the previous version:

1. MULTILINGUAL MODEL. ms-marco-MiniLM-L-6-v2 is trained on MS MARCO, an
   ENGLISH passage-ranking dataset. Feeding it Portuguese queries against
   Portuguese chunks from the regulamento produces near-noise reordering.
   With USE_RERANKER=true locally and False in production, that also means
   the two environments were running measurably different retrieval.

2. PINNED REVISION. An unpinned HF model resolves to whatever is current on
   the Hub at first download. Local and production can silently end up on
   different weights. revision= pins the commit.

3. SINGLETON. CrossEncoder(...) in __init__ loads ~150MB. If anything
   constructs this class per request, every question pays that cost. get()
   loads once per process.

4. NO DOCSTORE MUTATION. The old code did
       doc.metadata["rerank_score"] = float(score)
   on Documents owned by the FAISS docstore, so scores from one request
   persisted into the next. Now it copies first.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger("bgo_chatbot.reranker")

# Multilingual cross-encoder. Verify the exact repo and pick a commit sha:
#     huggingface-cli scan-cache
#     https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
# Heavier but stronger alternative if the host has the RAM:
#     BAAI/bge-reranker-base
DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Set this to a real commit sha from the model page. "main" is a moving
# target and defeats the point of pinning.
DEFAULT_REVISION: Optional[str] = None

_instance: Optional["CrossEncoderReranker"] = None
_lock = threading.Lock()


class CrossEncoderReranker:
    """Cross-encoder reranker. Jointly encodes (query, passage)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        revision: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 512,
    ):
        from src.app.core.config import settings

        self.model_name = model_name or getattr(
            settings, "reranker_model", DEFAULT_MODEL
        )
        self.revision = revision or getattr(
            settings, "reranker_revision", DEFAULT_REVISION
        )
        self.device = device
        self.max_length = max_length

        kwargs = {"device": device, "max_length": max_length}
        if self.revision:
            kwargs["revision"] = self.revision

        logger.info(
            "Loading cross-encoder %s (revision=%s, device=%s)",
            self.model_name, self.revision or "UNPINNED", device,
        )
        if not self.revision:
            logger.warning(
                "Reranker revision is not pinned. Local and production may "
                "resolve to different weights."
            )

        try:
            self.model = CrossEncoder(self.model_name, **kwargs)
        except TypeError:
            # Older sentence-transformers does not accept revision=.
            kwargs.pop("revision", None)
            self.model = CrossEncoder(self.model_name, **kwargs)
            logger.warning(
                "Installed sentence-transformers ignores revision=; pin the "
                "package version instead."
            )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            # A reranker failure must not cost the user their answer.
            logger.warning("Rerank failed (%s); returning original order", e)
            return documents[:top_k]

        scored = []
        for doc, score in zip(documents, scores):
            # Copy: the originals belong to the FAISS docstore.
            copy = Document(
                page_content=doc.page_content,
                metadata={**(doc.metadata or {}), "rerank_score": float(score)},
            )
            scored.append((float(score), copy))

        scored.sort(key=lambda x: x[0], reverse=True)

        logger.debug(
            "Reranked %d -> %d | best=%.3f worst=%.3f",
            len(documents), min(top_k, len(scored)),
            scored[0][0], scored[-1][0],
        )
        return [doc for _, doc in scored[:top_k]]


def get_reranker() -> CrossEncoderReranker:
    """Process-wide singleton. Use this instead of constructing directly."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CrossEncoderReranker()
    return _instance