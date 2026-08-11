"""
diagnose_v2.py - Lumie / OBG RAG retrieval diagnostics

Purpose
-------
1. Prove (or disprove) that the low scores come from LangChain's Euclidean
   -> "relevance score" rescale rather than from bad retrieval.
2. Fingerprint the index so two machines can be compared objectively.
3. Sweep the threshold in COSINE space and report retrieval recall.

Run on BOTH machines and diff the output.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings

INDEX_PATH = "data/processed/faiss_index"
EMBED_MODEL = "text-embedding-3-large"
K = 20  # retrieve wide, filter later


# ---------------------------------------------------------------- helpers
def l2_to_cosine(distance: float) -> float:
    """For UNIT-NORM vectors: d^2 = 2 - 2cos  ->  cos = 1 - d^2/2."""
    return 1.0 - (distance ** 2) / 2.0


def langchain_euclidean_rescale(distance: float) -> float:
    """What similarity_search_with_relevance_scores() is doing to you."""
    return 1.0 - distance / math.sqrt(2)


def fingerprint(vs: FAISS) -> dict:
    """Stable identity of the index. If two machines differ, this shows it."""
    idx = vs.index
    vecs = idx.reconstruct_n(0, idx.ntotal)
    norms = np.linalg.norm(vecs, axis=1)
    doc_ids = sorted(vs.docstore._dict.keys())
    return {
        "ntotal": int(idx.ntotal),
        "dim": int(idx.d),
        "faiss_class": type(idx).__name__,
        "distance_strategy": str(getattr(vs, "distance_strategy", None)),
        "mean_norm": round(float(norms.mean()), 6),
        "normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "vec_checksum": hashlib.sha256(
            np.ascontiguousarray(vecs).tobytes()
        ).hexdigest()[:16],
        "docid_checksum": hashlib.sha256(
            "".join(doc_ids).encode()
        ).hexdigest()[:16],
        "index_file_mtime": os.path.getmtime(
            Path(INDEX_PATH) / "index.faiss"
        ),
    }


# ---------------------------------------------------------------- main
def main():
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.load_local(
        INDEX_PATH, emb, allow_dangerous_deserialization=True
    )

    print("=== INDEX FINGERPRINT (diff this between machines) ===")
    print(json.dumps(fingerprint(vs), indent=2))

    # --- SANITY CHECK: does the query embedder match the index embedder? ---
    # Take a chunk that is IN the index and query with its own exact text.
    # Cosine must be ~1.0. If it is not, the index was built with a
    # different model/version than the one loaded here. That alone explains
    # machine-A-answers / machine-B-refuses.
    first_doc = next(iter(vs.docstore._dict.values()))
    probe_text = first_doc.page_content[:400]
    hits = vs.similarity_search_with_score(probe_text, k=1)
    d0 = float(hits[0][1])
    print("\n=== SELF-RETRIEVAL SANITY CHECK ===")
    print(f"raw L2 distance   : {d0:.4f}")
    print(f"cosine            : {l2_to_cosine(d0):.4f}   (expect >= 0.98)")
    if l2_to_cosine(d0) < 0.95:
        print("!! FAIL: query embedder != index embedder. Rebuild the index.")
    else:
        print("   OK: embedder and index agree.")

    # --- SCORE-SPACE COMPARISON ---
    queries = [
        ("pt", "quem pode participar da OBG?"),
        ("pt", "quando serao as provas?"),
        ("pt", "como recupero minha senha?"),
        ("pt", "quais navegadores sao suportados?"),
        ("pt", "o link do token expirou, o que faco?"),
        ("en", "how do I recover my password?"),
        ("en", "who can participate in the OBG?"),
        ("en", "which browsers are supported?"),
    ]

    print("\n=== SCORE SPACE: raw L2 vs cosine vs LangChain rescale ===")
    rows = []
    for lang, q in queries:
        hits = vs.similarity_search_with_score(q, k=K)
        print(f"\nQ [{lang}] {q}")
        print(f"  {'L2':>7} {'cosine':>7} {'lc_score':>9}  source")
        for doc, dist in hits[:5]:
            dist = float(dist)
            src = Path(doc.metadata.get("source", "?")).name[:42]
            print(
                f"  {dist:7.4f} {l2_to_cosine(dist):7.4f} "
                f"{langchain_euclidean_rescale(dist):9.4f}  {src}"
            )
        rows.append(
            (lang, q, [l2_to_cosine(float(d)) for _, d in hits])
        )

    # --- THRESHOLD SWEEP IN COSINE SPACE ---
    print("\n=== THRESHOLD SWEEP (cosine) : avg chunks kept per query ===")
    print(f"  {'thr':>5} {'pt':>6} {'en':>6}  {'queries with 0 chunks':>22}")
    for thr in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        kept_pt, kept_en, empty = [], [], 0
        for lang, _, cosines in rows:
            k = sum(1 for c in cosines if c >= thr)
            (kept_pt if lang == "pt" else kept_en).append(k)
            if k == 0:
                empty += 1
        print(
            f"  {thr:5.2f} {np.mean(kept_pt):6.1f} {np.mean(kept_en):6.1f}"
            f"  {empty:22d}"
        )

    print(
        "\nPick the lowest threshold that still excludes junk, then ALWAYS "
        "pass >=1 chunk to the reranker. Threshold the RERANKER score, not "
        "this one."
    )


if __name__ == "__main__":
    main()