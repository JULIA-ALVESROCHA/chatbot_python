"""
src/rag_pipeline/retrieval/manifest.py

Records exactly what went into the index, and refuses to serve an index
that does not match the current settings.

Call write_manifest() at the end of scripts/build_index.py, and
verify_manifest() from src/app/main.py startup.

This is what turns "it answers on one machine and not the other" from a
mystery into a two-line diff.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("bgo_chatbot.manifest")

MANIFEST_NAME = "index_manifest.json"


def _pkg_versions() -> dict:
    out = {}
    for pkg in ("langchain", "langchain-core", "langchain-community",
                "langchain-openai", "faiss-cpu", "openai", "numpy"):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = None
    return out


def _corpus_sha(raw_dir: Path) -> dict:
    """Per-file hashes, so you can see WHICH document changed."""
    files = {}
    for f in sorted(raw_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md"):
            h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            files[f.name] = {"sha": h, "bytes": f.stat().st_size}
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()[:16]
    return {"files": files, "combined_sha": combined}


def _vec_checksum(vs) -> dict:
    idx = vs.index
    vecs = idx.reconstruct_n(0, idx.ntotal)
    norms = np.linalg.norm(vecs, axis=1)
    return {
        "ntotal": int(idx.ntotal),
        "dim": int(idx.d),
        "faiss_class": type(idx).__name__,
        "mean_norm": round(float(norms.mean()), 6),
        "unit_norm": bool(np.allclose(norms, 1.0, atol=1e-3)),
        "sha": hashlib.sha256(np.ascontiguousarray(vecs).tobytes()).hexdigest()[:16],
    }


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def write_manifest(vs, chunks: List, index_path: str, raw_dir: str = "data/raw") -> dict:
    """Call at the end of scripts/build_index.py, after saving the index."""
    from src.app.core.config import settings

    sizes = [len(c.page_content) for c in chunks] or [0]
    items = sum(1 for c in chunks if (c.metadata or {}).get("item"))

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _pkg_versions(),
        "embedding_model": settings.embedding_model,
        "distance_strategy": getattr(settings, "distance_strategy", None),
        "chunking": {
            "strategy": "item_aware",
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "n_chunks": len(chunks),
            "chars_min": min(sizes),
            "chars_median": sorted(sizes)[len(sizes) // 2],
            "chars_max": max(sizes),
            "chunks_with_item_number": items,
        },
        "corpus": _corpus_sha(Path(raw_dir)),
        "vectors": _vec_checksum(vs),
    }

    path = Path(index_path).parent / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s (%d chunks, vec sha %s)",
                path, len(chunks), manifest["vectors"]["sha"])
    return manifest


def read_manifest(index_path: str) -> Optional[dict]:
    path = Path(index_path).parent / MANIFEST_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verify_manifest(index_path: str, strict: bool = False) -> List[str]:
    """
    Compare the manifest against current settings and corpus.
    Returns a list of problems. Raises if strict=True and any are found.

    Call from src/app/main.py startup so a stale index fails loudly instead
    of quietly returning worse answers.
    """
    from src.app.core.config import settings

    problems: List[str] = []
    m = read_manifest(index_path)
    if m is None:
        problems.append(
            f"No {MANIFEST_NAME} next to the index. Rebuild with "
            f"scripts/build_index.py to make this index identifiable."
        )
        if strict:
            raise RuntimeError(problems[0])
        logger.warning(problems[0])
        return problems

    if m.get("embedding_model") != settings.embedding_model:
        problems.append(
            f"Index built with embedding_model={m.get('embedding_model')!r} "
            f"but settings say {settings.embedding_model!r}. Query and index "
            f"vectors are not comparable — rebuild."
        )

    ds = getattr(settings, "distance_strategy", None)
    if ds and m.get("distance_strategy") and m["distance_strategy"] != ds:
        problems.append(
            f"Index distance_strategy={m['distance_strategy']} but settings "
            f"say {ds}. Score conversion will be wrong."
        )

    if not m.get("vectors", {}).get("unit_norm", True):
        problems.append(
            "Index vectors are NOT unit-norm, so cos = 1 - d^2/2 is invalid. "
            "Do not use the cosine conversion with this index."
        )

    current = _corpus_sha(Path("data/raw")).get("combined_sha")
    if current and m.get("corpus", {}).get("combined_sha") != current:
        old = set(m.get("corpus", {}).get("files", {}))
        new = set(_corpus_sha(Path("data/raw"))["files"])
        detail = ""
        if old - new:
            detail += f" removed={sorted(old - new)}"
        if new - old:
            detail += f" added={sorted(new - old)}"
        problems.append(f"Corpus changed since the index was built.{detail} Rebuild.")

    for p in problems:
        logger.warning("INDEX MANIFEST: %s", p)
    if problems and strict:
        raise RuntimeError("Index manifest check failed:\n  " + "\n  ".join(problems))
    if not problems:
        logger.info("Index manifest OK (built %s, %d chunks, vec sha %s)",
                    m.get("built_at"), m["chunking"]["n_chunks"],
                    m["vectors"]["sha"])
    return problems


if __name__ == "__main__":
    # python -m src.rag_pipeline.retrieval.manifest
    from src.app.core.config import settings
    m = read_manifest(settings.faiss_index_path)
    print(json.dumps(m, indent=2, ensure_ascii=False) if m else "no manifest")
    verify_manifest(settings.faiss_index_path)