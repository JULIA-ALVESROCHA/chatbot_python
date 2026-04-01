"""
retrieval_regression_runner.py
================================
Regression suite that measures retrieval quality (P@k, R@k, F1@k, MAP@k)
for the OBG RAG pipeline.

Dataset schema expected (dataset_retrival.json):
  {
    "id": "OBG_HALLU_001",
    "query": "...",
    "relevant_chunks": ["regulamento-pag13", "regulamento-pag28"],
    "original_category": "undocumented_future_event",
    "expected_behavior": "abstain_if_not_in_db",
    "gold_standard": "...",
    "notes": "..."
  }

The retriever must return objects whose .metadata["source"] (or .page_content
chunk id) matches the strings in relevant_chunks.  Adjust CHUNK_ID_FIELD below
if your metadata key is different (e.g. "chunk_id", "doc_id", "source").
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# PATH FIX — makes `src` importable regardless of where you run the script
# from (e.g. tests/, chatbot_python/, or the repo root).
# This walks up from this file's location until it finds the folder that
# contains `src`, then adds it to sys.path.
# ---------------------------------------------------------------------------
def _find_project_root(marker: str = "src") -> str:
    """Walk up the directory tree until we find the folder containing `marker`."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):  # max 10 levels up — avoids infinite loop
        if os.path.isdir(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break  # reached filesystem root
        current = parent
    raise RuntimeError(
        f"Could not find a parent directory containing '{marker}/'. "
        "Make sure you're running from inside the project tree."
    )

_PROJECT_ROOT = _find_project_root("src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lightweight env loader — reads .env manually so we never import `settings`
# (importing settings triggers vectorstore -> config -> vectorstore circular import)
# ---------------------------------------------------------------------------
def _get_env(key: str, default: str = "") -> str:
    """Return env var `key`, falling back to .env file, then `default`."""
    if key in os.environ:
        return os.environ[key]
    env_file = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as _f:
            for line in _f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    return default


# ---------------------------------------------------------------------------
# !! CONFIGURE THESE TO MATCH YOUR PROJECT !!
# ---------------------------------------------------------------------------

# Path to the evaluation dataset — relative to the project root found above
DATASET_FILE = os.path.join(_PROJECT_ROOT, "tests", "retrival_quality", "dataset_retrival.json")
# Where to write reports
REPORT_DIR = os.path.join(_PROJECT_ROOT, "reports")

# Number of chunks to retrieve per query
K = 5

# The metadata key that holds the chunk identifier on each LangChain Document.
# Common values: "source", "chunk_id", "doc_id", "id"
CHUNK_ID_FIELD = "chunk_id"

# ---------------------------------------------------------------------------
# NOTE: vectorstore is imported lazily inside main() to avoid the circular
# import: vectorstore.py -> config.py -> vectorstore.py
# Do NOT move this import to module level.
# ---------------------------------------------------------------------------


# ===========================================================================
# Metric helpers
# ===========================================================================

def _retrieved_ids(docs, chunk_id_field: str) -> list[str]:
    """Extract chunk IDs from a list of LangChain Documents."""
    ids = []
    for doc in docs:
        # Try metadata first, then fall back to page_content as-is
        chunk_id = doc.metadata.get(chunk_id_field)
        if chunk_id is None:
            # Some setups store the id directly in page_content header — skip
            chunk_id = doc.metadata.get("id") or doc.metadata.get("chunk_id")
        if chunk_id:
            # Strip file extension or path prefix that might differ
            ids.append(os.path.splitext(os.path.basename(str(chunk_id)))[0])
    return ids


def _normalize(chunk_id: str) -> str:
    """Normalise a chunk id for comparison (lowercase, strip extension/path)."""
    return os.path.splitext(os.path.basename(chunk_id.lower().strip()))[0]


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not retrieved:
        return 0.0
    retrieved_k = [_normalize(r) for r in retrieved[:k]]
    relevant_set = {_normalize(r) for r in relevant}
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = [_normalize(r) for r in retrieved[:k]]
    relevant_set = {_normalize(r) for r in relevant}
    hits = sum(1 for r in retrieved_k if r in relevant_set)
    return hits / len(relevant_set)


def f1_at_k(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Mean Average Precision — rewards finding relevant docs early."""
    if not relevant:
        return 0.0
    relevant_set = {_normalize(r) for r in relevant}
    retrieved_k = [_normalize(r) for r in retrieved[:k]]
    hits = 0
    score = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant_set:
            hits += 1
            score += hits / i
    return score / min(len(relevant_set), k)


# ===========================================================================
# Report generation  (mirrors reports.py style)
# ===========================================================================

def generate_retrieval_report(results: list[dict], k: int) -> dict:
    """Aggregate per-query results into a summary + per-category breakdown."""
    total = len(results)

    # Global accumulators
    sum_p = sum_r = sum_f1 = sum_map = 0.0

    # Per-category accumulators
    cat_accum = defaultdict(lambda: {"n": 0, "p": 0.0, "r": 0.0, "f1": 0.0, "map": 0.0})

    for r in results:
        p, rc, f1, ap = r["precision"], r["recall"], r["f1"], r["ap"]
        sum_p  += p
        sum_r  += rc
        sum_f1 += f1
        sum_map += ap

        cat = r.get("category", "uncategorized")
        cat_accum[cat]["n"]   += 1
        cat_accum[cat]["p"]   += p
        cat_accum[cat]["r"]   += rc
        cat_accum[cat]["f1"]  += f1
        cat_accum[cat]["map"] += ap

    summary = {
        "total_queries": total,
        "k": k,
        f"P@{k}":   round(sum_p   / total, 4),
        f"R@{k}":   round(sum_r   / total, 4),
        f"F1@{k}":  round(sum_f1  / total, 4),
        f"MAP@{k}": round(sum_map / total, 4),
    }

    by_category = {}
    for cat, acc in cat_accum.items():
        n = acc["n"]
        by_category[cat] = {
            "total_queries": n,
            f"P@{k}":   round(acc["p"]   / n, 4),
            f"R@{k}":   round(acc["r"]   / n, 4),
            f"F1@{k}":  round(acc["f1"]  / n, 4),
            f"MAP@{k}": round(acc["map"] / n, 4),
        }

    return {
        "summary": summary,
        "by_category": by_category,
        "metric_definitions": {
            f"P@{k}": "Fraction of the top-k retrieved chunks that are relevant",
            f"R@{k}": "Fraction of all relevant chunks found in the top-k",
            f"F1@{k}": "Harmonic mean of P@k and R@k",
            f"MAP@{k}": "Mean Average Precision — rewards relevant chunks appearing earlier",
        },
    }


def save_retrieval_reports(
    raw_results: list[dict],
    report: dict,
    output_dir: str = REPORT_DIR,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path    = os.path.join(output_dir, f"retrieval_results_raw_{ts}.json")
    report_path = os.path.join(output_dir, f"retrieval_report_{ts}.json")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Raw results  → {raw_path}")
    print(f"[OK] Summary      → {report_path}")

    return report_path


# ===========================================================================
# Console printer  (mirrors the table style in your original output)
# ===========================================================================

def _print_report(report: dict, k: int) -> None:
    s = report["summary"]
    sep = "=" * 60

    print(sep)
    print(f"  RESULTADO GLOBAL ({s['total_queries']} queries, k={k})")
    print(sep)
    print(f"  P@{k}   {s[f'P@{k}']:.4f}  — precisão média nos top-{k}")
    print(f"  R@{k}   {s[f'R@{k}']:.4f}  — recall médio nos top-{k}")
    print(f"  F1@{k}  {s[f'F1@{k}']:.4f}  — harmônica P/R")
    print(f"  MAP@{k} {s[f'MAP@{k}']:.4f}  — posição importa (acadêmico)")

    print()
    print("  POR CATEGORIA:")
    header = f"  {'categoria':<35} {'n':>4}  {'P@'+str(k):>6}  {'R@'+str(k):>6}  {'F1@'+str(k):>6}  {'MAP@'+str(k):>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cat, m in sorted(report["by_category"].items()):
        n  = m["total_queries"]
        p  = m[f"P@{k}"]
        r  = m[f"R@{k}"]
        f1 = m[f"F1@{k}"]
        ap = m[f"MAP@{k}"]
        print(f"  {cat:<35} {n:>4}  {p:.4f}  {r:.4f}  {f1:.4f}  {ap:.4f}")

    print(sep)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print(f"  OBG Retrieval Regression  |  k={K}  |  field='{CHUNK_ID_FIELD}'")
    print(f"  Dataset : {DATASET_FILE}")
    print("=" * 60)

    # Load dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        tests = json.load(f)

    total = len(tests)

    # Lazy import — must happen inside main() so that by the time Python
    # executes `from src.rag_pipeline...`, the sys.path fix above is already
    # active AND we avoid the circular import at module load time:
    #   vectorstore.py -> config.py -> vectorstore.py
    from src.rag_pipeline.retrieval.vectorstore import get_retriever, init_vectorstore

    _raw_index_path = _get_env("FAISS_INDEX_PATH", "data/processed/faiss_index")
    index_path = (
        _raw_index_path
        if os.path.isabs(_raw_index_path)
        else os.path.join(_PROJECT_ROOT, _raw_index_path)
    )
    print(f"  Loading FAISS index from: {index_path}")
    init_vectorstore(index_path)
    retriever = get_retriever(k=K)

    results = []

    try:
        for i, test in enumerate(tests, 1):
            test_id  = test.get("id", f"test_{i}")
            query    = test.get("query", "")
            relevant = test.get("relevant_chunks", [])
            category = test.get("original_category", "uncategorized")

            # ----------------------------------------------------------------
            # Retrieve
            # ----------------------------------------------------------------
            try:
                docs = retriever.get_relevant_documents(query)
            except Exception as e:
                print(f"  [{i}/{total}] RETRIEVER ERROR: {e}")
                docs = []

            retrieved_ids = _retrieved_ids(docs, CHUNK_ID_FIELD)

            # ----------------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------------
            p   = precision_at_k(retrieved_ids, relevant, K)
            r   = recall_at_k(retrieved_ids, relevant, K)
            f1  = f1_at_k(p, r)
            ap  = average_precision_at_k(retrieved_ids, relevant, K)

            results.append({
                "id":            test_id,
                "category":      category,
                "query":         query,
                "relevant":      relevant,
                "retrieved":     retrieved_ids,
                "precision":     round(p,  4),
                "recall":        round(r,  4),
                "f1":            round(f1, 4),
                "ap":            round(ap, 4),
                "expected_behavior": test.get("expected_behavior"),
                "notes":         test.get("notes"),
            })

            # Progress every 50
            if i % 50 == 0:
                print(f"  [{i}/{total}] processadas…")

    except KeyboardInterrupt:
        print("\n  [Interrompido pelo usuário — resultados parciais serão salvos]")

    print(f"  [{total}/{total}] processadas…")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    report      = generate_retrieval_report(results, K)
    report_path = save_retrieval_reports(results, report)

    _print_report(report, K)
    print(f"  Relatório salvo em: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()