"""
fix_dataset_chunks.py
======================
Rewrites relevant_chunks in dataset_retrival.json from AI-generated
shorthand formats to the real chunk_ids in the FAISS index.

Handles all formats found in the dataset:
  - "regulamento-pagN"          → regulamento_11obg_2026_1_pN_cX
  - "igeo-edital-pagN"          → edital chunk from iGeo PDF
  - "edital-igeo-pagN"          → same as above (alias)
  - "temas-ods-pagN"            → temas/ODS PDF chunks
  - "modelo-questoes-pagN"      → modelo questoes PDF chunks
  - "procedimentos-senhas-pagN" → procedimentos PDF chunks
  - "duvidas-acesso-pagN"       → duvidas/suporte PDF chunks
  - "senhas-pagN"               → same as procedimentos (alias)
  - "Regulamento_2025_.pdf - Section X.X"   → any regulamento chunk
  - "Edital_Selec_a_o_...pdf - Section X.X" → any igeo edital chunk
  - Any unrecognized format → kept as-is with a warning

Run from chatbot_python/:
  python fix_dataset_chunks.py
"""

import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
def _find_project_root(marker="src"):
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise RuntimeError(f"Could not find '{marker}/' in parent directories.")

_PROJECT_ROOT = _find_project_root("src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_IN  = os.path.join(_PROJECT_ROOT, "tests", "retrival_quality", "dataset_retrival.json")
DATASET_OUT = os.path.join(_PROJECT_ROOT, "tests", "retrival_quality", "dataset_retrival.json")
BACKUP_OUT  = os.path.join(_PROJECT_ROOT, "tests", "retrival_quality", "dataset_retrival.backup.json")

# ---------------------------------------------------------------------------
# Prefix -> needle mapping.
# The needle is matched against chunk_id.lower() and source.lower().
# More specific prefixes must come BEFORE shorter/overlapping ones.
# ---------------------------------------------------------------------------
SOURCE_KEYWORD_MAP = [
    # (prefix_in_dataset_lowercase,        needle_in_chunk_id_or_source)
    ("igeo-edital",                        "igeo"),
    ("edital-igeo",                        "igeo"),
    ("temas-ods",                          "temas"),
    ("modelo-questoes",                    "modelo"),
    ("procedimentos-senhas",               "procedimentos"),
    ("duvidas-acesso",                     "duvidas"),
    ("senhas",                             "procedimentos"),
    ("regulamento",                        "regulamento"),

    # Section-style PDF filename refs (lowercased substrings)
    ("regulamento_2025_.pdf",              "regulamento"),
    ("edital_selec_a_o_equipebrasil",      "igeo"),
    ("du_vidas_e_suporte",                 "duvidas"),
    ("procedimentos_alteracaosenhas",      "procedimentos"),
    ("anexo_2_-_temas_ods",               "temas"),
    ("modelo_quest",                       "modelo"),
]


def find_needle(old_lower):
    """Return the needle string for the first matching prefix, or None."""
    for prefix, needle in SOURCE_KEYWORD_MAP:
        if old_lower.startswith(prefix) or prefix in old_lower:
            return needle
    return None


# ---------------------------------------------------------------------------
# Load all real chunk_ids from FAISS
# ---------------------------------------------------------------------------
def load_index():
    from src.app.core.config import settings
    from src.rag_pipeline.retrieval.vectorstore import init_vectorstore

    print("  Loading FAISS index...")
    index_path = os.path.join(_PROJECT_ROOT, settings.faiss_index_path)
    vs = init_vectorstore(index_path)

    page_pat = re.compile(r"_p(\d+)_c\d+$")

    # page_index[needle][page_number] = [chunk_id, ...]
    page_index = defaultdict(lambda: defaultdict(list))
    # full_index[needle] = [chunk_id, ...]
    full_index = defaultdict(list)

    for _doc_id, doc in vs.docstore._dict.items():
        chunk_id = doc.metadata.get("chunk_id", "")
        source   = doc.metadata.get("source", "").lower().replace("\\", "/")

        if not chunk_id:
            continue

        cid_lower = chunk_id.lower()

        # find which needle this chunk belongs to
        matched_needle = None
        for _prefix, needle in SOURCE_KEYWORD_MAP:
            if needle in cid_lower or needle in source:
                matched_needle = needle
                break

        if matched_needle is None:
            continue

        full_index[matched_needle].append(chunk_id)

        m = page_pat.search(chunk_id)
        if m:
            page_index[matched_needle][int(m.group(1))].append(chunk_id)

    total = sum(len(v) for v in full_index.values())
    print(f"  Found {total} chunks across {len(full_index)} document groups:")
    for needle, ids in sorted(full_index.items()):
        pages = sorted(page_index[needle].keys())
        print(f"    [{needle:20s}] {len(ids):4d} chunks | pages: {pages[:8]}{'...' if len(pages) > 8 else ''}")

    return page_index, full_index


# ---------------------------------------------------------------------------
# Resolve one old chunk string -> list of real chunk_ids
# ---------------------------------------------------------------------------
def resolve_chunk(old, page_index, full_index):
    """Returns (list_of_real_ids, warning_or_None)."""
    old_lower = old.lower().strip()

    # ------------------------------------------------------------------
    # Format 1: "prefix-pagN"  e.g. "regulamento-pag13", "temas-ods-pag2"
    # ------------------------------------------------------------------
    page_match = re.search(r"pag(\d+)", old_lower)
    if page_match:
        page_num = int(page_match.group(1))
        needle = find_needle(old_lower)

        if needle is None:
            return [], f"Unrecognized prefix in: '{old}'"

        matches = page_index.get(needle, {}).get(page_num, [])
        if matches:
            return sorted(matches), None

        # Page missing: fall back to first available chunk from that doc
        fallback = sorted(full_index.get(needle, []))
        if fallback:
            return [fallback[0]], (
                f"Page {page_num} not found for needle='{needle}' (from '{old}') "
                f"-- used fallback: {fallback[0]}"
            )
        return [], f"No chunks at all for needle='{needle}' (from '{old}')"

    # ------------------------------------------------------------------
    # Format 2: "Filename.pdf - Section X"  or any section-style ref
    # ------------------------------------------------------------------
    if ".pdf" in old_lower or "section" in old_lower:
        needle = find_needle(old_lower)

        if needle is None:
            return [], f"Could not identify source from: '{old}'"

        all_chunks = sorted(full_index.get(needle, []))
        if all_chunks:
            # Return ALL chunks for that document (the metric handles recall)
            return all_chunks, None
        return [], f"No chunks found for needle='{needle}' (from '{old}')"

    # ------------------------------------------------------------------
    # Format 3: unrecognized
    # ------------------------------------------------------------------
    return [], f"Could not parse format of: '{old}'"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  fix_dataset_chunks.py")
    print("=" * 60)

    with open(DATASET_IN, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Backup first
    with open(BACKUP_OUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  Backup saved -> {BACKUP_OUT}")

    page_index, full_index = load_index()

    total     = len(dataset)
    fixed     = 0
    not_found = 0
    warnings  = []

    for entry in dataset:
        old_chunks = entry.get("relevant_chunks", [])
        new_chunks = []
        seen = set()

        for old in old_chunks:
            resolved, warn = resolve_chunk(old, page_index, full_index)

            if warn:
                warnings.append(f"  [WARN] {warn}  (id={entry.get('id')})")

            if resolved:
                fixed += 1
                for cid in resolved:
                    if cid not in seen:
                        new_chunks.append(cid)
                        seen.add(cid)
            else:
                not_found += 1
                if old not in seen:
                    new_chunks.append(old)   # keep original so failures are visible
                    seen.add(old)

        entry["relevant_chunks"] = new_chunks

    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n  Done.")
    print(f"  Total entries      : {total}")
    print(f"  Chunk refs fixed   : {fixed}")
    print(f"  Chunk refs failed  : {not_found}")
    print(f"  Dataset saved      -> {DATASET_OUT}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:30]:
            print(w)
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more.")
    else:
        print("\n  No warnings -- all chunk refs resolved successfully!")

    print("=" * 60)


if __name__ == "__main__":
    main()