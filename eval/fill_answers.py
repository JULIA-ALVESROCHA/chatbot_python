#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_answers.py
===============
Runs each test question through YOUR Lumie pipeline (process_query) and writes the
answer into testset_questions.jsonl, so lumie_eval.py can score it.

Your pipeline (pipeline.py) exposes:
    async def process_query(question, language=None, session_id=None)
        -> {"answer": str, "sources": list}

This script imports that function, calls it (handling the async + the global FAISS
vectorstore), and stores result["answer"].

RUN IT FROM THE REPO ROOT so that `src/`, your `.env`, and the FAISS index resolve:

    # from C:\\Users\\julia\\Downloads\\chatbot_python-main
    python eval\\fill_answers.py --dataset eval\\testset_questions.jsonl

If your pipeline.py lives somewhere other than the repo root, set its import path:
    set PIPELINE_MODULE=src.rag_pipeline.pipeline      (Windows)
    python eval\\fill_answers.py --dataset eval\\testset_questions.jsonl
"""

import argparse, asyncio, importlib, json, os, re, sys

# --- make the repo importable no matter where we're launched from ---
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.getcwd(), os.path.dirname(_HERE), _HERE):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# --- locate process_query ---
def _load_process_query():
    candidates = []
    env = os.environ.get("PIPELINE_MODULE")
    if env:
        candidates.append(env)
    candidates += ["pipeline", "src.rag_pipeline.pipeline", "src.pipeline",
                   "app.pipeline", "src.app.pipeline"]
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "process_query"):
                sys.stderr.write(f"[ok] using process_query from '{mod}'\n")
                return m.process_query
        except Exception:
            continue
    sys.exit("Could not import process_query. Run from the repo root, or set "
             "PIPELINE_MODULE to the dotted path of the module that defines it.")

process_query = _load_process_query()

# --- one persistent event loop for all the async calls ---
_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_LOOP)

def _warm_vectorstore():
    """Load the global FAISS vectorstore exactly like the app server does."""
    try:
        from src.rag_pipeline.retrieval.vectorstore import init_vectorstore
        from src.app.core.config import settings
        init_vectorstore(settings.faiss_index_path)
        sys.stderr.write(f"[ok] vectorstore loaded from {settings.faiss_index_path}\n")
    except Exception as exc:
        sys.stderr.write(f"[warn] could not warm vectorstore: {exc}\n")

_NOT_READY = "sistema de busca ainda não está pronto"

def get_lumie_answer(question: str) -> str:
    result = _LOOP.run_until_complete(
        process_query(question, language="auto", session_id=None)  # no history => independent Qs
    )
    return (result or {}).get("answer", "") or ""

# --- citation stripping (harmless here: your pipeline keeps sources separate) ---
def strip_citations(text):
    if not text:
        return text
    return re.split(r"\n\s*(?:Fontes|Sources|Refer[eê]ncias|Citations?)\s*:",
                    text, flags=re.IGNORECASE)[0].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="testset_questions.jsonl")
    ap.add_argument("--keep-citations", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-answer rows that already have an answer.")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]

    _warm_vectorstore()

    done, checked_ready = 0, False
    for i, r in enumerate(rows, 1):
        if r.get("generated_answer") and not args.overwrite:
            continue
        try:
            ans = get_lumie_answer(r["question"])
        except Exception as exc:                                   # noqa: BLE001
            sys.stderr.write(f"[warn] row {r.get('id')} failed: {exc}\n")
            r["generated_answer"] = None
            continue

        # Stop early instead of filling 200 junk answers if the index never loaded.
        if not checked_ready:
            checked_ready = True
            if _NOT_READY in (ans or "").lower():
                sys.exit("Retrieval returned 'sistema de busca ainda não está pronto' — the "
                         "FAISS vectorstore is not loaded.\nEasiest fix: keep your chatbot "
                         "SERVER running and answer via HTTP instead (tell me the local "
                         "endpoint and I'll switch the hook), or tell me the function that "
                         "initializes the vectorstore so I can call it here.")

        r["generated_answer"] = ans if args.keep_citations else strip_citations(ans)
        done += 1
        if i % 20 == 0:
            sys.stderr.write(f"  {i}/{len(rows)} answered\n")

    with open(args.dataset, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    sys.stderr.write(f"Done. Filled {done} answers in {args.dataset}\n")

if __name__ == "__main__":
    main()