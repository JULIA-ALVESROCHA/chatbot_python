#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lumie_eval.py
=============
Re-evaluation harness for the Lumie RAG chatbot (Brazilian Geography Olympiad).

Metrics (per the BOTTLEHUMOR paper, Hwang et al., Findings of ACL 2025):
  (A) Atomic Precision / Recall / F1
        recall    = fraction of GOLD atoms conveyed in the prediction   (Fig. 12 prompt)
        precision = fraction of PRED atoms inferable from the gold       (Fig. 13 prompt)
        f1        = harmonic mean per instance, reported as macro-F1
  (B) FactScore (Min et al., 2023)
        fraction of PRED atoms SUPPORTED by the knowledge source (faithfulness)

Judge / decomposer: OpenAI gpt-4o-mini (counterpart to the paper's Gemini-1.5-Flash),
temperature 0.2. The harness ALSO records which gold atoms were missed (recall gaps)
and which predicted atoms were unsupported (hallucinations), and writes a clean,
fully data-driven Markdown report (results_report.md) with the analysis computed
from the numbers - no pre-written prose.

    export OPENAI_API_KEY=...
    python lumie_eval.py --dataset testset_questions.jsonl --corpus knowledge_source.txt \
        --seeds 13 21 42 --model gpt-4o-mini --out results.json --report results_report.md
"""

import argparse, hashlib, json, os, statistics as stats, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Install the OpenAI SDK first:  pip install openai")

_CLIENT: Optional["OpenAI"] = None
def client() -> "OpenAI":
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI()
    return _CLIENT

# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #
_CACHE_PATH = ".lumie_cache.json"
_CACHE: Dict[str, str] = {}
def _load_cache():
    global _CACHE
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, encoding="utf-8") as fh:
            _CACHE = json.load(fh)
def _save_cache():
    with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(_CACHE, fh, ensure_ascii=False)
def _key(model, temp, seed, messages):
    raw = json.dumps([model, temp, seed, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def chat(messages, model, temperature=0.2, seed=0, max_retries=5) -> str:
    ck = _key(model, temperature, seed, messages)
    if ck in _CACHE:
        return _CACHE[ck]
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=model, messages=messages, temperature=temperature, seed=seed)
            out = (resp.choices[0].message.content or "").strip()
            _CACHE[ck] = out
            return out
        except Exception as exc:                                   # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            sys.stderr.write(f"[retry {attempt+1}] {exc}\n")
            time.sleep(delay); delay *= 2
    return ""

# --------------------------------------------------------------------------- #
# Prompts                                                                     #
# --------------------------------------------------------------------------- #
DECOMPOSE_SYS = (
    "You decompose a passage into atomic facts. An atomic fact is a single, "
    "self-contained, verifiable statement. Resolve pronouns and references. Keep "
    "each fact in the SAME language as the input. Ignore greetings, sign-offs, and "
    "politeness formulas. Return ONLY a JSON array of strings.")
RECALL_SYS = (
    "You judge whether the information in [Sentence1] is conveyed in [Sentence2]. "
    "[Sentence2] may contain several sentences. Treat Portuguese<->English paraphrases "
    "as equivalent. Do not rely on outside assumptions. Answer one word: Yes or No.")
PRECISION_SYS = (
    "You judge whether the information in [Sentence1] is inferable from [Sentence2]. "
    "Mark Yes if it is stated, implied, reworded, or supported; No if absent, "
    "uninferable, or contradicted. Treat PT<->EN paraphrases as equivalent. "
    "Answer one word: Yes or No.")
SUPPORT_SYS = (
    "You verify factual support. Decide whether [Claim] is SUPPORTED by the [Source]. "
    "Mark Yes only if the source states or directly entails the claim; No if the source "
    "is silent on it or contradicts it. Treat PT<->EN paraphrases as equivalent. Do not "
    "use outside knowledge. Answer one word: Yes or No.")

def _yes(t): return t.strip().lower().startswith(("yes", "sim"))

def decompose(text, model, seed) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out = chat([{"role": "system", "content": DECOMPOSE_SYS},
                {"role": "user", "content": f"[Passage]:\n{text}\n\n[Atomic facts JSON array]:"}],
               model=model, temperature=0.2, seed=seed)
    try:
        s, e = out.find("["), out.rfind("]")
        arr = json.loads(out[s:e + 1])
        facts = [str(a).strip() for a in arr if str(a).strip()]
        if facts:
            return facts
    except Exception:                                              # noqa: BLE001
        pass
    return [ln.strip(" -*1234567890.)\t") for ln in out.splitlines()
            if len(ln.strip(" -*1234567890.)\t")) > 3]

def _verify(sys_prompt, s1, s2, model, seed) -> bool:
    return _yes(chat([{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": f"[Sentence1]: {s1}\n[Sentence2]: {s2}\n[Output]:"}],
                     model=model, temperature=0.2, seed=seed))

def verify_support(atom, source, model, seed) -> bool:
    return _yes(chat([{"role": "system", "content": SUPPORT_SYS},
                      {"role": "user", "content": f"[Source]:\n{source}\n\n[Claim]: {atom}\n[Output]:"}],
                     model=model, temperature=0.2, seed=seed))

# --------------------------------------------------------------------------- #
# Per-instance scoring (records diagnostics)                                  #
# --------------------------------------------------------------------------- #
def score_instance(rec, source, model, seed) -> dict:
    gold, pred = rec["gold_answer"], rec["generated_answer"]
    gold_atoms = decompose(gold, model, seed)
    pred_atoms = decompose(pred, model, seed)

    missed = [a for a in gold_atoms if not _verify(RECALL_SYS, a, pred, model, seed)]
    recall = (len(gold_atoms) - len(missed)) / len(gold_atoms) if gold_atoms else float("nan")

    unsup_prec = [a for a in pred_atoms if not _verify(PRECISION_SYS, a, gold, model, seed)]
    precision = (len(pred_atoms) - len(unsup_prec)) / len(pred_atoms) if pred_atoms else float("nan")

    unsupported = [a for a in pred_atoms if not verify_support(a, source, model, seed)]
    factscore = (len(pred_atoms) - len(unsupported)) / len(pred_atoms) if pred_atoms else float("nan")

    if precision == precision and recall == recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision == precision and recall == recall:
        f1 = 0.0
    else:
        f1 = float("nan")

    return {"id": rec.get("id"), "lang": rec.get("lang", "pt"),
            "intent": rec.get("intent", ""), "persona": rec.get("persona", ""),
            "question": rec.get("question", ""),
            "n_gold_atoms": len(gold_atoms), "n_pred_atoms": len(pred_atoms),
            "precision": precision, "recall": recall, "f1": f1, "factscore": factscore,
            "missed_gold": missed, "unsupported_pred": unsupported}

# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")
def _std(xs):
    xs = [x for x in xs if x == x]
    return stats.pstdev(xs) if len(xs) > 1 else 0.0

def run_seed(records, source, model, seed, workers) -> dict:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(score_instance, r, source, model, seed) for r in records]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 25 == 0:
                sys.stderr.write(f"  seed {seed}: {i}/{len(records)} scored\n")
    M = lambda k: _mean([r[k] for r in rows])
    return {"seed": seed, "precision": M("precision"), "recall": M("recall"),
            "f1": M("f1"), "factscore": M("factscore"), "rows": rows}

def aggregate(seed_results) -> dict:
    def ms(metric):
        vals = [s[metric] for s in seed_results if s[metric] == s[metric]]
        return {"mean": _mean(vals), "std": _std(vals)}
    overall = {m: ms(m) for m in ("precision", "recall", "f1", "factscore")}

    def group(keyfn):
        g = defaultdict(lambda: defaultdict(list))
        for s in seed_results:
            for r in s["rows"]:
                for m in ("precision", "recall", "f1", "factscore"):
                    if r[m] == r[m]:
                        g[keyfn(r)][m].append(r[m])
        out = {}
        for k, d in g.items():
            out[k] = {m: {"mean": _mean(d[m]), "std": _std(d[m])} for m in d}
            out[k]["n"] = max(len(d[m]) for m in d)
        return out

    all_rows = [r for s in seed_results for r in s["rows"]]
    return {"overall": overall, "by_lang": group(lambda r: r["lang"]),
            "by_intent": group(lambda r: r["intent"]),
            "n_units": len(seed_results[0]["rows"]) if seed_results else 0,
            "seeds": [s["seed"] for s in seed_results], "all_rows": all_rows}

# --------------------------------------------------------------------------- #
# Markdown helpers                                                            #
# --------------------------------------------------------------------------- #
def pct(x): return "n/a" if x != x else f"{100*x:.1f}"

def cell(d, multi):
    if d["mean"] != d["mean"]:
        return "n/a"
    return f"{100*d['mean']:.1f} \u00b1 {100*d['std']:.1f}" if multi else f"{100*d['mean']:.1f}"

def md_table(headers, rows):
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    w = [max(len(str(c)) for c in col) for col in cols]
    def line(vals): return "| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(vals)) + " |"
    sep = "|" + "|".join("-" * (w[i] + 2) for i in range(len(headers))) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])

# --------------------------------------------------------------------------- #
# Data-driven report                                                          #
# --------------------------------------------------------------------------- #
def render_report(agg, model) -> str:
    multi = len(agg["seeds"]) > 1
    o = agg["overall"]
    P, R, F, FS = (o["precision"]["mean"], o["recall"]["mean"],
                   o["f1"]["mean"], o["factscore"]["mean"])
    L = []
    L.append("# Lumie RAG chatbot - re-evaluation report\n")
    L.append(f"Test units: **{agg['n_units']}** | Seeds: **{agg['seeds']}** | "
             f"Judge/decomposer: **{model}** (temp 0.2)")
    L.append("Metrics: atomic Precision/Recall/F1 (vs. the gold answer) and FactScore "
             "(grounding vs. the official corpus). All values are percentages"
             + (", mean \u00b1 std across seeds.\n" if multi else ".\n"))

    # ---- 1. Main table ----
    L.append("## 1. Overall results\n")
    rows = [["Overall", cell(o["precision"], multi), cell(o["recall"], multi),
             cell(o["f1"], multi), cell(o["factscore"], multi), str(agg["n_units"])]]
    for lang in sorted(agg["by_lang"]):
        d = agg["by_lang"][lang]
        rows.append([f"lang = {lang}", cell(d["precision"], multi), cell(d["recall"], multi),
                     cell(d["f1"], multi), cell(d["factscore"], multi), str(d["n"])])
    L.append(md_table(["Slice", "Precision", "Recall", "F1", "FactScore", "n"], rows))
    L.append("")

    # ---- 2. Verdict (computed from the numbers) ----
    L.append("## 2. What the numbers say\n")
    v = []
    v.append(f"Overall the chatbot reaches an F1 of {pct(F)} (precision {pct(P)}, "
             f"recall {pct(R)}) and a FactScore of {pct(FS)}.")
    if FS == FS and FS >= 0.95:
        v.append(f"Grounding is strong: about {pct(FS)}% of everything the bot asserts is "
                 "supported by the official documents, so hallucination is rare.")
    elif FS == FS and FS >= 0.85:
        v.append(f"Grounding is acceptable but not airtight: roughly {100*(1-FS):.0f}% of "
                 "asserted facts are not supported by the corpus (potential hallucination).")
    elif FS == FS:
        v.append(f"Grounding is weak: about {100*(1-FS):.0f}% of asserted facts are "
                 "unsupported by the corpus - hallucination is a real problem and is the "
                 "first thing to fix.")
    if P == P and FS == FS and (FS - P) >= 0.10:
        v.append(f"FactScore ({pct(FS)}) is clearly higher than precision-vs-gold ({pct(P)}). "
                 "That gap means the bot adds true, corpus-grounded details that are not in the "
                 "short canonical answer - i.e. it is verbose rather than wrong. This is a "
                 "style issue, not a factual one.")
    elif P == P and FS == FS and (P - FS) >= 0.10:
        v.append(f"Precision-vs-gold ({pct(P)}) exceeds FactScore ({pct(FS)}): the bot echoes "
                 "the canonical wording but still asserts things the corpus does not support - "
                 "look at the unsupported facts in section 4.")
    if R == R and R < 0.6:
        v.append(f"Recall is low ({pct(R)}): the bot omits a large share of the facts the gold "
                 "answer contains. In a RAG system this usually points to retrieval gaps - the "
                 "right chunk was not retrieved or not used. See the missing facts in section 5.")
    elif R == R and R < 0.8:
        v.append(f"Recall ({pct(R)}) is moderate: some canonical facts are consistently missing "
                 "(section 5).")
    if multi:
        worst_std = max(o[m]["std"] for m in o)
        if worst_std >= 0.05:
            v.append(f"Seed-to-seed variation is high (up to \u00b1{100*worst_std:.1f} points), "
                     "which suggests unstable generation - consider lowering the chatbot's "
                     "temperature or fixing a generation seed.")
        else:
            v.append("Results are stable across seeds (low standard deviation), so the scores "
                     "are reliable.")
    L.append(" ".join(v) + "\n")

    # ---- 3. Per-intent (worst first) ----
    L.append("## 3. Performance by intent (weakest first)\n")
    items = sorted(agg["by_intent"].items(), key=lambda kv: (kv[1]["f1"]["mean"]
                   if kv[1]["f1"]["mean"] == kv[1]["f1"]["mean"] else 1e9))
    rows = [[k, cell(d["precision"], multi), cell(d["recall"], multi),
             cell(d["f1"], multi), cell(d["factscore"], multi), str(d["n"])] for k, d in items]
    L.append(md_table(["Intent", "Precision", "Recall", "F1", "FactScore", "n"], rows))
    L.append("")
    if items:
        worst = [k for k, _ in items[:5]]
        best = [k for k, _ in items[-3:]][::-1]
        L.append(f"Weakest intents: {', '.join(worst)}.")
        L.append(f"Strongest intents: {', '.join(best)}.\n")

    # ---- 4. Hallucinations (unsupported predicted facts) ----
    L.append("## 4. Unsupported facts (hallucination set)\n")
    halluc = [(r["intent"], a) for r in agg["all_rows"] for a in r["unsupported_pred"]]
    total_pred = sum(r["n_pred_atoms"] for r in agg["all_rows"])
    if not halluc:
        L.append("No predicted fact failed the corpus-support check. The bot did not "
                 "hallucinate on this test set.\n")
    else:
        L.append(f"{len(halluc)} of {total_pred} predicted atomic facts "
                 f"({100*len(halluc)/max(total_pred,1):.1f}%) were not supported by the corpus. "
                 "Examples (intent -> unsupported claim):\n")
        seen, shown = set(), 0
        for intent, a in halluc:
            if a in seen:
                continue
            seen.add(a)
            L.append(f"- *{intent}*: {a}")
            shown += 1
            if shown >= 12:
                break
        L.append("")

    # ---- 5. Recall gaps (missed gold facts) ----
    L.append("## 5. Missing facts (recall gaps)\n")
    missed = [(r["intent"], a) for r in agg["all_rows"] for a in r["missed_gold"]]
    total_gold = sum(r["n_gold_atoms"] for r in agg["all_rows"])
    if not missed:
        L.append("Every gold fact was conveyed in the predictions. Recall is complete.\n")
    else:
        L.append(f"{len(missed)} of {total_gold} gold atomic facts "
                 f"({100*len(missed)/max(total_gold,1):.1f}%) were missing from the answers. "
                 "Examples (intent -> missing fact the answer should have contained):\n")
        seen, shown = set(), 0
        for intent, a in missed:
            if a in seen:
                continue
            seen.add(a)
            L.append(f"- *{intent}*: {a}")
            shown += 1
            if shown >= 12:
                break
        L.append("")

    # ---- 6. Method note ----
    L.append("## 6. How these numbers were produced\n")
    L.append("Each gold and predicted answer was decomposed into atomic facts by "
             f"{model}. Recall counts gold facts conveyed in the prediction; precision counts "
             "predicted facts inferable from the gold answer; FactScore counts predicted facts "
             "supported by the official OBG corpus. F1 is the macro-average of per-question "
             "harmonic means. PT<->EN paraphrases are treated as equivalent. Sections 2-5 are "
             "generated directly from the per-question results, not written in advance.")
    return "\n".join(L) + "\n"

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Lumie BOTTLEHUMOR-style + FactScore evaluator")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, sample this many units per seed (reproducible random splits).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--report", default="results_report.md")
    args = ap.parse_args()

    _load_cache()
    with open(args.dataset, encoding="utf-8") as fh:
        data = [json.loads(l) for l in fh if l.strip()]
    data = [r for r in data if r.get("generated_answer")]
    if not data:
        sys.exit("No rows with a non-empty 'generated_answer'. Run fill_answers.py first.")
    with open(args.corpus, encoding="utf-8") as fh:
        source = fh.read()

    import random
    seed_results = []
    for sd in args.seeds:
        recs = data
        if args.sample and args.sample < len(data):
            recs = random.Random(sd).sample(data, args.sample)
        sys.stderr.write(f"[seed {sd}] scoring {len(recs)} units...\n")
        seed_results.append(run_seed(recs, source, args.model, sd, args.workers))
        _save_cache()

    agg = aggregate(seed_results)
    report = render_report(agg, args.model)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "aggregate":
                   {k: v for k, v in agg.items() if k != "all_rows"},
                   "per_seed": seed_results}, fh, ensure_ascii=False, indent=2)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(report)
    sys.stderr.write(f"\nDone. Wrote {args.out} and {args.report}\n")
    print(report)

if __name__ == "__main__":
    main()