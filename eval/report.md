# Re-evaluation of the Lumie RAG Chatbot — Methodology

This document describes *how* the Lumie chatbot is evaluated. The **results and the
analysis are not written here** — they are generated automatically from the run, in
`results_report.md`, so that every number and every conclusion is computed from the
data rather than fixed in advance. Run `lumie_eval.py` to produce them.

## 1. Objective

Measure the factual quality of Lumie, the RAG assistant for the Brazilian Geography
Olympiad (OBG), against the official document corpus, using two complementary
decomposition-based metrics: an atomic Precision/Recall/F1 metric adapted from Hwang
et al. (BOTTLEHUMOR, Findings of ACL 2025), and FactScore (Min et al., 2023).

## 2. Gold answer and knowledge source

The gold answers are the canonical OBG responses in `gold_corpus.jsonl`, grounded in
the 10ª OBG 2025 documents (Regulamento, support and password-procedure notes, iGeo
edital, question guidelines, ODS themes). The full factual content of those documents
forms the knowledge source (`knowledge_source.txt`) used by FactScore.

## 3. Metrics

The two metrics share an atomic-decomposition front end but verify against different
references, which is why both are reported.

**Atomic Precision / Recall / F1 (vs. the gold answer).** An LLM decomposes the gold
answer *y* into atomic facts and the generated answer *x* into atomic facts. Then:

- Recall = share of gold facts conveyed in *x* (coverage of the canonical answer).
- Precision = share of *x*'s facts inferable from *y* (no off-canonical additions).
- F1 = per-question harmonic mean; reported as the macro-average.

**FactScore (vs. the corpus).** Decomposes only *x* and checks each fact against the
whole official corpus: FactScore = share of *x*'s facts supported by the documents.
This is the faithfulness / anti-hallucination measure.

**Why both.** Precision is verified against a short canonical answer; FactScore against
the entire corpus. A true, corpus-grounded detail that is absent from the terse gold
answer *lowers precision but not FactScore* — that pattern means the bot is verbose,
not wrong. An ungrounded claim lowers FactScore even if it mimics the gold wording.
Reporting both separates "answered the canonical question" from "stayed grounded."

## 4. Protocol

- **Test set:** 200 persona-varied questions (`testset_questions.jsonl`), built from 42
  intents and phrased as professor, aluno, coordenador de escola, and coordenador de
  olimpíada. Each carries its intent's gold answer; `generated_answer` is produced by
  running the question through Lumie (`fill_answers.py`).
- **Seeds:** 3 runs (default 13, 21, 42); results reported as mean ± std.
- **Judge / decomposer:** OpenAI `gpt-4o-mini` (the OpenAI counterpart to the paper's
  Gemini-1.5-Flash evaluator), temperature 0.2, for decomposition and verification.
- **Cross-lingual matching:** PT↔EN paraphrases are scored as equivalent, for Lumie's
  multilingual setting.
- **Recommended reliability check:** as in the source paper (≈77% LLM–human agreement,
  κ≈0.54), have two annotators label ~130 atom-level judgements to report agreement and
  κ for this corpus.

## 5. What the generated report contains

`results_report.md` (written by `lumie_eval.py`) contains, all computed from the run:

1. an overall table (Precision, Recall, F1, FactScore, with PT/EN breakdown);
2. a computed verdict — the diagnosis (grounded-but-verbose vs. hallucination vs.
   recall/retrieval gaps vs. unstable generation) is selected from the actual scores;
3. a per-intent table ranked weakest-first;
4. the hallucination set — the exact predicted facts that failed corpus support;
5. the recall gaps — the exact gold facts the answers omitted.

Example table layout (numbers illustrative only):

| Slice     | Precision  | Recall     | F1         | FactScore  | n   |
|-----------|------------|------------|------------|------------|-----|
| Overall   | 81.0 ± 0.3 | 67.8 ± 0.3 | 73.4 ± 0.2 | 85.7 ± 0.5 | 200 |
| lang = pt | 81.3 ± 0.4 | 68.0 ± 0.5 | 73.6 ± 0.3 | 85.9 ± 0.6 | 158 |
| lang = en | 80.3 ± 0.6 | 67.3 ± 0.7 | 72.9 ± 0.4 | 85.4 ± 0.8 |  42 |

## 6. Threats to validity

- **LLM-as-judge bias** — decomposition granularity and Yes/No calibration depend on
  gpt-4o-mini; the human-agreement check (§4) bounds this.
- **Single gold per intent** — multiple phrasings are valid; FactScore-vs-corpus
  mitigates this for precision.
- **Edition/date conflicts** — the corpus is anchored on the 10ª OBG 2025 documents;
  do not merge it with 11ª OBG 2026 material, or FactScore will penalize version
  differences as if they were hallucinations.

## 7. Reproducibility

```
export OPENAI_API_KEY=...
python build_testset_obg.py --target 200
python fill_answers.py --dataset testset_questions.jsonl      # wire your chatbot here
python lumie_eval.py --dataset testset_questions.jsonl --corpus knowledge_source.txt \
    --seeds 13 21 42 --model gpt-4o-mini --out results.json --report results_report.md
```

Judge responses are cached (`.lumie_cache.json`) for deterministic, low-cost re-runs.

## 8. References

- E. Hwang, P. West, V. Shwartz. *BOTTLEHUMOR.* Findings of ACL 2025.
- S. Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long
  Form Text Generation.* EMNLP 2023.