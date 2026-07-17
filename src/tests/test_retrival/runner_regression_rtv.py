"""
runner_regression.py
====================
Executa a avaliação de retrieval sobre o dataset OBG e salva um relatório JSON.

USO:
    python runner_regression.py --dataset dataset_retrieval_filled.json

O relatório é salvo automaticamente como:
    reports/regression_YYYYMMDD_HHMMSS.json

────────────────────────────────────────────────────────────────
ANTES DE RODAR: conecte seu retriever em retrieve_top_k() abaixo.
────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import List

from evaluation_rtv import compute_metrics

# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER — SUBSTITUA O CORPO DESTA FUNÇÃO PELO SEU RETRIEVER FAISS
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_top_k(query: str, k: int = 5) -> List[str]:
    """
    Chama o retriever FAISS e retorna os top-k chunk IDs em ordem de relevância.

    Retorno esperado: lista de strings com os IDs dos chunks, ex.:
        ["regulamento-pag1", "regulamento-pag13", ...]

    ── COMO CONECTAR SEU CHATBOT ─────────────────────────────────────────────

    Opção A — se seu retriever retorna objetos Document com metadata:
        from chatbot_python.retriever import search
        docs = search(query, k=k)
        return [doc.metadata["chunk_id"] for doc in docs]

    Opção B — se retorna tuplas (chunk_id, score):
        from chatbot_python.retriever import search
        results = search(query, k=k)
        return [chunk_id for chunk_id, score in results]

    Opção C — FAISS direto com LangChain:
        from chatbot_python.vectorstore import vectorstore
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.metadata["source"] for doc in docs]

    ─────────────────────────────────────────────────────────────────────────
    """

    # ── STUB: substitua tudo abaixo pelo seu retriever ──────────────────────
    # Enquanto não conectado, retorna lista vazia → métricas zeradas.
    return []
    # ────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(dataset_path: str, output_dir: str = "reports", k: int = 5) -> str:
    """Executa o regression run completo e salva o relatório JSON."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Carrega dataset ──────────────────────────────────────────────────────
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"  OBG Retrieval Regression  |  k={k}  |  {total} queries")
    print(f"  Dataset : {dataset_path}")
    print(f"{'='*60}")

    # ── Por query ────────────────────────────────────────────────────────────
    per_query = []
    by_category = defaultdict(list)

    for i, item in enumerate(dataset, 1):
        qid      = item["id"]
        query    = item["query"]
        relevant = item["relevant_chunks"]
        category = item["original_category"]

        retrieved = retrieve_top_k(query, k=k)

        metrics = compute_metrics(retrieved, relevant, k=k)

        record = {
            "id":            qid,
            "category":      category,
            "query":         query,
            "relevant":      relevant,
            "retrieved":     retrieved,
            "metrics":       metrics,
            "expected_behavior": item.get("expected_behavior", ""),
        }
        per_query.append(record)
        by_category[category].append(metrics)

        # progresso a cada 50
        if i % 50 == 0 or i == total:
            print(f"  [{i:>3}/{total}] processadas...")

    # ── Agregados globais ────────────────────────────────────────────────────
    def mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    global_p   = mean([r["metrics"]["P@5"]  for r in per_query])
    global_r   = mean([r["metrics"]["R@5"]  for r in per_query])
    global_f1  = mean([r["metrics"]["F1@5"] for r in per_query])
    global_map = mean([r["metrics"]["AP@5"] for r in per_query])

    # ── Agregados por categoria ──────────────────────────────────────────────
    category_summary = {}
    for cat, results in sorted(by_category.items()):
        category_summary[cat] = {
            "n":     len(results),
            "P@5":   mean([r["P@5"]  for r in results]),
            "R@5":   mean([r["R@5"]  for r in results]),
            "F1@5":  mean([r["F1@5"] for r in results]),
            "MAP@5": mean([r["AP@5"] for r in results]),
        }

    # ── Monta relatório ──────────────────────────────────────────────────────
    report = {
        "meta": {
            "timestamp":    datetime.now().isoformat(),
            "dataset":      dataset_path,
            "total_queries": total,
            "k":            k,
        },
        "global": {
            "P@5":   global_p,
            "R@5":   global_r,
            "F1@5":  global_f1,
            "MAP@5": global_map,
        },
        "by_category": category_summary,
        "per_query":   per_query,
    }

    # ── Salva JSON ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"regression_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Imprime summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTADO GLOBAL  ({total} queries, k={k})")
    print(f"{'='*60}")
    print(f"  P@5    {global_p:.4f}   — precisão média nos top-5")
    print(f"  R@5    {global_r:.4f}   — recall médio nos top-5")
    print(f"  F1@5   {global_f1:.4f}   — harmônica P/R")
    print(f"  MAP@5  {global_map:.4f}   — posição importa (acadêmico)")
    print(f"\n  POR CATEGORIA:")
    print(f"  {'categoria':<35} {'n':>4}  {'P@5':>6}  {'R@5':>6}  {'F1@5':>6}  {'MAP@5':>6}")
    print(f"  {'-'*35} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cat, s in category_summary.items():
        print(f"  {cat:<35} {s['n']:>4}  {s['P@5']:>6.4f}  {s['R@5']:>6.4f}  {s['F1@5']:>6.4f}  {s['MAP@5']:>6.4f}")
    print(f"\n  Relatório salvo em: {report_path}")
    print(f"{'='*60}\n")

    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBG Retrieval Regression Runner")
    parser.add_argument(
        "--dataset",
        default="dataset_retrieval_filled.json",
        help="Caminho para o arquivo JSON do dataset (default: dataset_retrieval_filled.json)"
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Pasta onde o relatório JSON será salvo (default: reports/)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de documentos top-k avaliados (default: 5)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERRO: dataset não encontrado: {args.dataset}")
        sys.exit(1)

    run(dataset_path=args.dataset, output_dir=args.output, k=args.k)