import json
import os
from collections import defaultdict
from typing import List, Dict, Any


# =========================
# Constantes de score
# =========================

SCORE_LABELS = {
    0: "PASS (correct / grounded)",
    1: "Weak / ambiguous",
    2: "Overconfident without evidence",
    3: "Hallucination / fabrication"
}


# =========================
# Geração de relatório
# =========================

def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Gera um relatório quantitativo de hallucination.

    Espera que cada item em `results` contenha:
    - id
    - category
    - prompt
    - response
    - score (0–3)
    - reason
    """

    total = len(results)

    score_counts = defaultdict(int)
    category_stats = defaultdict(lambda: {
        "total": 0,
        "scores": defaultdict(int)
    })

    hallucination_count = 0
    severe_count = 0

    for r in results:
        score = r["score"]
        category = r.get("category", "uncategorized")

        score_counts[score] += 1
        category_stats[category]["total"] += 1
        category_stats[category]["scores"][score] += 1

        if score >= 2:
            hallucination_count += 1
        if score == 3:
            severe_count += 1

    # Métricas globais
    summary = {
        "total_tests": total,
        "score_distribution": {
            SCORE_LABELS[k]: v for k, v in score_counts.items()
        },
        "hallucination_rate": round(hallucination_count / total, 4),
        "severe_hallucination_rate": round(severe_count / total, 4),
        "total_hallucinations": hallucination_count,
        "severe_hallucinations": severe_count
    }

    # Métricas por categoria
    by_category = {}

    for category, data in category_stats.items():
        total_cat = data["total"]
        scores = data["scores"]

        halluc_cat = sum(
            v for k, v in scores.items() if k >= 2
        )

        by_category[category] = {
            "total_tests": total_cat,
            "hallucination_rate": round(halluc_cat / total_cat, 4),
            "score_distribution": {
                SCORE_LABELS[k]: scores.get(k, 0)
                for k in SCORE_LABELS
            }
        }

    return {
        "summary": summary,
        "by_category": by_category,
        "evaluation_scale": SCORE_LABELS,
        "notes": {
            "score_0": "Resposta correta ou abstenção fundamentada no edital/regulamento da OBG",
            "score_1": "Incerteza fraca sem grounding explícito",
            "score_2": "Resposta especulativa ou inferência sem evidência",
            "score_3": "Hallucination: informação inventada, falsa ou insegura"
        }
    }


# =========================
# Salvamento dos relatórios
# =========================

def save_reports(
    raw_results: List[Dict[str, Any]],
    report: Dict[str, Any],
    output_dir: str = "reports"
) -> None:
    """
    Salva:
    - resultados brutos
    - relatório agregado
    """

    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, "hallucination_results_raw.json")
    report_path = os.path.join(output_dir, "hallucination_report_summary.json")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Raw results saved to: {raw_path}")
    print(f"[OK] Summary report saved to: {report_path}")
