"""
evaluation.py
=============
Funções puras de métricas de recuperação.
Todas operam sobre os top-k documentos retornados pelo FAISS.
"""

from typing import List


def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """P@k = hits nos top-k / k"""
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """R@k = hits nos top-k / total de relevantes"""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / len(relevant_set)


def f1_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """F1@k = média harmônica entre P@k e R@k"""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """
    AP@k = (1 / |relevantes|) * sum_{i=1}^{k} [ P@i * rel(i) ]
    Documento relevante na posição 1 vale mais que na posição 5.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(top_k, start=1):
        if doc in relevant_set:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant_set)


def compute_metrics(retrieved: List[str], relevant: List[str], k: int = 5) -> dict:
    """Calcula todas as métricas para uma única query."""
    return {
        "P@5":  round(precision_at_k(retrieved, relevant, k), 4),
        "R@5":  round(recall_at_k(retrieved, relevant, k), 4),
        "F1@5": round(f1_at_k(retrieved, relevant, k), 4),
        "AP@5": round(average_precision_at_k(retrieved, relevant, k), 4),
    }