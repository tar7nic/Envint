"""Retrieval metrics, written from scratch.

Relevance: `relevant` maps chunk_id -> grade (1 or 2). Precision/Recall/MRR treat any grade > 0 as
relevant (binary); NDCG uses the grade itself as the gain (linear gain).
"""
from __future__ import annotations

import math

import numpy as np


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for c in ranked[:k] if relevant.get(c, 0) > 0) / k


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    total = sum(1 for g in relevant.values() if g > 0)
    if total == 0:
        return 0.0
    return sum(1 for c in ranked[:k] if relevant.get(c, 0) > 0) / total


def reciprocal_rank(ranked: list[str], relevant: dict[str, int]) -> float:
    for i, c in enumerate(ranked, start=1):
        if relevant.get(c, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg(grades: list[int]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    actual = [relevant.get(c, 0) for c in ranked[:k]]
    return dcg(actual) / dcg(ideal)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0