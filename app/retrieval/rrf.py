"""Custom Reciprocal Rank Fusion (no library).  score(d) = sum over rankers of 1 / (k + rank_d)."""
from __future__ import annotations

from app.schemas import RetrievalResult, Strategy


def rrf_fuse(
    rankings: list[list[RetrievalResult]],
    k: int = 60,
    top_n: int | None = None,
    strategy: Strategy = Strategy.HYBRID,
) -> list[RetrievalResult]:
    """Fuse several ranked lists by rank position (raw scores are ignored, so BM25 and
    cosine scales never need normalising). Ties break on best single rank, then chunk_id,
    so output is deterministic."""
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, RetrievalResult] = {}

    for ranking in rankings:
        seen_in_list: set[str] = set()
        for pos, r in enumerate(ranking, start=1):
            if r.chunk_id in seen_in_list:  # ignore duplicates within one list
                continue
            seen_in_list.add(r.chunk_id)
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + pos)
            best_rank[r.chunk_id] = min(best_rank.get(r.chunk_id, pos), pos)
            first_seen.setdefault(r.chunk_id, r)

    order = sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))
    if top_n is not None:
        order = order[:top_n]
    return [
        first_seen[c].model_copy(update={"score": scores[c], "rank": i, "strategy": strategy})
        for i, c in enumerate(order, start=1)
    ]