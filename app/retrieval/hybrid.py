"""Tier 2: hybrid = BM25 + vector, fused with RRF. Degrades to one retriever if the other fails."""
from __future__ import annotations

import logging

from app.config import get_settings
from app.observability.tracing import span
from app.retrieval.base import DependencyUnavailableError, Retriever, validate_query
from app.retrieval.rrf import rrf_fuse
from app.schemas import RetrievalResult, Strategy

log = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    strategy = Strategy.HYBRID

    def __init__(self, vector: Retriever, bm25: Retriever, pool: int | None = None, rrf_k: int | None = None):
        s = get_settings()
        self.vector, self.bm25 = vector, bm25
        self.pool = pool or s.candidate_pool
        self.rrf_k = rrf_k if rrf_k is not None else s.rrf_k

    def candidates(self, query: str, n: int) -> list[RetrievalResult]:
        """Fused top-n (also used as the input to the reranker)."""
        q = validate_query(query)
        rankings, errors = [], []
        for r in (self.vector, self.bm25):
            try:
                with span("retrieve." + type(r).__name__):
                    rankings.append(r.retrieve(q, self.pool))
            except DependencyUnavailableError as e:
                log.warning("%s failed, continuing without it: %s", type(r).__name__, e)
                errors.append(e)
        if len(errors) == 2:
            raise DependencyUnavailableError("both retrievers failed")
        with span("rrf_fuse"):
            return rrf_fuse(rankings, k=self.rrf_k, top_n=n, strategy=self.strategy)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return self.candidates(query, k)