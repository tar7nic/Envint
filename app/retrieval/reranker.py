"""Tier 3: hybrid (BM25 + vector -> RRF) -> cross-encoder rerank."""
from __future__ import annotations

import logging
from typing import Protocol

from app.config import get_settings
from app.observability.tracing import span
from app.retrieval.base import DependencyUnavailableError, Retriever, validate_query
from app.retrieval.hybrid import HybridRetriever
from app.schemas import RetrievalResult, Strategy

log = logging.getLogger(__name__)


class ScorerProtocol(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """Local bge cross-encoder (free). Loads lazily on first use."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().reranker_model
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return [float(x) for x in self._model.predict([(query, t) for t in texts])]


class RerankRetriever(Retriever):
    strategy = Strategy.HYBRID_RERANK

    def __init__(self, hybrid: HybridRetriever, scorer: ScorerProtocol | None = None, candidates: int | None = None):
        self.hybrid = hybrid
        self.scorer = scorer or CrossEncoderScorer()
        self.n = candidates or get_settings().rerank_candidates

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q = validate_query(query)
        cands = self.hybrid.candidates(q, max(self.n, k))
        if not cands:
            return []
        try:
            with span("cross_encoder_rerank", candidates=len(cands)):
                scores = self.scorer.score(q, [c.text for c in cands])
        except Exception as e:  # reranker failure must not kill the query
            log.warning("reranker failed, falling back to RRF order: %s", e)
            return [c.model_copy(update={"strategy": self.strategy}) for c in cands[:k]]
        ranked = sorted(zip(cands, scores), key=lambda cs: (-cs[1], cs[0].rank))[:k]
        return [
            c.model_copy(update={"score": s, "rank": i, "strategy": self.strategy})
            for i, (c, s) in enumerate(ranked, start=1)
        ]