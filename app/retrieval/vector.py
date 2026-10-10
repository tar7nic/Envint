"""Tier 1: vector-only (semantic) retrieval from Qdrant."""
from __future__ import annotations

from qdrant_client import QdrantClient

from app.config import get_settings
from app.observability.tracing import span
from app.ingestion.indexer import Embedder, EmbedderProtocol
from app.retrieval.base import DependencyUnavailableError, Retriever, validate_query
from app.schemas import RetrievalResult, Strategy


class VectorRetriever(Retriever):
    strategy = Strategy.VECTOR

    def __init__(self, client: QdrantClient | None = None, embedder: EmbedderProtocol | None = None):
        self.s = get_settings()
        self.client = client or QdrantClient(url=self.s.qdrant_url, timeout=self.s.qdrant_timeout_s)
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q = validate_query(query)
        try:
            with span("embed_query"):
                vec = self.embedder.embed_query(q)
            with span("qdrant_search"):
                hits = self.client.query_points(
                    self.s.qdrant_collection, query=vec, limit=k, with_payload=True
                ).points
        except Exception as e:
            raise DependencyUnavailableError(f"vector search failed: {e}") from e
        return [
            RetrievalResult(
                chunk_id=h.payload["chunk_id"], doc_id=h.payload["doc_id"],
                clause_id=h.payload.get("clause_id"), text=h.payload["text"],
                score=float(h.score), rank=i, strategy=self.strategy,
            )
            for i, h in enumerate(hits, start=1)
        ]