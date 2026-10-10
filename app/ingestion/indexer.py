"""Embeds chunks and upserts them into Qdrant. Idempotent: unchanged chunks are skipped."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.retrieval.base import DependencyUnavailableError
from app.schemas import ChunkPayload

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention


class EmbedderProtocol(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Embedder:
    """Local bge embedder (free). Model loads lazily on first use."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().embedding_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._load().encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._load().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def point_id(chunk_id: str) -> str:
    """Deterministic ID: same chunk_id -> same point, so upserts never duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@dataclass
class IndexStats:
    total: int = 0
    embedded: int = 0   # new or changed -> embedded + upserted
    skipped: int = 0    # unchanged -> not re-embedded


class Indexer:
    def __init__(self, client: QdrantClient | None = None, embedder: EmbedderProtocol | None = None):
        s = get_settings()
        self.s = s
        self.client = client or QdrantClient(url=s.qdrant_url, timeout=s.qdrant_timeout_s)
        self.embedder = embedder or Embedder()

    def ensure_collection(self) -> None:
        try:
            if not self.client.collection_exists(self.s.qdrant_collection):
                self.client.create_collection(
                    self.s.qdrant_collection,
                    vectors_config=qm.VectorParams(size=self.s.embedding_dim, distance=qm.Distance.COSINE),
                )
        except Exception as e:
            raise DependencyUnavailableError(f"Qdrant unavailable: {e}") from e

    def upsert_chunks(self, chunks: list[ChunkPayload]) -> IndexStats:
        stats = IndexStats(total=len(chunks))
        if not chunks:
            return stats
        try:
            ids = [point_id(c.chunk_id) for c in chunks]
            existing = {
                str(p.id): p.payload
                for p in self.client.retrieve(self.s.qdrant_collection, ids=ids, with_payload=True)
            }
            todo = [
                c for c, pid in zip(chunks, ids)
                if not (
                    (old := existing.get(pid))
                    and old.get("content_hash") == c.content_hash
                    and old.get("embedding_version") == c.embedding_version
                )
            ]
            stats.embedded, stats.skipped = len(todo), len(chunks) - len(todo)
            if todo:
                vectors = self.embedder.embed_documents([c.text for c in todo])
                self.client.upsert(
                    self.s.qdrant_collection,
                    points=[
                        qm.PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump())
                        for c, v in zip(todo, vectors)
                    ],
                )
        except DependencyUnavailableError:
            raise
        except Exception as e:
            raise DependencyUnavailableError(f"Qdrant unavailable: {e}") from e
        return stats