"""Shared fixtures: in-memory Qdrant, deterministic fake embedder/scorer/LLM (no downloads, no network)."""
import math
import re
import zlib
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import openai
import pytest
from qdrant_client import QdrantClient

from app.ingestion.chunker import load_corpus
from app.ingestion.indexer import Indexer
from app.llm.client import LLMResponse
from app.retrieval.bm25 import BM25Retriever, tokenize
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankRetriever
from app.retrieval.vector import VectorRetriever
from app.schemas import Strategy

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, similar texts -> similar vectors."""
    calls = 0

    def _vec(self, text):
        v = [0.0] * 384
        for t in tokenize(text):
            v[zlib.crc32(t.encode()) % 384] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeScorer:
    def score(self, query, texts):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(t)))) for t in texts]


class FakeLLM:
    """Cites the first document it is given; records every prompt it receives."""
    def __init__(self):
        self.calls, self.systems, self.users = 0, [], []

    def chat(self, system, user):
        self.calls += 1
        self.systems.append(system)
        self.users.append(user)
        ids = re.findall(r'document id="([^"]+)"', user)
        return LLMResponse(f"Answer [{ids[0]}]" if ids else "INSUFFICIENT_EVIDENCE", 100, 10, 1.0)


def rate_limit_error():
    resp = httpx.Response(429, request=httpx.Request("POST", "http://llm"))
    return openai.RateLimitError("rate limited", response=resp, body=None)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def indexed(embedder):
    client = QdrantClient(":memory:")
    ix = Indexer(client=client, embedder=embedder)
    ix.ensure_collection()
    chunks, failed = load_corpus(CORPUS)
    assert not failed
    ix.upsert_chunks(chunks)
    return NS(client=client, indexer=ix, chunks=chunks, embedder=embedder)


@pytest.fixture
def retrievers(indexed):
    vec = VectorRetriever(client=indexed.client, embedder=indexed.embedder)
    bm25 = BM25Retriever.from_qdrant(indexed.client)
    hybrid = HybridRetriever(vec, bm25)
    return {Strategy.VECTOR: vec, Strategy.HYBRID: hybrid,
            Strategy.HYBRID_RERANK: RerankRetriever(hybrid, scorer=FakeScorer())}