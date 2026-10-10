"""Keyword (BM25) retriever. Used on its own for tests and as one half of hybrid."""
from __future__ import annotations

import re

from qdrant_client import QdrantClient
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.retrieval.base import DependencyUnavailableError, Retriever, validate_query
from app.schemas import RetrievalResult, Strategy

# Keeps clause IDs like '4.2.1' and hyphenated IDs like 'pol-001' as single tokens.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


_STEM = PorterStemmer().stem
STOPWORDS = frozenset(
    "a an the of to in on at by for and or is are be been was were it its this that these those "
    "what which who whom how do does did can could may might shall should will would with from as "
    "into than then there their they we our us".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, stem words. Tokens containing digits (clause IDs like '4.2.1',
    'pol-001', '8') are kept verbatim so exact-ID matching still works. 'not'/'no'/'must' are
    kept on purpose: negations and obligations carry meaning in policy text."""
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if any(ch.isdigit() for ch in t):
            out.append(t)
        elif t not in STOPWORDS:
            out.append(_STEM(t))
    return out


class BM25Retriever(Retriever):
    # Not one of the 3 benchmark tiers; reuses HYBRID label only inside HybridRetriever.
    strategy = Strategy.HYBRID

    def __init__(self, payloads: list[dict]):
        """payloads: dicts with chunk_id, doc_id, clause_id, text (i.e. ChunkPayload.model_dump())."""
        self.payloads = payloads
        # index doc_id + clause_id too, so queries like 'POL-001 1.2' match exactly
        corpus = [tokenize(f"{p['doc_id']} {p.get('clause_id') or ''} {p['text']}") for p in payloads]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_qdrant(cls, client: QdrantClient | None = None) -> "BM25Retriever":
        s = get_settings()
        client = client or QdrantClient(url=s.qdrant_url, timeout=s.qdrant_timeout_s)
        try:
            payloads, offset = [], None
            while True:
                points, offset = client.scroll(
                    s.qdrant_collection, limit=256, offset=offset, with_payload=True
                )
                payloads += [p.payload for p in points]
                if offset is None:
                    break
        except Exception as e:
            raise DependencyUnavailableError(f"could not load corpus for BM25: {e}") from e
        return cls(payloads)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q = validate_query(query)
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(q))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if scores[i] <= 0 or len(results) >= k:
                break
            p = self.payloads[i]
            results.append(RetrievalResult(
                chunk_id=p["chunk_id"], doc_id=p["doc_id"], clause_id=p.get("clause_id"),
                text=p["text"], score=float(scores[i]), rank=len(results) + 1, strategy=self.strategy,
            ))
        return results