"""Common interface + exceptions shared by all three retrieval strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas import RetrievalResult, Strategy


class RetrievalError(Exception):
    """Base class for retrieval failures."""


class InvalidQueryError(RetrievalError):
    """Empty / malformed query."""


class DependencyUnavailableError(RetrievalError):
    """Qdrant, embedding model, or reranker is unreachable / failed to load."""


def validate_query(query: str, max_len: int = 1000) -> str:
    if not isinstance(query, str):
        raise InvalidQueryError("query must be a string")
    q = query.strip()
    if not q:
        raise InvalidQueryError("query must not be empty")
    if len(q) > max_len:
        raise InvalidQueryError(f"query exceeds {max_len} characters")
    return q


class Retriever(ABC):
    """All strategies implement this so the API, benchmark and tests treat them identically."""

    strategy: Strategy

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Return up to k results, best first, with 1-based ranks. Empty list if nothing matches."""
        raise NotImplementedError