"""Pydantic schemas: Qdrant payload, API I/O, and benchmark query format."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Strategy(str, Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


# --------------------------------------------------------------------------- #
# Vector store payload (explicit schema, chunk metadata, embedding versioning)
# --------------------------------------------------------------------------- #
class ChunkPayload(BaseModel):
    """Exactly what is stored next to each vector in Qdrant."""

    chunk_id: str = Field(..., description="Stable ID, e.g. 'POL-007#4.2.1'")
    doc_id: str
    clause_id: str | None = Field(None, description="Fine-grained clause number, e.g. '4.2.1'")
    section: str | None = None
    text: str = Field(..., min_length=1)
    source_file: str
    doc_version: str = "1"
    content_hash: str = Field(..., description="SHA-256 of normalized text; detects unchanged chunks")
    embedding_model: str
    embedding_version: str


# --------------------------------------------------------------------------- #
# Retrieval results
# --------------------------------------------------------------------------- #
class RetrievalResult(BaseModel):
    chunk_id: str
    doc_id: str
    clause_id: str | None = None
    text: str
    score: float
    rank: int = Field(..., ge=1)
    strategy: Strategy


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    strategy: Strategy = Strategy.HYBRID_RERANK
    k: int = Field(5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace")
        return v


class SearchResponse(BaseModel):
    query: str
    strategy: Strategy
    results: list[RetrievalResult]
    latency_ms: float


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    snippet: str
    score: float


class AnswerResponse(BaseModel):
    query: str
    strategy: Strategy
    answer: str
    refused: bool = Field(..., description="True if the model declined due to missing evidence")
    citations: list[Citation]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float


# --------------------------------------------------------------------------- #
# Benchmark dataset format (one JSON object per line in data/queries/*.jsonl)
# --------------------------------------------------------------------------- #
QueryType = Literal[
    "exact_clause",      # "What does clause 4.2.1 say?"
    "hard_negative",     # keyword match, opposite meaning
    "numeric_threshold", # 24h vs 48h, 5% vs 15%
    "semantic",          # paraphrase with little keyword overlap
    "unanswerable",      # no evidence in corpus -> must refuse
]


class EvalQuery(BaseModel):
    query_id: str
    query: str
    split: Literal["dev", "eval"]
    query_type: QueryType
    # chunk_id -> graded relevance (2 = exact answer, 1 = partially relevant)
    relevant: dict[str, int] = Field(default_factory=dict)
    answerable: bool = True

    @field_validator("relevant")
    @classmethod
    def grades_valid(cls, v: dict[str, int]) -> dict[str, int]:
        if any(g not in (1, 2) for g in v.values()):
            raise ValueError("relevance grades must be 1 or 2")
        return v