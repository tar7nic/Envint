"""Quick manual check of all 3 tiers.  Run: python -m scripts.try_retrieval "your query" """
import sys

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankRetriever
from app.retrieval.vector import VectorRetriever

query = " ".join(sys.argv[1:]) or "What is the maximum storage temperature for vaccines?"
vec, bm25 = VectorRetriever(), BM25Retriever.from_qdrant()
hybrid = HybridRetriever(vec, bm25)
for name, r in [("1. VECTOR", vec), ("2. HYBRID (RRF)", hybrid), ("3. HYBRID + RERANK", RerankRetriever(hybrid))]:
    print(f"\n== {name} ==")
    for x in r.retrieve(query, k=3):
        print(f"{x.rank}. [{x.score:.4f}] {x.chunk_id}  {x.text[:70]}")