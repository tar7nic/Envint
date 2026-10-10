"""Builds the three retrieval strategies (shared by the API and the benchmark)."""
from app.retrieval.base import Retriever


def build_retrievers() -> list[Retriever]:
    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.reranker import RerankRetriever
    from app.retrieval.vector import VectorRetriever

    vec, bm25 = VectorRetriever(), BM25Retriever.from_qdrant()
    hybrid = HybridRetriever(vec, bm25)
    return [vec, hybrid, RerankRetriever(hybrid)]