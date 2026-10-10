import pytest
from app.retrieval.base import DependencyUnavailableError, InvalidQueryError, Retriever
from app.retrieval.bm25 import BM25Retriever, tokenize
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import RerankRetriever
from app.schemas import RetrievalResult, Strategy


def R(cid, rank):
    return RetrievalResult(chunk_id=cid, doc_id="D", text=cid, score=1, rank=rank, strategy=Strategy.VECTOR)


class Fake(Retriever):
    strategy = Strategy.VECTOR

    def __init__(self, res=None, fail=False):
        self.res, self.fail = res or [], fail

    def retrieve(self, q, k=5):
        if self.fail:
            raise DependencyUnavailableError("down")
        return self.res[:k]


PAYLOADS = [
    {"chunk_id": "P#1.1", "doc_id": "P", "clause_id": "1.1", "text": "Verbal orders are not permitted for chemotherapy."},
    {"chunk_id": "P#1.2", "doc_id": "P", "clause_id": "1.2", "text": "Overtime may be approved by the department head."},
    {"chunk_id": "P#1.3", "doc_id": "P", "clause_id": "1.3", "text": "Defibrillators must be tested daily."},
    {"chunk_id": "P#1.4", "doc_id": "P", "clause_id": "1.4", "text": "Visitors must wear masks during outbreaks."},
]  # 4 docs: BM25Okapi gives a term IDF 0 when it appears in half the corpus, so tiny corpora need filler


def test_tokenizer_stems_but_keeps_clause_ids():
    assert tokenize("approved verbally 4.2.1 POL-001") == ["approv", "verbal", "4.2.1", "pol-001"]


def test_bm25_matches_across_word_forms():
    top = BM25Retriever(PAYLOADS).retrieve("who can approve overtime", 1)
    assert top[0].chunk_id == "P#1.2"


def test_bm25_unsupported_query_returns_empty_and_blank_rejected():
    b = BM25Retriever(PAYLOADS)
    assert b.retrieve("quantum physics") == []
    with pytest.raises(InvalidQueryError):
        b.retrieve("   ")
    assert BM25Retriever([]).retrieve("anything") == []


def test_hybrid_degrades_when_one_retriever_down_and_fails_when_both_down():
    h = HybridRetriever(Fake(fail=True), Fake([R("A", 1)]))
    assert [r.chunk_id for r in h.retrieve("x", 3)] == ["A"]
    with pytest.raises(DependencyUnavailableError):
        HybridRetriever(Fake(fail=True), Fake(fail=True)).retrieve("x")


def test_reranker_failure_falls_back_to_rrf_order():
    class Boom:
        def score(self, q, t):
            raise RuntimeError("model missing")

    h = HybridRetriever(Fake([R("A", 1), R("B", 2)]), Fake([R("A", 1)]))
    out = RerankRetriever(h, scorer=Boom()).retrieve("x", 2)
    assert out[0].chunk_id == "A" and out[0].strategy == Strategy.HYBRID_RERANK