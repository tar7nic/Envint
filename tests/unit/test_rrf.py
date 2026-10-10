import pytest
from app.retrieval.rrf import rrf_fuse
from app.schemas import RetrievalResult, Strategy


def R(cid, rank):
    return RetrievalResult(chunk_id=cid, doc_id="D", text=cid, score=99.0, rank=rank, strategy=Strategy.VECTOR)


def test_rrf_scores_match_formula():
    out = rrf_fuse([[R("A", 1), R("B", 2)], [R("B", 1), R("A", 2)]], k=60)
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert {r.chunk_id for r in out} == {"A", "B"}


def test_rrf_item_in_both_lists_beats_item_in_one():
    out = rrf_fuse([[R("A", 1), R("B", 2)], [R("B", 1), R("C", 2)]])
    assert out[0].chunk_id == "B"


def test_rrf_ignores_raw_scores():
    a = [R("A", 1)]
    a[0].score = 1000.0
    assert rrf_fuse([a])[0].score == pytest.approx(1 / 61)


def test_rrf_ties_are_deterministic():
    out = rrf_fuse([[R("B", 1)], [R("A", 1)]])
    assert [r.chunk_id for r in out] == ["A", "B"]


def test_rrf_duplicates_within_one_list_counted_once():
    out = rrf_fuse([[R("A", 1), R("A", 2)]])
    assert out[0].score == pytest.approx(1 / 61)


def test_rrf_empty_and_top_n_and_ranks():
    assert rrf_fuse([[], []]) == []
    out = rrf_fuse([[R("A", 1), R("B", 2), R("C", 3)]], top_n=2)
    assert [r.rank for r in out] == [1, 2]


def test_rrf_negative_k_rejected():
    with pytest.raises(ValueError):
        rrf_fuse([[R("A", 1)]], k=-1)
