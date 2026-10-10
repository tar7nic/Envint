import pytest
from benchmark.metrics import ndcg_at_k, percentile, precision_at_k, recall_at_k, reciprocal_rank

REL = {"A": 2, "B": 1}


def test_precision_and_recall():
    assert precision_at_k(["A", "X", "B"], REL, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["A", "X", "B"], REL, 3) == 1.0
    assert recall_at_k(["X", "Y"], REL, 5) == 0.0


def test_precision_rejects_bad_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], REL, 0)


def test_reciprocal_rank():
    assert reciprocal_rank(["X", "B"], REL) == 0.5
    assert reciprocal_rank(["X"], REL) == 0.0


def test_ndcg_perfect_and_swapped():
    assert ndcg_at_k(["A", "B"], REL, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["B", "A"], REL, 5) < 1.0


def test_ndcg_with_no_relevant_is_zero_not_error():
    assert ndcg_at_k(["A"], {}, 5) == 0.0


def test_percentile_empty_and_values():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
