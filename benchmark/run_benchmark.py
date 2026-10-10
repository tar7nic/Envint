"""Benchmark the 3 retrieval strategies.

Run:  python -m benchmark.run_benchmark --split dev      (tune on this)
      python -m benchmark.run_benchmark --split eval     (final numbers only)
Writes data/results/benchmark_<split>.json and prints a comparison table.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import psutil

from app.config import get_settings
from app.retrieval.base import Retriever
from app.retrieval.factory import build_retrievers
from app.schemas import EvalQuery
from benchmark.metrics import ndcg_at_k, percentile, precision_at_k, recall_at_k, reciprocal_rank


def load_queries(split: str) -> list[EvalQuery]:
    path = get_settings().queries_dir / f"{split}.jsonl"
    return [EvalQuery.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(retriever: Retriever, queries: list[EvalQuery], k: int = 5) -> dict:
    """Run every query through one retriever. Quality metrics use answerable queries only;
    latency uses all queries. The first query is timed separately as cold start."""
    latencies, per_query = [], []
    by_type: dict[str, list[dict]] = defaultdict(list)
    cold_start_ms = None

    for q in queries:
        t0 = time.perf_counter()
        results = retriever.retrieve(q.query, k=k)
        ms = (time.perf_counter() - t0) * 1000
        if cold_start_ms is None:
            cold_start_ms = ms          # includes lazy model loading
        else:
            latencies.append(ms)
        ranked = [r.chunk_id for r in results]
        row = {"query_id": q.query_id, "query_type": q.query_type, "latency_ms": ms, "ranked": ranked}
        if q.answerable:
            row.update(
                precision=precision_at_k(ranked, q.relevant, k),
                recall=recall_at_k(ranked, q.relevant, k),
                rr=reciprocal_rank(ranked, q.relevant),
                ndcg=ndcg_at_k(ranked, q.relevant, k),
            )
            by_type[q.query_type].append(row)
        per_query.append(row)

    scored = [r for r in per_query if "ndcg" in r]
    mean = lambda key, rows: sum(r[key] for r in rows) / len(rows) if rows else 0.0
    return {
        "strategy": retriever.strategy.value,
        "k": k,
        "n_queries": len(queries),
        "n_scored": len(scored),
        f"precision@{k}": mean("precision", scored),
        f"recall@{k}": mean("recall", scored),
        "mrr": mean("rr", scored),
        "ndcg@5": mean("ndcg", scored) if k == 5 else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "cold_start_ms": cold_start_ms,
        "rss_mb": psutil.Process().memory_info().rss / 1e6,
        "by_query_type": {t: {"ndcg": mean("ndcg", rows), "mrr": mean("rr", rows), "n": len(rows)} for t, rows in by_type.items()},
        "per_query": per_query,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. large")
    args = ap.parse_args()

    s = get_settings()
    queries = load_queries(args.split)
    results = [evaluate(r, queries, args.k) for r in build_retrievers()]

    s.results_dir.mkdir(parents=True, exist_ok=True)
    out = s.results_dir / f"benchmark_{args.split}{'_' + args.tag if args.tag else ''}.json"
    out.write_text(json.dumps({"split": args.split, "reranker": s.reranker_model,
                               "embedding_model": s.embedding_model, "results": results}, indent=2), encoding="utf-8")

    k = args.k
    print(f"\nsplit={args.split}  reranker={s.reranker_model}  scored={results[0]['n_scored']}/{results[0]['n_queries']}")
    print(f"{'strategy':<15}{'P@'+str(k):>8}{'R@'+str(k):>8}{'MRR':>8}{'NDCG@5':>8}{'p50 ms':>9}{'p95 ms':>9}{'cold ms':>9}{'RSS MB':>9}")
    for r in results:
        print(f"{r['strategy']:<15}{r[f'precision@{k}']:>8.3f}{r[f'recall@{k}']:>8.3f}{r['mrr']:>8.3f}"
              f"{(r['ndcg@5'] or 0):>8.3f}{r['latency_p50_ms']:>9.1f}{r['latency_p95_ms']:>9.1f}"
              f"{r['cold_start_ms']:>9.0f}{r['rss_mb']:>9.0f}")
    print(f"\nraw output: {out}")


if __name__ == "__main__":
    main()