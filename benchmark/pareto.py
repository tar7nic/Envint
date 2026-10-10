"""Pareto plot: p50 latency vs NDCG@5.  Run: python -m benchmark.pareto data/results/benchmark_eval.json [more.json ...]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def label(res: dict, reranker: str) -> str:
    return f"{res['strategy']} ({reranker.split('/')[-1]})" if res["strategy"] == "hybrid_rerank" else res["strategy"]


def pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Non-dominated points: no other point is both faster and at least as accurate."""
    return [p for p in points if not any(q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]) for q in points)]


def main(paths: list[str]) -> None:
    points = []
    for path in paths:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for r in d["results"]:
            points.append((r["latency_p50_ms"], r["ndcg@5"], f"{label(r, d['reranker'])} [{d['split']}]"))
    front = sorted(pareto_front(points))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for x, y, name in points:
        ax.scatter(x, y, s=70, color="tab:green" if (x, y, name) in front else "tab:gray")
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], "--", color="tab:green", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency (ms, log scale)")
    ax.set_ylabel("NDCG@5")
    ax.set_title("Latency vs relevance (green = Pareto-efficient)")
    ax.grid(alpha=0.3)
    out = Path(paths[0]).with_name("pareto.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for x, y, name in front:
        print(f"  Pareto-efficient: {name}  p50={x:.0f}ms  ndcg={y:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/results/benchmark_eval.json"])