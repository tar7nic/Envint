"""Show queries where any strategy scored below NDCG 1.0.  Run: python -m scripts.show_diffs dev"""
import json
import sys

from app.config import get_settings
from benchmark.run_benchmark import load_queries

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
data = json.loads((get_settings().results_dir / f"benchmark_{split}.json").read_text(encoding="utf-8"))
queries = {q.query_id: q for q in load_queries(split)}
for qid, q in queries.items():
    if not q.answerable:
        continue
    rows = {r["strategy"]: next(x for x in r["per_query"] if x["query_id"] == qid) for r in data["results"]}
    if all(row["ndcg"] >= 0.999 for row in rows.values()):
        continue
    print(f"\n[{qid}] ({q.query_type}) {q.query}\n  expected: {q.relevant}")
    for name, row in rows.items():
        print(f"  {name:<14} ndcg={row['ndcg']:.2f}  top3={row['ranked'][:3]}")