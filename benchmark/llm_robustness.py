"""LLM robustness across the 3 retrieval strategies.

Run: python -m benchmark.llm_robustness --split eval
Needs Qdrant + Ollama running (LLM_BASE_URL=http://localhost:11434/v1 when run on your machine).
Measures: correct refusal on unsupported queries, false refusals on answerable ones, citation
validity, and whether answers cite a truly relevant chunk. Also runs two fixed-context cases
(conflicting clauses, irrelevant context) that do not depend on the retrieval strategy.
"""
from __future__ import annotations

import argparse
import json

from app.config import get_settings
from app.llm.client import LLMClient, LLMUnavailableError
from app.llm.qa import run_qa, run_qa_with_context
from app.schemas import RetrievalResult, Strategy
from benchmark.metrics import percentile
from benchmark.run_benchmark import build_retrievers, load_queries


def _chunk(cid: str, text: str, rank: int) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, doc_id=cid.split("#")[0], text=text, score=1.0, rank=rank, strategy=Strategy.HYBRID)


def evaluate_llm(retriever, llm, queries, k: int = 5) -> dict:
    rows, lat = [], []
    for q in queries:
        try:
            ans, parsed = run_qa(q.query, retriever, llm, k)
            err = None
        except LLMUnavailableError as e:
            rows.append({"query_id": q.query_id, "answerable": q.answerable, "error": str(e)}); continue
        lat.append(ans.latency_ms)
        rows.append({
            "query_id": q.query_id, "query": q.query, "answerable": q.answerable, "refused": ans.refused,
            "cited": parsed.cited, "invalid_citations": parsed.invalid, "answer": ans.answer,
            "cites_relevant": any(c in q.relevant for c in parsed.cited),
            "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
        })
    ok = [r for r in rows if "error" not in r]
    unans = [r for r in ok if not r["answerable"]]
    ans_ = [r for r in ok if r["answerable"]]
    rate = lambda xs, f: (sum(1 for r in xs if f(r)) / len(xs)) if xs else None
    return {
        "n": len(rows), "errors": len(rows) - len(ok),
        "correct_refusal_rate": rate(unans, lambda r: r["refused"]),
        "false_refusal_rate": rate(ans_, lambda r: r["refused"]),
        "answers_cite_relevant_rate": rate(ans_, lambda r: not r["refused"] and r["cites_relevant"]),
        "invalid_citation_rate": rate(ok, lambda r: bool(r["invalid_citations"])),
        "uncited_answer_rate": rate([r for r in ok if not r["refused"]], lambda r: not r["cited"]),
        "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95),
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in ok) / len(ok) if ok else 0,
        "avg_completion_tokens": sum(r["completion_tokens"] for r in ok) / len(ok) if ok else 0,
        "rows": rows,
    }


def fixed_cases(llm) -> dict:
    conflict_ctx = [_chunk("POL-A#1.1", "Nurses must have at least 10 hours of rest between shifts.", 1),
                    _chunk("POL-B#1.1", "Nurses must have at least 8 hours of rest between shifts.", 2)]
    ans, p = run_qa_with_context("How many hours of rest do nurses need between shifts?", conflict_ctx, llm, Strategy.HYBRID)
    conflict = {"answer": ans.answer, "mentions_both_values": "10" in ans.answer and "8" in ans.answer,
                "cites_both": set(p.cited) == {"POL-A#1.1", "POL-B#1.1"}}
    irrelevant_ctx = [_chunk("POL-X#1.1", "General visiting hours are 10:00 to 20:00.", 1),
                      _chunk("POL-Y#1.1", "Equipment maintenance records must be kept for 5 years.", 2)]
    ans, p = run_qa_with_context("What is the maximum storage temperature for vaccines?", irrelevant_ctx, llm, Strategy.HYBRID)
    return {"conflicting_clauses": conflict, "irrelevant_context": {"answer": ans.answer, "refused_correctly": ans.refused}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], default="eval")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    s, llm = get_settings(), LLMClient()
    queries = load_queries(args.split)
    out = {"split": args.split, "llm_model": s.llm_model, "strategies": {}}
    for r in build_retrievers():
        out["strategies"][r.strategy.value] = evaluate_llm(r, llm, queries)
    out["fixed_cases"] = fixed_cases(llm)
    s.results_dir.mkdir(parents=True, exist_ok=True)
    path = s.results_dir / f"llm_robustness_{args.split}{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    f = lambda v: "  n/a" if v is None else f"{v:5.2f}"
    print(f"\nllm={s.llm_model} split={args.split}")
    print(f"{'strategy':<15}{'refuse✓':>8}{'falseRef':>9}{'citeRel':>8}{'badCite':>8}{'uncited':>8}{'p50 ms':>9}{'tok in':>8}{'tok out':>8}")
    for name, m in out["strategies"].items():
        print(f"{name:<15}{f(m['correct_refusal_rate']):>8}{f(m['false_refusal_rate']):>9}{f(m['answers_cite_relevant_rate']):>8}"
              f"{f(m['invalid_citation_rate']):>8}{f(m['uncited_answer_rate']):>8}{m['latency_p50_ms']:>9.0f}"
              f"{m['avg_prompt_tokens']:>8.0f}{m['avg_completion_tokens']:>8.0f}")
    print("\nfixed cases:", json.dumps(out["fixed_cases"], indent=2))
    print(f"\nraw output: {path}")


if __name__ == "__main__":
    main()