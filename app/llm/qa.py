"""Retrieve -> prompt -> LLM -> validate."""
from __future__ import annotations

import time

from app.llm.client import LLMClient
from app.llm.guardrails import ParsedAnswer, parse_answer
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.base import Retriever
from app.observability.tracing import span
from app.schemas import AnswerResponse, Citation, RetrievalResult


def run_qa_with_context(query: str, results: list[RetrievalResult], llm: LLMClient, strategy) -> tuple[AnswerResponse, ParsedAnswer]:
    t0 = time.perf_counter()
    if not results:  # nothing retrieved: refuse without spending an LLM call
        parsed = ParsedAnswer(text="No supporting evidence was found.", refused=True)
        return AnswerResponse(query=query, strategy=strategy, answer=parsed.text, refused=True, citations=[],
                              latency_ms=(time.perf_counter() - t0) * 1000), parsed
    with span("llm") as sp:
        resp = llm.chat(SYSTEM_PROMPT, build_user_prompt(query, results))
        sp["prompt_tokens"], sp["completion_tokens"] = resp.prompt_tokens, resp.completion_tokens
    parsed = parse_answer(resp.text, {r.chunk_id for r in results})
    by_id = {r.chunk_id: r for r in results}
    citations = [
        Citation(chunk_id=c, doc_id=by_id[c].doc_id, snippet=by_id[c].text[:200], score=by_id[c].score)
        for c in parsed.cited
    ]
    return AnswerResponse(
        query=query, strategy=strategy, answer=parsed.text, refused=parsed.refused, citations=citations,
        prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
        latency_ms=(time.perf_counter() - t0) * 1000,
    ), parsed


def run_qa(query: str, retriever: Retriever, llm: LLMClient, k: int = 5) -> tuple[AnswerResponse, ParsedAnswer]:
    t0 = time.perf_counter()
    results = retriever.retrieve(query, k)
    ans, parsed = run_qa_with_context(query, results, llm, retriever.strategy)
    ans.latency_ms = (time.perf_counter() - t0) * 1000  # include retrieval time
    return ans, parsed


def answer_question(query: str, retriever: Retriever, llm: LLMClient, k: int = 5) -> AnswerResponse:
    return run_qa(query, retriever, llm, k)[0]