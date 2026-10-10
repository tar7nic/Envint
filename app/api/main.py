"""FastAPI app: /search, /answer, /health, /metrics, with per-request tracing."""
from __future__ import annotations

import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm.client import LLMClient, LLMUnavailableError
from app.llm.qa import answer_question
from app.observability import tracing
from app.retrieval.base import DependencyUnavailableError, InvalidQueryError
from app.retrieval.factory import build_retrievers
from app.schemas import AnswerResponse, SearchRequest, SearchResponse


def check_dependencies() -> dict:
    """Probe Qdrant and the LLM endpoint. Never raises."""
    s, out = get_settings(), {}
    try:
        r = httpx.get(f"{s.qdrant_url}/collections/{s.qdrant_collection}", timeout=3)
        out["qdrant"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as e:
        out["qdrant"] = f"down: {type(e).__name__}"
    try:
        r = httpx.get(f"{s.llm_base_url}/models", timeout=3)
        out["llm"] = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as e:
        out["llm"] = f"down: {type(e).__name__}"
    return out


def create_app(retrievers: dict | None = None, llm=None) -> FastAPI:
    app = FastAPI(title="Hybrid Retrieval Benchmark API")
    app.state.retrievers, app.state.llm = retrievers, llm

    def get_retrievers() -> dict:
        if app.state.retrievers is None:  # lazy: models load on first request
            app.state.retrievers = {r.strategy: r for r in build_retrievers()}
        return app.state.retrievers

    def get_llm():
        if app.state.llm is None:
            app.state.llm = LLMClient()
        return app.state.llm

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        spans, rid, t0 = tracing.start_trace(), uuid.uuid4().hex[:12], time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        tracing.metrics.record(f"route{request.url.path}", ms, None if response.status_code < 400 else f"http_{response.status_code}", {})
        response.headers["X-Request-ID"], response.headers["X-Latency-Ms"] = rid, f"{ms:.1f}"
        tracing.log_request(request.method, request.url.path, response.status_code, ms, spans, rid)
        return response

    @app.exception_handler(DependencyUnavailableError)
    async def _dep(_: Request, e: DependencyUnavailableError):
        return JSONResponse(status_code=503, content={"error": "dependency_unavailable", "detail": str(e)})

    @app.exception_handler(LLMUnavailableError)
    async def _llm(_: Request, e: LLMUnavailableError):
        return JSONResponse(status_code=503, content={"error": "llm_unavailable", "detail": str(e)})

    @app.exception_handler(InvalidQueryError)
    async def _inv(_: Request, e: InvalidQueryError):
        return JSONResponse(status_code=422, content={"error": "invalid_query", "detail": str(e)})

    @app.get("/health")
    def health():
        deps = check_dependencies()
        ok = all(v == "ok" for v in deps.values())
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", **deps})

    @app.get("/metrics")
    def get_metrics():
        return tracing.metrics.snapshot()

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        t0 = time.perf_counter()
        retriever = get_retrievers()[req.strategy]
        with tracing.span("retrieve_total", strategy=req.strategy.value):
            results = retriever.retrieve(req.query, req.k)
        return SearchResponse(query=req.query, strategy=req.strategy, results=results,
                              latency_ms=(time.perf_counter() - t0) * 1000)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: SearchRequest):
        retriever = get_retrievers()[req.strategy]
        return answer_question(req.query, retriever, get_llm(), req.k)

    return app


app = create_app()