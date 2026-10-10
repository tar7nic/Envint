"""Lightweight tracing (custom middleware logs): per-request spans with latency, status and
token counts, plus in-process aggregate metrics exposed at /metrics."""
from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np

log = logging.getLogger("trace")
_current: contextvars.ContextVar[list | None] = contextvars.ContextVar("trace_spans", default=None)


class Metrics:
    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self._lat: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._count: dict[str, int] = defaultdict(int)
        self._err: dict[str, int] = defaultdict(int)
        self._err_types: dict[str, int] = defaultdict(int)
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(self, name: str, ms: float, error: str | None, attrs: dict) -> None:
        with self._lock:
            self._count[name] += 1
            self._lat[name].append(ms)
            if error:
                self._err[name] += 1
                self._err_types[error] += 1
            self.prompt_tokens += attrs.get("prompt_tokens", 0)
            self.completion_tokens += attrs.get("completion_tokens", 0)

    def snapshot(self) -> dict:
        with self._lock:
            steps = {}
            for name, n in self._count.items():
                lat = list(self._lat[name])
                steps[name] = {
                    "count": n, "errors": self._err[name], "error_rate": self._err[name] / n,
                    "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                }
            return {"steps": steps, "error_types": dict(self._err_types),
                    "tokens": {"prompt": self.prompt_tokens, "completion": self.completion_tokens}}

    def reset(self) -> None:
        self.__init__()


metrics = Metrics()


def start_trace() -> list:
    spans: list = []
    _current.set(spans)
    return spans


@contextmanager
def span(name: str, **attrs):
    """Time a step. Yields a dict you can add attributes to (e.g. token counts)."""
    t0 = time.perf_counter()
    error = None
    try:
        yield attrs
    except Exception as e:
        error = type(e).__name__
        raise
    finally:
        ms = (time.perf_counter() - t0) * 1000
        metrics.record(name, ms, error, attrs)
        spans = _current.get()
        if spans is not None:
            spans.append({"step": name, "ms": round(ms, 2), "status": "error" if error else "ok",
                          **({"error": error} if error else {}), **attrs})


def log_request(method: str, path: str, status: int, total_ms: float, spans: list, request_id: str) -> None:
    log.info(json.dumps({"request_id": request_id, "method": method, "path": path, "status": status,
                         "total_ms": round(total_ms, 2), "spans": spans}))