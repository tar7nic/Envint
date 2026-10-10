"""LLM client for any OpenAI-compatible endpoint (Ollama / Groq / Gemini) with retry + backoff."""
from __future__ import annotations

import time
from dataclasses import dataclass

import openai
from openai import OpenAI
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


class LLMUnavailableError(Exception):
    """LLM could not answer after retries (outage, rate limit, or non-retryable error)."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient:
    def __init__(self, client: OpenAI | None = None, sleep=time.sleep):
        self.s = get_settings()
        self.client = client or OpenAI(
            base_url=self.s.llm_base_url, api_key=self.s.llm_api_key,
            timeout=self.s.llm_timeout_s, max_retries=0,  # we do our own retries below
        )
        self._sleep = sleep

    def _call(self, system: str, user: str):
        return self.client.chat.completions.create(
            model=self.s.llm_model, temperature=self.s.llm_temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )

    def chat(self, system: str, user: str) -> LLMResponse:
        retrying = Retrying(
            retry=retry_if_exception_type(RETRYABLE),
            wait=wait_exponential(multiplier=self.s.llm_backoff_base_s, max=30),
            stop=stop_after_attempt(self.s.llm_max_retries),
            sleep=self._sleep,
        )
        t0 = time.perf_counter()
        try:
            resp = retrying(self._call, system, user)
        except RetryError as e:
            raise LLMUnavailableError(f"LLM failed after {self.s.llm_max_retries} attempts: {e.last_attempt.exception()}") from e
        except Exception as e:  # auth errors, bad requests: retrying won't help
            raise LLMUnavailableError(f"LLM call failed: {e}") from e
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None) or (len(system) + len(user)) // 4,
            completion_tokens=getattr(usage, "completion_tokens", None) or len(text) // 4,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )