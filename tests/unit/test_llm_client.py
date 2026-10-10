from types import SimpleNamespace as NS

import openai
import pytest

from app.llm.client import LLMClient, LLMUnavailableError
from tests.conftest import rate_limit_error


def ok(text="hi"):
    return NS(choices=[NS(message=NS(content=text))], usage=NS(prompt_tokens=5, completion_tokens=2))


class FakeOpenAI:
    def __init__(self, seq):
        self.seq, self.calls = list(seq), 0
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        x = self.seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


def test_retries_429_with_exponential_backoff_then_succeeds():
    waits, fake = [], FakeOpenAI([rate_limit_error(), rate_limit_error(), ok("done")])
    r = LLMClient(client=fake, sleep=waits.append).chat("s", "u")
    assert r.text == "done" and fake.calls == 3 and waits == [1.0, 2.0] and r.prompt_tokens == 5


def test_gives_up_after_max_retries():
    fake = FakeOpenAI([rate_limit_error()] * 10)
    with pytest.raises(LLMUnavailableError):
        LLMClient(client=fake, sleep=lambda s: None).chat("s", "u")
    assert fake.calls == 4


def test_non_retryable_error_fails_fast():
    err = openai.AuthenticationError("bad key", response=__import__("httpx").Response(401, request=__import__("httpx").Request("POST", "http://x")), body=None)
    fake = FakeOpenAI([err])
    with pytest.raises(LLMUnavailableError):
        LLMClient(client=fake, sleep=lambda s: None).chat("s", "u")
    assert fake.calls == 1
