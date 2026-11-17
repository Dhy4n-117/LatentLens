"""Tests for GeminiQAReasoner with caching, rate limiting, and retry backoff."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from latentlens.llm.cache import ResponseCache
from latentlens.llm.ratelimit import RateLimiter
from latentlens.llm.reasoner import GeminiQAReasoner, QAReasonerError


class _FakeModels:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: int = 0

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = _FakeModels(responses)


def test_reasoner_caches_and_returns_response(tmp_path: Path) -> None:
    client = _FakeClient([SimpleNamespace(text="The mug is on the table [v1 @ 00:00.0–00:04.0].")])
    cache = ResponseCache(tmp_path / "cache")
    reasoner = GeminiQAReasoner(
        api_key="fake",
        model="gemini-2.5-flash",
        limiter=RateLimiter.per_minute(100),
        cache=cache,
        client=client,
    )
    assert reasoner.name == "gemini:gemini-2.5-flash"

    out1 = reasoner.generate("Where is the mug?")
    out2 = reasoner.generate("Where is the mug?")
    assert out1 == out2
    assert client.models.calls == 1


def test_reasoner_retries_on_429_and_fails_on_empty() -> None:
    err = RuntimeError("quota")
    err.code = 429  # type: ignore[attr-defined]
    sleeps: list[float] = []
    client = _FakeClient([err, SimpleNamespace(text="Grounded answer.")])
    reasoner = GeminiQAReasoner(
        api_key="fake",
        model="gemini-2.5-flash",
        limiter=RateLimiter.per_minute(100),
        backoff_sec=1.0,
        sleep=sleeps.append,
        client=client,
    )
    assert reasoner.generate("Question?") == "Grounded answer."
    assert sleeps == [1.0]

    empty_reasoner = GeminiQAReasoner(
        api_key="fake",
        model="gemini-2.5-flash",
        limiter=RateLimiter.per_minute(100),
        client=_FakeClient([SimpleNamespace(text="   ")]),
    )
    with pytest.raises(QAReasonerError, match="empty"):
        empty_reasoner.generate("Question?")


def test_reasoner_validation() -> None:
    limiter = RateLimiter.per_minute(100)
    with pytest.raises(ValueError, match="max_retries"):
        GeminiQAReasoner(api_key="k", model="m", limiter=limiter, max_retries=-1, client=object())
    with pytest.raises(ValueError, match="backoff_sec"):
        GeminiQAReasoner(api_key="k", model="m", limiter=limiter, backoff_sec=0.0, client=object())

    r = GeminiQAReasoner(api_key="k", model="m", limiter=limiter, client=object())
    with pytest.raises(ValueError, match="empty"):
        r.generate("   ")
