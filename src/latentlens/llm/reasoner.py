"""Text-based grounded QA reasoner backed by Gemini with caching and rate limiting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from latentlens.llm.cache import ResponseCache
from latentlens.llm.prompts import QA_SYSTEM_PROMPT
from latentlens.llm.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class QAReasonerError(RuntimeError):
    """Raised when the QA reasoner fails to generate a grounded response."""


@runtime_checkable
class QAReasoner(Protocol):
    """Protocol mapping a formatted prompt and system instruction to an answer string."""

    @property
    def name(self) -> str: ...

    def generate(self, prompt: str, *, system: str = QA_SYSTEM_PROMPT) -> str: ...


class GeminiQAReasoner:
    """Gemini text reasoner hardened for free-tier rate limits and response caching."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        limiter: RateLimiter,
        cache: ResponseCache | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 512,
        max_retries: int = 4,
        backoff_sec: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_sec <= 0:
            raise ValueError("backoff_sec must be positive")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._limiter = limiter
        self._cache = cache
        self._max_retries = max_retries
        self._backoff_sec = backoff_sec
        self._sleep = sleep
        self._client = client if client is not None else _make_client(api_key)

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def generate(self, prompt: str, *, system: str = QA_SYSTEM_PROMPT) -> str:
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("prompt must not be empty")

        cache_key: str | None = None
        if self._cache is not None:
            cache_key = ResponseCache.make_key(self.model, system, cleaned)
            if (cached := self._cache.get(cache_key)) is not None:
                return cached

        answer = self._generate_with_retry(cleaned, system=system)
        if self._cache is not None and cache_key is not None:
            self._cache.set(cache_key, answer)
        return answer

    def _generate_with_retry(self, prompt: str, *, system: str) -> str:
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                return self._call_gemini(prompt, system=system)
            except QAReasonerError:
                raise
            except Exception as exc:
                status = getattr(exc, "code", None)
                if status not in RETRYABLE_STATUS or attempt == self._max_retries:
                    raise QAReasonerError(f"Gemini QA request failed: {exc}") from exc
                delay = self._backoff_sec * 2**attempt
                logger.warning("Gemini QA returned %s; retrying in %.0fs", status, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _call_gemini(self, prompt: str, *, system: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise QAReasonerError("Gemini returned an empty QA response")
        return text


def _make_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)


__all__ = [
    "GeminiQAReasoner",
    "QAReasoner",
    "QAReasonerError",
]
