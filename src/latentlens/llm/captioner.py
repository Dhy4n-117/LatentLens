"""Turn clip keyframes into short factual captions with Gemini (free tier)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from latentlens.llm.cache import ResponseCache
from latentlens.llm.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "These images are keyframes, in temporal order, from one short video clip. "
    "Describe what happens in one or two plain sentences: the main subjects, their "
    "actions, notable objects and the setting. State only what is visible; do not "
    "speculate, mention the images themselves, or use bullet points."
)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class CaptionError(RuntimeError):
    """Raised when a caption could not be produced."""


@runtime_checkable
class Captioner(Protocol):
    """Maps an ordered list of JPEG keyframes to a caption."""

    @property
    def name(self) -> str: ...

    def caption(self, images: Sequence[bytes]) -> str: ...


class GeminiCaptioner:
    """Gemini captioner hardened for the free tier.

    Every request goes through the cache first (zero quota on a hit), then the
    rate limiter, and transient failures (429 / 5xx) are retried with
    exponential backoff. The SDK client is injectable for offline tests.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        limiter: RateLimiter,
        cache: ResponseCache,
        prompt: str = CAPTION_PROMPT,
        max_retries: int = 4,
        backoff_sec: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self._limiter = limiter
        self._cache = cache
        self._max_retries = max_retries
        self._backoff_sec = backoff_sec
        self._sleep = sleep
        self._client = client if client is not None else _make_client(api_key)

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def caption(self, images: Sequence[bytes]) -> str:
        if not images:
            raise ValueError("At least one image is required")

        key = ResponseCache.make_key(self.model, self.prompt, *images)
        if (cached := self._cache.get(key)) is not None:
            return cached

        text = self._generate_with_retry(images)
        self._cache.set(key, text)
        return text

    # -- internals ---------------------------------------------------------

    def _generate_with_retry(self, images: Sequence[bytes]) -> str:
        for attempt in range(self._max_retries + 1):
            self._limiter.acquire()
            try:
                return self._generate(images)
            except CaptionError:
                raise
            except Exception as exc:
                status = getattr(exc, "code", None)
                if status not in RETRYABLE_STATUS or attempt == self._max_retries:
                    raise CaptionError(f"Gemini request failed: {exc}") from exc
                delay = self._backoff_sec * 2**attempt
                logger.warning("Gemini returned %s; retrying in %.0fs", status, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _generate(self, images: Sequence[bytes]) -> str:
        from google.genai import types

        parts = [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images]
        response = self._client.models.generate_content(
            model=self.model,
            contents=[*parts, self.prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=256,
                # Captioning needs no reasoning; skipping it saves latency and quota.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise CaptionError("Gemini returned an empty caption (possibly safety-filtered)")
        return " ".join(text.split())


def _make_client(api_key: str) -> Any:
    from google import genai

    return genai.Client(api_key=api_key)
