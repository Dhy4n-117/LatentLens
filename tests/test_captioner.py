from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from latentlens.llm import Captioner, CaptionError, GeminiCaptioner, RateLimiter, ResponseCache

IMAGES = [b"\xff\xd8frame-1", b"\xff\xd8frame-2"]


class ApiError(Exception):
    """Mimics google.genai.errors.APIError, which exposes an HTTP ``code``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeModels:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


@pytest.fixture
def make_captioner(
    tmp_path: Path,
) -> Callable[..., tuple[GeminiCaptioner, FakeModels, list[float]]]:
    def factory(*responses: str | Exception) -> tuple[GeminiCaptioner, FakeModels, list[float]]:
        models = FakeModels(list(responses))
        sleeps: list[float] = []
        captioner = GeminiCaptioner(
            api_key="unused",
            model="gemini-test",
            limiter=RateLimiter(1000, 60.0),
            cache=ResponseCache(tmp_path / "cache"),
            max_retries=2,
            backoff_sec=1.0,
            sleep=sleeps.append,
            client=SimpleNamespace(models=models),
        )
        return captioner, models, sleeps

    return factory


def test_satisfies_protocol(make_captioner: Callable[..., Any]) -> None:
    captioner, _, _ = make_captioner()
    assert isinstance(captioner, Captioner)
    assert captioner.name == "gemini:gemini-test"


def test_sends_images_then_prompt_and_normalises_whitespace(
    make_captioner: Callable[..., Any],
) -> None:
    captioner, models, _ = make_captioner("  A dog\n catches   a frisbee. ")

    assert captioner.caption(IMAGES) == "A dog catches a frisbee."
    (call,) = models.calls
    assert call["model"] == "gemini-test"
    assert len(call["contents"]) == 3 and call["contents"][-1] == captioner.prompt


def test_second_call_is_served_from_cache(make_captioner: Callable[..., Any]) -> None:
    captioner, models, _ = make_captioner("A cat sleeps.")

    first = captioner.caption(IMAGES)
    second = captioner.caption(IMAGES)

    assert first == second == "A cat sleeps."
    assert len(models.calls) == 1  # no quota spent on the repeat


def test_retries_transient_errors_with_exponential_backoff(
    make_captioner: Callable[..., Any],
) -> None:
    captioner, models, sleeps = make_captioner(ApiError(429), ApiError(503), "Recovered.")

    assert captioner.caption(IMAGES) == "Recovered."
    assert len(models.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries(make_captioner: Callable[..., Any]) -> None:
    captioner, models, _ = make_captioner(ApiError(429), ApiError(429), ApiError(429))

    with pytest.raises(CaptionError):
        captioner.caption(IMAGES)
    assert len(models.calls) == 3


def test_does_not_retry_client_errors(make_captioner: Callable[..., Any]) -> None:
    captioner, models, sleeps = make_captioner(ApiError(400))

    with pytest.raises(CaptionError):
        captioner.caption(IMAGES)
    assert len(models.calls) == 1 and sleeps == []


def test_empty_response_is_an_error_and_not_cached(make_captioner: Callable[..., Any]) -> None:
    captioner, _, _ = make_captioner("", "Second try works.")

    with pytest.raises(CaptionError, match="empty"):
        captioner.caption(IMAGES)
    assert captioner.caption(IMAGES) == "Second try works."


def test_requires_images(make_captioner: Callable[..., Any]) -> None:
    captioner, _, _ = make_captioner()
    with pytest.raises(ValueError):
        captioner.caption([])
