"""Offline unit tests for GeminiAgentModel adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from latentlens.agent.gemini import AgentModelError, GeminiAgentModel
from latentlens.agent.model import Message, ToolCall
from latentlens.agent.types import ToolDefinition, ToolParameter, ToolResult
from latentlens.llm.ratelimit import RateLimiter


class _FakeModels:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.models = _FakeModels(responses)


def _unlimited() -> RateLimiter:
    return RateLimiter.per_minute(1000)


def test_gemini_agent_model_parses_tool_calls_and_replays_raw() -> None:
    raw_content = SimpleNamespace(
        role="model",
        parts=[
            SimpleNamespace(thought=True, text="internal reasoning", function_call=None),
            SimpleNamespace(
                thought=False,
                text=None,
                function_call=SimpleNamespace(
                    name="search_moments",
                    args={"query": "red mug", "top_k": 3},
                    id="call-1",
                ),
            ),
        ],
    )
    resp1 = SimpleNamespace(candidates=[SimpleNamespace(content=raw_content)])
    resp2 = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    role="model",
                    parts=[
                        SimpleNamespace(
                            thought=False,
                            text="Found at [v1 @ 00:00.0–00:04.0].",
                            function_call=None,
                        )
                    ],
                )
            )
        ]
    )

    client = _FakeClient([resp1, resp2])
    adapter = GeminiAgentModel(
        api_key="fake",
        model="gemini-2.5-flash",
        limiter=_unlimited(),
        client=client,
    )
    tool_def = ToolDefinition(
        name="search_moments",
        description="Search moments",
        parameters=(ToolParameter(name="query", type="string", description="q"),),
    )

    turn1 = adapter.generate(
        system="You are an agent.",
        messages=[Message.user("Where is the red mug?")],
        tools=[tool_def],
    )
    assert not turn1.is_final
    assert len(turn1.tool_calls) == 1
    assert turn1.tool_calls[0].name == "search_moments"
    assert turn1.tool_calls[0].arguments == {"query": "red mug", "top_k": 3}
    assert turn1.raw is raw_content

    tool_res = ToolResult(tool_name="search_moments", output=[{"video_id": "v1"}])
    history = [
        Message.user("Where is the red mug?"),
        Message.from_turn(turn1),
        Message.tool(turn1.tool_calls[0], tool_res, '{"ok": true}'),
    ]
    turn2 = adapter.generate(system="You are an agent.", messages=history, tools=[tool_def])
    assert turn2.is_final
    assert turn2.text == "Found at [v1 @ 00:00.0–00:04.0]."
    # Verify raw model Content was replayed verbatim in turn 2
    sent_contents = client.models.calls[1]["contents"]
    assert sent_contents[1] is raw_content


def test_gemini_agent_model_retries_on_429() -> None:
    err = RuntimeError("rate limit")
    err.code = 429  # type: ignore[attr-defined]
    ok_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(thought=False, text="Done.", function_call=None)],
                )
            )
        ]
    )
    sleeps: list[float] = []
    client = _FakeClient([err, ok_resp])
    adapter = GeminiAgentModel(
        api_key="fake",
        model="gemini-2.5-flash",
        limiter=_unlimited(),
        backoff_sec=1.5,
        sleep=sleeps.append,
        client=client,
    )
    turn = adapter.generate(system="sys", messages=[Message.user("hi")], tools=[])
    assert turn.text == "Done."
    assert sleeps == [1.5]


def test_gemini_agent_model_raises_on_empty_response() -> None:
    client = _FakeClient([SimpleNamespace(candidates=[], text="")])
    adapter = GeminiAgentModel(
        api_key="fake",
        model="gemini-2.5-flash",
        limiter=_unlimited(),
        client=client,
    )
    with pytest.raises(AgentModelError, match="no candidates"):
        adapter.generate(system="sys", messages=[Message.user("hi")], tools=[])


def test_gemini_agent_model_validates_init_params() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        GeminiAgentModel(
            api_key="k", model="m", limiter=_unlimited(), max_retries=-1, client=object()
        )
    with pytest.raises(ValueError, match="backoff_sec"):
        GeminiAgentModel(
            api_key="k", model="m", limiter=_unlimited(), backoff_sec=0.0, client=object()
        )
    m = GeminiAgentModel(api_key="k", model="m", limiter=_unlimited(), client=object())
    assert m.name == "gemini:m"
    # Also test fallback model message building without raw payload
    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(thought=False, text="ok", function_call=None)],
                )
            )
        ]
    )
    m2 = GeminiAgentModel(api_key="k", model="m", limiter=_unlimited(), client=_FakeClient([resp]))
    synth_msg = Message(
        role="model",
        content="thinking",
        tool_calls=(ToolCall(name="list_videos", arguments={}),),
        raw=None,
    )
    out = m2.generate(system="s", messages=[Message.user("q"), synth_msg], tools=[])
    assert out.text == "ok"
