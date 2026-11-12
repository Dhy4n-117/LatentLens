from __future__ import annotations

from latentlens.agent.model import Message, ModelTurn, ToolCall
from latentlens.agent.types import ToolResult


def test_model_turn_finality() -> None:
    assert ModelTurn(text="done").is_final
    assert not ModelTurn(tool_calls=(ToolCall("search_moments", {"query": "dog"}),)).is_final


def test_message_constructors_preserve_raw_payload() -> None:
    raw = object()
    turn = ModelTurn(text="thinking", tool_calls=(ToolCall("t"),), raw=raw)

    msg = Message.from_turn(turn)
    assert msg.role == "model"
    assert msg.raw is raw  # replayed verbatim to keep provider reasoning state
    assert msg.tool_calls == turn.tool_calls


def test_tool_message_links_call_and_result() -> None:
    call = ToolCall("t", {"a": 1}, call_id="c1")
    result = ToolResult(tool_name="t", output=[1, 2])

    msg = Message.tool(call, result, rendered="[1, 2]")
    assert msg.role == "tool"
    assert msg.tool_call is call and msg.tool_result is result
    assert Message.user("hi").content == "hi"
