"""Tests for AgentStep and AgentTrace telemetry dataclasses."""

from __future__ import annotations

from latentlens.agent.model import ToolCall
from latentlens.agent.trace import AgentStep, AgentTrace
from latentlens.agent.types import ToolResult


def test_agent_step_properties_and_serialization() -> None:
    tc = ToolCall(name="search_moments", arguments={"query": "door"}, call_id="c1")
    tr = ToolResult(tool_name="search_moments", output=[{"video_id": "v1"}])
    step = AgentStep(
        step_index=0,
        thought="Let me search for the door",
        text="",
        tool_calls=(tc,),
        tool_results=(tr,),
        elapsed_ms=12.345,
    )
    assert step.num_tool_calls == 1
    data = step.to_dict()
    assert data["step_index"] == 0
    assert data["thought"] == "Let me search for the door"
    assert data["tool_calls"][0]["name"] == "search_moments"
    assert data["tool_results"][0]["is_error"] is False
    assert data["elapsed_ms"] == 12.35


def test_agent_trace_aggregation() -> None:
    s0 = AgentStep(
        step_index=0,
        tool_calls=(
            ToolCall(name="search_moments", arguments={"query": "a"}),
            ToolCall(name="get_adjacent_clips", arguments={"video_id": "v1"}),
        ),
        elapsed_ms=10.0,
    )
    s1 = AgentStep(step_index=1, text="Final answer", elapsed_ms=5.0)
    trace = AgentTrace(
        query="What happened?",
        steps=[s0, s1],
        termination_reason="final_answer",
        total_elapsed_ms=15.0,
    )
    assert trace.total_steps == 2
    assert trace.total_tool_calls == 2
    serialized = trace.to_dict()
    assert serialized["query"] == "What happened?"
    assert serialized["total_steps"] == 2
    assert serialized["total_tool_calls"] == 2
    assert len(serialized["steps"]) == 2
