"""Execution telemetry dataclasses for the multi-turn agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from latentlens.agent.model import ToolCall
from latentlens.agent.types import ToolResult

TerminationReason = Literal[
    "final_answer",
    "max_steps_reached",
    "error",
]


@dataclass(frozen=True, slots=True)
class AgentStep:
    """Record of a single reasoning/action turn in the agent harness."""

    step_index: int
    thought: str = ""
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    elapsed_ms: float = 0.0

    @property
    def num_tool_calls(self) -> int:
        """Number of tool invocations requested during this step."""
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Serialize step telemetry into a JSON-friendly dictionary."""
        return {
            "step_index": self.step_index,
            "thought": self.thought,
            "text": self.text,
            "tool_calls": [
                {"name": tc.name, "arguments": dict(tc.arguments), "call_id": tc.call_id}
                for tc in self.tool_calls
            ],
            "tool_results": [
                {
                    "tool_name": tr.tool_name,
                    "is_error": tr.is_error,
                    "output": tr.output,
                }
                for tr in self.tool_results
            ],
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass(slots=True)
class AgentTrace:
    """Full trajectory of an agent harness execution for a single user query."""

    query: str
    steps: list[AgentStep] = field(default_factory=list)
    termination_reason: TerminationReason = "final_answer"
    total_elapsed_ms: float = 0.0

    @property
    def total_steps(self) -> int:
        """Total number of reasoning turns executed."""
        return len(self.steps)

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool invocations across all steps."""
        return sum(s.num_tool_calls for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete execution trace to a dictionary."""
        return {
            "query": self.query,
            "termination_reason": self.termination_reason,
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "steps": [s.to_dict() for s in self.steps],
        }


__all__ = [
    "AgentStep",
    "AgentTrace",
    "TerminationReason",
]
