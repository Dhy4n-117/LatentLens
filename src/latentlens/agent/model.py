"""Model-agnostic conversation primitives for the agent harness.

The harness never imports a vendor SDK directly. It speaks in these types, and
an adapter (e.g. ``GeminiAgentModel``) translates them to and from a provider.
That keeps the reasoning loop fully testable offline with a scripted model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from latentlens.agent.types import ToolDefinition, ToolResult

Role = Literal["user", "model", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to invoke a harness tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One model response: optional visible text plus zero or more tool calls.

    ``raw`` carries the provider's untouched payload. Providers with reasoning
    state (e.g. Gemini thought signatures) require it to be replayed verbatim.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = None

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


@dataclass(frozen=True, slots=True)
class Message:
    """An entry in the conversation history sent to the model."""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    raw: Any = None

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def from_turn(cls, turn: ModelTurn) -> Message:
        return cls(role="model", content=turn.text, tool_calls=turn.tool_calls, raw=turn.raw)

    @classmethod
    def tool(cls, call: ToolCall, result: ToolResult, rendered: str) -> Message:
        return cls(role="tool", content=rendered, tool_call=call, tool_result=result)


@runtime_checkable
class AgentModel(Protocol):
    """A chat model capable of function calling."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelTurn: ...
