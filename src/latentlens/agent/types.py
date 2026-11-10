"""Domain contracts and definitions for the agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Description of a parameter expected by an agent tool."""

    name: str
    type: str  # "string", "integer", "number", "boolean"
    description: str
    required: bool = True
    default: Any = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Specification of an agent tool compatible with function-calling schemas."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def to_function_declaration(self) -> dict[str, Any]:
        """Convert tool specification to standard JSON schema representation."""
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        if required:
            schema["parameters"]["required"] = required
        return schema


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured response from a tool execution."""

    tool_name: str
    output: Any
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentTool(Protocol):
    """Protocol satisfied by any callable harness tool."""

    @property
    def definition(self) -> ToolDefinition: ...

    def execute(self, **kwargs: Any) -> ToolResult: ...
