"""Registry for managing and safely executing harness tools."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from latentlens.agent.types import AgentTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry holding registered tools with dispatch and error containment."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        """Register a tool. Replaces any tool with the same name."""
        name = tool.definition.name
        self._tools[name] = tool
        logger.debug("Registered harness tool: %s", name)

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[AgentTool]:
        return iter(self._tools.values())

    def get_definitions(self) -> list[ToolDefinition]:
        """Return specifications for all registered tools."""
        return [t.definition for t in self._tools.values()]

    def get_function_declarations(self) -> list[dict[str, Any]]:
        """Return list of JSON function declarations for model tool configs."""
        return [t.definition.to_function_declaration() for t in self._tools.values()]

    def dispatch(self, tool_name: str, kwargs: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with exception isolation."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                output=f"Error: Tool '{tool_name}' not found in registry.",
                is_error=True,
            )

        try:
            return tool.execute(**kwargs)
        except Exception as exc:
            logger.warning("Execution error in tool '%s': %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name,
                output=f"Error executing tool '{tool_name}': {exc}",
                is_error=True,
            )
