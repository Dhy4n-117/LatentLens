from __future__ import annotations

from latentlens.agent.registry import ToolRegistry
from latentlens.agent.types import ToolDefinition, ToolParameter, ToolResult


class AddTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="add",
            description="Add two integers",
            parameters=(
                ToolParameter(name="a", type="integer", description="number a"),
                ToolParameter(name="b", type="integer", description="number b"),
            ),
        )

    def execute(self, a: int, b: int) -> ToolResult:
        return ToolResult(tool_name="add", output=a + b)


class FailingTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="fail", description="Fails intentionally")

    def execute(self) -> ToolResult:
        raise RuntimeError("Disk read error")


def test_registry_register_and_dispatch() -> None:
    reg = ToolRegistry()
    reg.register(AddTool())

    assert len(reg) == 1
    assert "add" in reg
    assert len(reg.get_definitions()) == 1
    assert len(reg.get_function_declarations()) == 1

    res = reg.dispatch("add", {"a": 2, "b": 3})
    assert not res.is_error
    assert res.output == 5


def test_registry_dispatch_missing_tool() -> None:
    reg = ToolRegistry()
    res = reg.dispatch("nonexistent", {})
    assert res.is_error
    assert "not found" in str(res.output)


def test_registry_dispatch_exception_containment() -> None:
    reg = ToolRegistry()
    reg.register(FailingTool())

    res = reg.dispatch("fail", {})
    assert res.is_error
    assert "Disk read error" in str(res.output)
