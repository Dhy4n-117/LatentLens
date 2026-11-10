from __future__ import annotations

from latentlens.agent.types import AgentTool, ToolDefinition, ToolParameter, ToolResult


class DummyTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dummy_tool",
            description="A test tool",
            parameters=(
                ToolParameter(name="arg1", type="string", description="first arg"),
                ToolParameter(
                    name="arg2", type="integer", description="second arg", required=False, default=1
                ),
            ),
        )

    def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(tool_name="dummy_tool", output=kwargs)


def test_tool_definition_to_function_declaration() -> None:
    tool = DummyTool()
    assert isinstance(tool, AgentTool)

    schema = tool.definition.to_function_declaration()
    assert schema["name"] == "dummy_tool"
    assert "arg1" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["arg1"]
    assert schema["parameters"]["properties"]["arg2"]["default"] == 1


def test_tool_result_creation() -> None:
    res = ToolResult(tool_name="test", output="ok", is_error=False, metadata={"count": 5})
    assert res.tool_name == "test"
    assert res.output == "ok"
    assert not res.is_error
    assert res.metadata["count"] == 5
