"""Agent harness module for autonomous video QA and tool dispatching."""

from latentlens.agent.citations import (
    ParsedCitation,
    extract_citations,
    parse_timestamp,
    verify_citations,
)
from latentlens.agent.config import HarnessConfig
from latentlens.agent.gemini import AgentModelError, GeminiAgentModel
from latentlens.agent.harness import VideoAgentHarness
from latentlens.agent.model import AgentModel, Message, ModelTurn, Role, ToolCall
from latentlens.agent.observation import format_observation, format_tool_response_dict
from latentlens.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    BUDGET_SYNTHESIS_INSTRUCTION,
    build_synthesis_prompt,
)
from latentlens.agent.registry import ToolRegistry
from latentlens.agent.tools import (
    GetAdjacentClipsTool,
    InspectClipTool,
    ListVideosTool,
    SearchMomentsTool,
    build_default_registry,
)
from latentlens.agent.trace import AgentStep, AgentTrace, TerminationReason
from latentlens.agent.types import AgentTool, ToolDefinition, ToolParameter, ToolResult

__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "BUDGET_SYNTHESIS_INSTRUCTION",
    "AgentModel",
    "AgentModelError",
    "AgentStep",
    "AgentTool",
    "AgentTrace",
    "GeminiAgentModel",
    "GetAdjacentClipsTool",
    "HarnessConfig",
    "InspectClipTool",
    "ListVideosTool",
    "Message",
    "ModelTurn",
    "ParsedCitation",
    "Role",
    "SearchMomentsTool",
    "TerminationReason",
    "ToolCall",
    "ToolDefinition",
    "ToolParameter",
    "ToolRegistry",
    "ToolResult",
    "VideoAgentHarness",
    "build_default_registry",
    "build_synthesis_prompt",
    "extract_citations",
    "format_observation",
    "format_tool_response_dict",
    "parse_timestamp",
    "verify_citations",
]
