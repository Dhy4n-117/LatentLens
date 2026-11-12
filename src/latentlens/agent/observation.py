"""Tool observation serialization and truncation for the agent harness."""

from __future__ import annotations

import json
from typing import Any

from latentlens.agent.types import ToolResult

_TRUNCATION_SUFFIX = "... [truncated]"


def format_observation(result: ToolResult, max_chars: int = 4000) -> str:
    """Serialize a ``ToolResult`` into a bounded text observation for the model."""
    if max_chars < 64:
        raise ValueError("max_chars must be >= 64")

    payload: dict[str, Any] = {
        "tool": result.tool_name,
        "ok": not result.is_error,
    }
    if not result.is_error:
        payload["output"] = result.output
    else:
        payload["error"] = str(result.output) if result.output is not None else "Unknown tool error"

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) <= max_chars:
        return serialized

    cutoff = max_chars - len(_TRUNCATION_SUFFIX)
    return serialized[:cutoff] + _TRUNCATION_SUFFIX


def format_tool_response_dict(result: ToolResult, max_chars: int = 4000) -> dict[str, Any]:
    """Build a structured dictionary suitable for ``Part.from_function_response``."""
    text = format_observation(result, max_chars=max_chars)
    if not result.is_error and not text.endswith(_TRUNCATION_SUFFIX):
        return {"ok": True, "result": result.output}
    if result.is_error:
        return {
            "ok": False,
            "error": str(result.output) if result.output is not None else "Unknown tool error",
        }
    return {"ok": True, "result_truncated": text}


__all__ = [
    "format_observation",
    "format_tool_response_dict",
]
