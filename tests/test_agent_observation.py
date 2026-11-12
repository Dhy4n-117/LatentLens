"""Tests for tool observation serialization and truncation."""

from __future__ import annotations

import json

import pytest

from latentlens.agent.observation import format_observation, format_tool_response_dict
from latentlens.agent.types import ToolResult


def test_format_observation_success() -> None:
    res = ToolResult(tool_name="search_moments", output=[{"video_id": "v1"}])
    obs = format_observation(res, max_chars=500)
    parsed = json.loads(obs)
    assert parsed["tool"] == "search_moments"
    assert parsed["ok"] is True
    assert parsed["output"] == [{"video_id": "v1"}]


def test_format_observation_error() -> None:
    res = ToolResult(tool_name="search_moments", output="invalid query", is_error=True)
    obs = format_observation(res, max_chars=500)
    parsed = json.loads(obs)
    assert parsed["ok"] is False
    assert parsed["error"] == "invalid query"


def test_format_observation_truncates_long_output() -> None:
    huge = "x" * 300
    res = ToolResult(tool_name="search_moments", output={"blob": huge})
    obs = format_observation(res, max_chars=100)
    assert len(obs) == 100
    assert obs.endswith("... [truncated]")


def test_format_observation_rejects_tiny_max_chars() -> None:
    res = ToolResult(tool_name="t", output="ok")
    with pytest.raises(ValueError, match="max_chars"):
        format_observation(res, max_chars=32)


def test_format_tool_response_dict_variants() -> None:
    ok_res = ToolResult(tool_name="t", output={"count": 2})
    assert format_tool_response_dict(ok_res) == {"ok": True, "result": {"count": 2}}

    err_res = ToolResult(tool_name="t", output="boom", is_error=True)
    assert format_tool_response_dict(err_res) == {"ok": False, "error": "boom"}

    big_res = ToolResult(tool_name="t", output={"data": "y" * 200})
    trunc_dict = format_tool_response_dict(big_res, max_chars=80)
    assert trunc_dict["ok"] is True
    assert "result_truncated" in trunc_dict
