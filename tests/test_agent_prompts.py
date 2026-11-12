"""Tests for agent system and synthesis prompts."""

from __future__ import annotations

from latentlens.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    BUDGET_SYNTHESIS_INSTRUCTION,
    build_synthesis_prompt,
)


def test_agent_system_prompt_mentions_core_tools_and_citations() -> None:
    assert "search_moments" in AGENT_SYSTEM_PROMPT
    assert "get_adjacent_clips" in AGENT_SYSTEM_PROMPT
    assert "inspect_clip" in AGENT_SYSTEM_PROMPT
    assert "list_videos" in AGENT_SYSTEM_PROMPT
    assert "MM:SS.s" in AGENT_SYSTEM_PROMPT


def test_build_synthesis_prompt_includes_query_and_instruction() -> None:
    out = build_synthesis_prompt("  Where did I leave my mug?  ")
    assert BUDGET_SYNTHESIS_INSTRUCTION in out
    assert "Original question: Where did I leave my mug?" in out
