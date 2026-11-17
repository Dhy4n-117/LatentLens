"""Tests for ConversationTurn and ConversationMemory."""

from __future__ import annotations

import pytest

from latentlens.agent.memory import ConversationMemory
from latentlens.types import ClipSpan


def test_conversation_memory_eviction_and_formatting() -> None:
    mem = ConversationMemory(max_turns=2)
    assert len(mem) == 0
    assert mem.last_turn is None
    assert mem.format_context() == ""

    s1 = ClipSpan("kitchen", "kitchen.mp4", 0.0, 4.0)
    mem.add("Where is the mug?", "On the counter.", citations=(s1,))
    mem.add("What color is it?", "Red.", rewritten_query="What color is the mug?")
    mem.add("Who picked it up?", "A person in a jacket.")

    assert len(mem) == 2
    assert mem.turns[0].query == "What color is it?"
    assert mem.turns[0].rewritten_query == "What color is the mug?"
    assert mem.last_turn is not None
    assert mem.last_turn.query == "Who picked it up?"

    ctx = mem.format_context(max_recent=2)
    assert "Conversation History:" in ctx
    assert "What color is it?" in ctx
    assert "Who picked it up?" in ctx

    mem.clear()
    assert len(mem) == 0


def test_conversation_memory_rejects_invalid_max_turns() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        ConversationMemory(max_turns=0)
