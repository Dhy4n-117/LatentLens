"""Tests for HeuristicQueryRewriter and LLMQueryRewriter."""

from __future__ import annotations

import pytest

from latentlens.agent.memory import ConversationMemory
from latentlens.agent.rewriter import HeuristicQueryRewriter, LLMQueryRewriter
from latentlens.types import ClipSpan


class _FakeReasoner:
    def __init__(self, output: str | Exception) -> None:
        self.output = output

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, prompt: str, *, system: str = "") -> str:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def test_heuristic_query_rewriter_handles_first_and_followup_turns() -> None:
    mem = ConversationMemory()
    rewriter = HeuristicQueryRewriter()

    q1 = "Where did the person put the red kettle?"
    assert rewriter.rewrite(q1, mem) == q1

    span = ClipSpan("kitchen", "kitchen.mp4", 4.0, 8.0)
    mem.add(q1, "On the stove.", citations=(span,))

    q2 = "Did they turn it on afterward?"
    rewritten = rewriter.rewrite(q2, mem)
    assert "context: Where did the person put the red kettle?" in rewritten
    assert "kitchen around 00:04.0–00:08.0" in rewritten

    # Standalone long question without pronouns is left untouched
    q3 = "How many cars are parked outside the garage in the morning?"
    assert rewriter.rewrite(q3, mem) == q3


def test_llm_query_rewriter_and_fallback() -> None:
    mem = ConversationMemory()
    rewriter = LLMQueryRewriter(_FakeReasoner("Standalone rewritten query?"))

    # Empty memory bypasses LLM call
    assert rewriter.rewrite("First question?", mem) == "First question?"

    mem.add("First question?", "First answer.")
    assert rewriter.rewrite("What about it?", mem) == "Standalone rewritten query?"

    # Fallback when reasoner raises
    failing = LLMQueryRewriter(_FakeReasoner(RuntimeError("network error")))
    fallback_out = failing.rewrite("What about it?", mem)
    assert "context: First question?" in fallback_out

    with pytest.raises(ValueError, match="empty"):
        rewriter.rewrite("   ", mem)
    with pytest.raises(ValueError, match="empty"):
        HeuristicQueryRewriter().rewrite("   ", mem)
