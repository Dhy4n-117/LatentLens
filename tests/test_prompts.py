from __future__ import annotations

from latentlens.llm.prompts import QA_SYSTEM_PROMPT, build_qa_prompt, format_qa_context
from latentlens.types import ClipSpan, SearchHit


def test_format_qa_context_empty() -> None:
    assert format_qa_context([]) == "No relevant video clips found."


def test_format_qa_context_with_hits() -> None:
    span = ClipSpan("vid1", "/path/to/cooking.mp4", 10.0, 14.0)
    hit = SearchHit(rank=1, score=0.92, span=span, caption="Chef slices tomatoes.")
    context = format_qa_context([hit])

    assert "Retrieved Video Evidence:" in context
    assert "cooking.mp4" in context
    assert "Chef slices tomatoes." in context


def test_build_qa_prompt() -> None:
    span = ClipSpan("vid1", "/path/to/cooking.mp4", 0.0, 4.0)
    hit = SearchHit(rank=1, score=0.88, span=span, caption="Pan heating on stove.")
    prompt = build_qa_prompt("What is on the stove?", [hit])

    assert "What is on the stove?" in prompt
    assert "cooking.mp4" in prompt
    assert QA_SYSTEM_PROMPT is not None
