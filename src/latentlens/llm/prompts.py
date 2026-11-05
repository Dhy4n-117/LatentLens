"""Prompt engineering templates for retrieval-augmented grounded video QA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from latentlens.types import SearchHit

QA_SYSTEM_PROMPT = (
    "You are an AI video assistant analyzing retrieved video clips to answer user questions. "
    "Base your answer ONLY on the visual evidence and clip descriptions provided below. "
    "Do NOT fabricate information or speculate. "
    "When making statements, cite the relevant clip as: [video_filename @ start-end]. "
    "If the provided video context does not contain enough evidence to answer, state clearly: "
    "'The retrieved video clips do not provide enough evidence to answer this question.'"
)


def format_qa_context(hits: Sequence[SearchHit]) -> str:
    """Format retrieved SearchHit clips into a structured context string for Gemini."""
    if not hits:
        return "No relevant video clips found."

    lines: list[str] = ["Retrieved Video Evidence:"]
    for hit in hits:
        filename = Path(hit.span.video_path).name
        desc = hit.caption if hit.caption else "Visual match (no caption available)"
        lines.append(
            f"- Clip {hit.rank}: [{filename} @ {hit.span.label()}] (score: {hit.score:.3f})\n"
            f"  Description: {desc}"
        )
    return "\n".join(lines)


def build_qa_prompt(query: str, hits: Sequence[SearchHit]) -> str:
    """Construct prompt combining retrieved clips context with user question."""
    context = format_qa_context(hits)
    return (
        f"{context}\n\n"
        f"User Question: {query}\n\n"
        f"Answer the question directly, citing the supporting timestamp ranges."
    )
