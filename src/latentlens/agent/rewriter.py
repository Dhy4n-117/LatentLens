"""Contextual query rewriting for multi-turn video question answering."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from latentlens.agent.memory import ConversationMemory
from latentlens.llm.reasoner import QAReasoner

_FOLLOWUP_PRONOUNS = re.compile(
    r"\b(it|its|they|them|their|that|those|there|afterward|beforehand|next|then)\b",
    re.IGNORECASE,
)

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's follow-up question into a standalone, self-contained video search "
    "question using the provided conversation history. Return ONLY the rewritten question."
)


@runtime_checkable
class QueryRewriter(Protocol):
    """Protocol for rewriting follow-up queries using conversation history."""

    def rewrite(self, query: str, memory: ConversationMemory) -> str: ...


class HeuristicQueryRewriter:
    """Zero-latency offline query rewriter that attaches prior context to follow-ups."""

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be empty")

        last = memory.last_turn
        if last is None:
            return cleaned

        is_short_followup = len(cleaned.split()) <= 5
        has_pronoun = bool(_FOLLOWUP_PRONOUNS.search(cleaned))
        if not (is_short_followup or has_pronoun):
            return cleaned

        cite_hint = ""
        if last.citations:
            c0 = last.citations[0]
            cite_hint = f" in {c0.video_id} around {c0.label()}"
        return f"{cleaned} (context: {last.rewritten_query}{cite_hint})"


class LLMQueryRewriter:
    """LLM-backed query rewriter that falls back to ``HeuristicQueryRewriter`` when no history."""

    def __init__(self, reasoner: QAReasoner) -> None:
        self._reasoner = reasoner
        self._fallback = HeuristicQueryRewriter()

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        if len(memory) == 0:
            return cleaned

        history_block = memory.format_context(max_recent=3)
        prompt = f"{history_block}\n\nFollow-up question: {cleaned}\nStandalone question:"
        try:
            rewritten = self._reasoner.generate(prompt, system=REWRITE_SYSTEM_PROMPT).strip()
            return rewritten or cleaned
        except Exception:
            return self._fallback.rewrite(cleaned, memory)


__all__ = [
    "REWRITE_SYSTEM_PROMPT",
    "HeuristicQueryRewriter",
    "LLMQueryRewriter",
    "QueryRewriter",
]
