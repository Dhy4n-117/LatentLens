"""Multi-turn conversation memory for video question answering."""

from __future__ import annotations

from dataclasses import dataclass

from latentlens.types import ClipSpan


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """A completed user-question / agent-answer exchange with cited spans."""

    query: str
    rewritten_query: str
    answer: str
    citations: tuple[ClipSpan, ...] = ()


class ConversationMemory:
    """Bounded rolling memory of prior QA exchanges for multi-turn sessions."""

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """Return all stored turns in chronological order."""
        return tuple(self._turns)

    @property
    def last_turn(self) -> ConversationTurn | None:
        """Return the most recent conversation turn, if any."""
        return self._turns[-1] if self._turns else None

    def add(
        self,
        query: str,
        answer: str,
        *,
        rewritten_query: str | None = None,
        citations: tuple[ClipSpan, ...] = (),
    ) -> ConversationTurn:
        """Append a turn and evict the oldest if ``max_turns`` is exceeded."""
        turn = ConversationTurn(
            query=query.strip(),
            rewritten_query=(rewritten_query or query).strip(),
            answer=answer.strip(),
            citations=tuple(citations),
        )
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        return turn

    def clear(self) -> None:
        """Remove all stored conversation turns."""
        self._turns.clear()

    def format_context(self, max_recent: int = 3) -> str:
        """Render the most recent turns into a compact text block for prompting."""
        if not self._turns or max_recent < 1:
            return ""
        recent = self._turns[-max_recent:]
        lines: list[str] = ["Conversation History:"]
        for idx, t in enumerate(recent, start=1):
            cites = ", ".join(f"[{c.video_id} @ {c.label()}]" for c in t.citations)
            cite_suffix = f" (Cited: {cites})" if cites else ""
            lines.append(f"{idx}. User: {t.query}\n   Assistant: {t.answer}{cite_suffix}")
        return "\n".join(lines)


__all__ = [
    "ConversationMemory",
    "ConversationTurn",
]
