"""Single-pass retrieval-augmented Grounded QA pipeline with weak-evidence refusal."""

from __future__ import annotations

from latentlens.agent.citations import verify_citations
from latentlens.llm.prompts import QA_SYSTEM_PROMPT, build_qa_prompt
from latentlens.llm.reasoner import QAReasoner
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan, GroundedAnswer, SearchHit

WEAK_EVIDENCE_REFUSAL = (
    "The retrieved video clips do not provide enough evidence to answer this question."
)


class GroundedQAPipeline:
    """Retrieves top-k clips, checks evidence strength, and synthesizes a cited answer."""

    def __init__(
        self,
        search_service: SearchService,
        reasoner: QAReasoner,
        *,
        top_k: int = 5,
        min_score: float = 0.05,
        system_prompt: str = QA_SYSTEM_PROMPT,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._search = search_service
        self._reasoner = reasoner
        self.top_k = top_k
        self.min_score = min_score
        self.system_prompt = system_prompt

    def answer(self, query: str, *, top_k: int | None = None) -> GroundedAnswer:
        """Answer ``query`` grounded in retrieved clips, refusing when evidence is weak."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query must not be empty")

        k = top_k if top_k is not None else self.top_k
        hits: list[SearchHit] = self._search.search_by_text(cleaned, k=k)
        if not hits or max(h.score for h in hits) < self.min_score:
            return GroundedAnswer(
                query=cleaned,
                answer=WEAK_EVIDENCE_REFUSAL,
                citations=(),
                hits=tuple(hits),
            )

        prompt = build_qa_prompt(cleaned, hits)
        raw_answer = self._reasoner.generate(prompt, system=self.system_prompt)
        candidate_spans = _augment_candidate_spans([h.span for h in hits])
        citations = verify_citations(raw_answer, candidate_spans)
        return GroundedAnswer(
            query=cleaned,
            answer=raw_answer,
            citations=citations,
            hits=tuple(hits),
        )


def _augment_candidate_spans(spans: list[ClipSpan]) -> list[ClipSpan]:
    """Allow citations by either ``video_id`` or video filename stem/basename."""
    from pathlib import Path

    expanded: list[ClipSpan] = list(spans)
    for s in spans:
        basename = Path(s.video_path).name
        if basename != s.video_id:
            expanded.append(ClipSpan(basename, s.video_path, s.start_sec, s.end_sec))
    return expanded


__all__ = [
    "WEAK_EVIDENCE_REFUSAL",
    "GroundedQAPipeline",
]
