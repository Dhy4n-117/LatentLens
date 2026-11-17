"""Pydantic schemas for the LatentLens REST API and clip playback UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from latentlens.types import ClipSpan, GroundedAnswer, SearchHit


class ClipPlaybackInfo(BaseModel):
    """Metadata and media-fragment URL for playing a specific video clip window."""

    video_id: str
    video_name: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    label: str
    media_url: str

    @classmethod
    def from_span(cls, span: ClipSpan) -> ClipPlaybackInfo:
        start = round(span.start_sec, 2)
        end = round(span.end_sec, 2)
        return cls(
            video_id=span.video_id,
            video_name=Path(span.video_path).name,
            start_sec=start,
            end_sec=end,
            label=span.label(),
            media_url=f"/api/video/{span.video_id}#t={start},{end}",
        )


class SearchHitSchema(BaseModel):
    """Serialized representation of a ranked ``SearchHit``."""

    rank: int
    score: float
    caption: str | None = None
    clip: ClipPlaybackInfo

    @classmethod
    def from_hit(cls, hit: SearchHit) -> SearchHitSchema:
        return cls(
            rank=hit.rank,
            score=round(hit.score, 4),
            caption=hit.caption,
            clip=ClipPlaybackInfo.from_span(hit.span),
        )


class AskRequest(BaseModel):
    """Payload for asking a question via ``POST /api/ask``."""

    question: str = Field(min_length=1)
    mode: str = Field(default="agent", pattern="^(agent|rag)$")
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    """Response payload for grounded video QA with clickable citations."""

    query: str
    answer: str
    citations: list[ClipPlaybackInfo]
    hits: list[SearchHitSchema]
    trace: dict[str, Any] | None = None

    @classmethod
    def from_grounded(
        cls,
        grounded: GroundedAnswer,
        trace: dict[str, Any] | None = None,
    ) -> AskResponse:
        return cls(
            query=grounded.query,
            answer=grounded.answer,
            citations=[ClipPlaybackInfo.from_span(s) for s in grounded.citations],
            hits=[SearchHitSchema.from_hit(h) for h in grounded.hits],
            trace=trace,
        )


class IndexStatsResponse(BaseModel):
    """Summary of indexed videos and captions."""

    encoder_name: str
    dim: int
    total_clips: int
    total_captions: int
    videos: list[dict[str, Any]]


__all__ = [
    "AskRequest",
    "AskResponse",
    "ClipPlaybackInfo",
    "IndexStatsResponse",
    "SearchHitSchema",
]
