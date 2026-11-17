"""Tests for web API Pydantic schemas and clip playback URL generation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from latentlens.types import ClipSpan, GroundedAnswer, SearchHit
from latentlens.web.schemas import (
    AskRequest,
    AskResponse,
    ClipPlaybackInfo,
    SearchHitSchema,
)


def test_clip_playback_info_and_search_hit_schema() -> None:
    span = ClipSpan("kitchen", "/data/kitchen.mp4", 4.0, 8.0)
    info = ClipPlaybackInfo.from_span(span)
    assert info.video_id == "kitchen"
    assert info.video_name == "kitchen.mp4"
    assert info.media_url == "/api/video/kitchen#t=4.0,8.0"

    hit = SearchHit(rank=1, score=0.87654, span=span, caption="Boiling water")
    schema = SearchHitSchema.from_hit(hit)
    assert schema.rank == 1
    assert schema.score == 0.8765
    assert schema.clip.label == "00:04.0–00:08.0"


def test_ask_request_and_response_serialization() -> None:
    req = AskRequest(question="What happened?", mode="agent", top_k=3)
    assert req.mode == "agent"

    with pytest.raises(ValidationError):
        AskRequest(question="", mode="agent")
    with pytest.raises(ValidationError):
        AskRequest(question="Valid", mode="invalid_mode")

    span = ClipSpan("v1", "/path/v1.mp4", 0.0, 4.0)
    grounded = GroundedAnswer(
        query="What happened?",
        answer="Event at [v1 @ 00:00.0–00:04.0].",
        citations=(span,),
        hits=(SearchHit(rank=1, score=0.9, span=span, caption="c"),),
    )
    resp = AskResponse.from_grounded(grounded, trace={"total_steps": 2})
    dumped = resp.model_dump()
    assert dumped["citations"][0]["media_url"] == "/api/video/v1#t=0.0,4.0"
    assert dumped["trace"] == {"total_steps": 2}
