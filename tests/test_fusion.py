from __future__ import annotations

import pytest

from latentlens.pipeline.fusion import reciprocal_rank_fusion
from latentlens.types import ClipSpan, SearchHit


def test_reciprocal_rank_fusion_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_reciprocal_rank_fusion_combines_ranks() -> None:
    s1 = ClipSpan("v1", "/p/v1.mp4", 0.0, 4.0)
    s2 = ClipSpan("v2", "/p/v2.mp4", 0.0, 4.0)
    s3 = ClipSpan("v3", "/p/v3.mp4", 0.0, 4.0)

    # List 1: s1 ranked 1, s2 ranked 2
    l1 = [
        SearchHit(rank=1, score=0.9, span=s1, caption="c1"),
        SearchHit(rank=2, score=0.8, span=s2, caption="c2"),
    ]
    # List 2: s2 ranked 1, s3 ranked 2
    l2 = [
        SearchHit(rank=1, score=0.95, span=s2, caption="c2"),
        SearchHit(rank=2, score=0.7, span=s3, caption="c3"),
    ]

    fused = reciprocal_rank_fusion([l1, l2], k=3, c=60.0)
    assert len(fused) == 3
    # s2 appears in both lists: (1/62) + (1/61)
    # s1: 1/61
    # s3: 1/62
    assert fused[0].span.video_id == "v2"
    assert fused[0].rank == 1
    assert fused[1].span.video_id == "v1"
    assert fused[2].span.video_id == "v3"


def test_reciprocal_rank_fusion_validations() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        reciprocal_rank_fusion([], k=0)
    with pytest.raises(ValueError, match="c must be positive"):
        reciprocal_rank_fusion([], c=0.0)
