"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from latentlens.types import ClipSpan, SearchHit


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchHit]],
    *,
    k: int = 5,
    c: float = 60.0,
) -> list[SearchHit]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion: score = sum(1 / (c + rank)).

    Args:
        ranked_lists: Sequences of SearchHit results from different retrievers.
        k: Maximum number of fused hits to return.
        c: Smoothing constant (standard default is 60).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if c <= 0:
        raise ValueError("c must be positive")
    if not ranked_lists:
        return []

    scores: dict[str, float] = defaultdict(float)
    span_map: dict[str, ClipSpan] = {}
    caption_map: dict[str, str | None] = {}

    for hit_list in ranked_lists:
        for rank, hit in enumerate(hit_list, start=1):
            key = f"{hit.span.video_id}:{hit.span.start_sec:.3f}:{hit.span.end_sec:.3f}"
            scores[key] += 1.0 / (c + rank)
            if key not in span_map:
                span_map[key] = hit.span
            if hit.caption and key not in caption_map:
                caption_map[key] = hit.caption

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

    return [
        SearchHit(
            rank=i + 1,
            score=scores[key],
            span=span_map[key],
            caption=caption_map.get(key),
        )
        for i, key in enumerate(sorted_keys)
    ]
