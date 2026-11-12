"""Tests for citation parsing and span verification."""

from __future__ import annotations

import pytest

from latentlens.agent.citations import (
    extract_citations,
    parse_timestamp,
    verify_citations,
)
from latentlens.types import ClipSpan


def test_parse_timestamp_formats() -> None:
    assert parse_timestamp("12.5") == pytest.approx(12.5)
    assert parse_timestamp("01:05.0") == pytest.approx(65.0)
    assert parse_timestamp("1:02:03.5") == pytest.approx(3723.5)

    with pytest.raises(ValueError, match="Empty"):
        parse_timestamp("   ")
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_timestamp("1:2:3:4")


def test_extract_citations_supports_dash_variants() -> None:
    text = (
        "The person enters at [kitchen @ 00:04.0–00:08.0] and leaves at "
        "[kitchen @ 00:12.0-00:16.0] or [hallway @ 01:00.0 to 01:04.0]."
    )
    citations = extract_citations(text)
    assert len(citations) == 3
    assert citations[0].video_id == "kitchen"
    assert citations[0].start_sec == pytest.approx(4.0)
    assert citations[0].end_sec == pytest.approx(8.0)
    assert citations[2].video_id == "hallway"
    assert citations[2].start_sec == pytest.approx(60.0)


def test_verify_citations_matches_and_deduplicates_candidates() -> None:
    s1 = ClipSpan("kitchen", "kitchen.mp4", 4.0, 8.0)
    s2 = ClipSpan("kitchen", "kitchen.mp4", 8.0, 12.0)
    s3 = ClipSpan("garage", "garage.mp4", 0.0, 4.0)

    answer = (
        "First seen at [kitchen @ 00:04.0–00:08.0], hallucinated [ attic @ 00:00.0-00:04.0 ], "
        "and repeated [kitchen @ 00:04.0-00:08.0]."
    )
    verified = verify_citations(answer, [s1, s2, s3])
    assert verified == (s1,)
