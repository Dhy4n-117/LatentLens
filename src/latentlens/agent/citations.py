"""Citation parsing and verification against tool-discovered video spans."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from latentlens.types import ClipSpan

_CITATION_PATTERN = re.compile(
    r"\[\s*(?P<video_id>[^\s@\[\]]+)\s*@\s*"
    r"(?P<start>\d+(?::\d{2}){0,2}(?:\.\d+)?)\s*"
    r"(?:[–—\-]|to)\s*"
    r"(?P<end>\d+(?::\d{2}){0,2}(?:\.\d+)?)\s*\]"
)


@dataclass(frozen=True, slots=True)
class ParsedCitation:
    """A timestamp citation extracted from an agent response."""

    video_id: str
    start_sec: float
    end_sec: float
    raw: str


def parse_timestamp(token: str) -> float:
    """Convert ``MM:SS.s``, ``H:MM:SS.s``, or numeric seconds into float seconds."""
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("Empty timestamp token")

    parts = cleaned.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    raise ValueError(f"Unrecognized timestamp format: {token!r}")


def extract_citations(text: str) -> list[ParsedCitation]:
    """Extract all ``[video_id @ start–end]`` citations from ``text``."""
    results: list[ParsedCitation] = []
    for match in _CITATION_PATTERN.finditer(text):
        video_id = match.group("video_id")
        start_sec = parse_timestamp(match.group("start"))
        end_sec = parse_timestamp(match.group("end"))
        if end_sec >= start_sec:
            results.append(
                ParsedCitation(
                    video_id=video_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    raw=match.group(0),
                )
            )
    return results


def verify_citations(
    text: str,
    candidate_spans: Sequence[ClipSpan],
    tolerance_sec: float = 0.5,
) -> tuple[ClipSpan, ...]:
    """Match inline citations in ``text`` against tool-retrieved ``candidate_spans``."""
    parsed = extract_citations(text)
    verified: list[ClipSpan] = []
    seen: set[tuple[str, float, float]] = set()

    for cit in parsed:
        for span in candidate_spans:
            if span.video_id != cit.video_id:
                continue
            strict_overlap = max(span.start_sec, cit.start_sec) < min(span.end_sec, cit.end_sec)
            endpoint_match = (
                abs(span.start_sec - cit.start_sec) <= tolerance_sec
                and abs(span.end_sec - cit.end_sec) <= tolerance_sec
            )
            if strict_overlap or endpoint_match:
                key = (span.video_id, span.start_sec, span.end_sec)
                if key not in seen:
                    seen.add(key)
                    verified.append(span)

    return tuple(verified)


__all__ = [
    "ParsedCitation",
    "extract_citations",
    "parse_timestamp",
    "verify_citations",
]
