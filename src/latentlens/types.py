"""Core domain types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipSpan:
    """A contiguous time range inside a source video."""

    video_id: str
    video_path: str
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def overlaps(self, start_sec: float, end_sec: float) -> bool:
        return self.start_sec < end_sec and start_sec < self.end_sec

    def label(self) -> str:
        return f"{format_timestamp(self.start_sec)}–{format_timestamp(self.end_sec)}"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A ranked retrieval result."""

    rank: int
    score: float
    span: ClipSpan


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS.s`` (or ``H:MM:SS.s`` for long videos)."""
    minutes, secs = divmod(max(seconds, 0.0), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{secs:04.1f}" if hours else f"{minutes:02d}:{secs:04.1f}"
