"""Harness tools for video searching and temporal inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latentlens.agent.types import ToolDefinition, ToolParameter, ToolResult
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.search import SearchService


class SearchMomentsTool:
    """Tool allowing an agent to search indexed clips by natural language or visual match."""

    def __init__(self, search_service: SearchService) -> None:
        self._service = search_service

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_moments",
            description=(
                "Search indexed video clips using a natural language query describing "
                "objects, actions, or scenes."
            ),
            parameters=(
                ToolParameter(
                    name="query",
                    type="string",
                    description="Text query describing what to search for in video clips.",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Maximum number of clips to retrieve (default: 5).",
                    required=False,
                    default=5,
                ),
            ),
        )

    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                tool_name="search_moments",
                output="Error: Query cannot be empty.",
                is_error=True,
            )

        hits = self._service.search_by_text(query.strip(), k=max(1, top_k))
        results: list[dict[str, Any]] = []
        for h in hits:
            results.append(
                {
                    "rank": h.rank,
                    "video": Path(h.span.video_path).name,
                    "video_id": h.span.video_id,
                    "timestamp": h.span.label(),
                    "start_sec": h.span.start_sec,
                    "end_sec": h.span.end_sec,
                    "score": round(h.score, 3),
                    "caption": h.caption or "No caption",
                }
            )

        return ToolResult(
            tool_name="search_moments",
            output=results,
            metadata={"count": len(results)},
        )


class GetAdjacentClipsTool:
    """Tool allowing an agent to inspect clips before or after a discovered timestamp."""

    def __init__(self, store: VectorStore, captions: CaptionStore | None = None) -> None:
        self._store = store
        self._captions = captions

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_adjacent_clips",
            description=(
                "Retrieve clips immediately before or after a given timestamp within a video "
                "to track chronological context or sequence of events."
            ),
            parameters=(
                ToolParameter(
                    name="video_id",
                    type="string",
                    description="Identifier of the video.",
                    required=True,
                ),
                ToolParameter(
                    name="timestamp",
                    type="number",
                    description="Anchor timestamp in seconds.",
                    required=True,
                ),
                ToolParameter(
                    name="window_sec",
                    type="number",
                    description="Temporal window radius in seconds to look around (default: 8.0).",
                    required=False,
                    default=8.0,
                ),
            ),
        )

    def execute(self, video_id: str, timestamp: float, window_sec: float = 8.0) -> ToolResult:
        all_spans = [s for s in self._store.spans() if s.video_id == video_id]
        if not all_spans:
            return ToolResult(
                tool_name="get_adjacent_clips",
                output=f"No clips found for video_id '{video_id}'.",
                is_error=False,
            )

        min_t = max(0.0, timestamp - window_sec)
        max_t = timestamp + window_sec

        matching = [s for s in all_spans if s.end_sec >= min_t and s.start_sec <= max_t]
        matching.sort(key=lambda s: s.start_sec)

        results: list[dict[str, Any]] = []
        for s in matching:
            cap_text = None
            if self._captions is not None:
                cap = self._captions.get(s)
                if cap:
                    cap_text = cap.text

            results.append(
                {
                    "video": Path(s.video_path).name,
                    "video_id": s.video_id,
                    "timestamp": s.label(),
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "caption": cap_text or "No caption",
                }
            )

        return ToolResult(
            tool_name="get_adjacent_clips",
            output=results,
            metadata={"count": len(results)},
        )
