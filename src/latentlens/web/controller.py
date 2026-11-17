"""Backend service controller bridging index, search, QA, and video file lookup."""

from __future__ import annotations

from pathlib import Path

from latentlens.agent.harness import VideoAgentHarness
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.qa import GroundedQAPipeline
from latentlens.pipeline.search import SearchService
from latentlens.web.schemas import (
    AskRequest,
    AskResponse,
    IndexStatsResponse,
    SearchHitSchema,
)


class WebController:
    """Facade coordinating index inspection, search, QA, and video file resolution."""

    def __init__(
        self,
        store: VectorStore,
        captions: CaptionStore | None = None,
        search_service: SearchService | None = None,
        qa_pipeline: GroundedQAPipeline | None = None,
        agent_harness: VideoAgentHarness | None = None,
    ) -> None:
        self._store = store
        self._captions = captions
        self._search = search_service
        self._qa = qa_pipeline
        self._harness = agent_harness

    def get_stats(self) -> IndexStatsResponse:
        videos = [
            {"video_id": vid, "path": path, "clips": count}
            for vid, (path, count) in sorted(self._store.videos().items())
        ]
        return IndexStatsResponse(
            encoder_name=self._store.encoder_name,
            dim=self._store.dim,
            total_clips=len(self._store),
            total_captions=len(self._captions) if self._captions is not None else 0,
            videos=videos,
        )

    def search_text(self, query: str, top_k: int = 5) -> list[SearchHitSchema]:
        if self._search is None:
            raise RuntimeError("SearchService is not configured")
        hits = self._search.search_by_text(query.strip(), k=max(1, top_k))
        return [SearchHitSchema.from_hit(h) for h in hits]

    def ask(self, request: AskRequest) -> AskResponse:
        if request.mode == "agent" and self._harness is not None:
            grounded, trace = self._harness.run(request.question)
            return AskResponse.from_grounded(grounded, trace=trace.to_dict())
        if self._qa is not None:
            grounded = self._qa.answer(request.question, top_k=request.top_k)
            return AskResponse.from_grounded(grounded)
        raise RuntimeError("Neither AgentHarness nor GroundedQAPipeline is configured")

    def resolve_video_path(self, video_id: str) -> Path | None:
        """Resolve a registered ``video_id`` to an existing file path on disk."""
        info = self._store.videos().get(video_id)
        if info is None:
            return None
        path = Path(info[0])
        return path if path.is_file() else None


__all__ = [
    "WebController",
]
