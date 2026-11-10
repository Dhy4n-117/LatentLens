from __future__ import annotations

from pathlib import Path

from latentlens.agent.tools import GetAdjacentClipsTool, SearchMomentsTool
from latentlens.config import Settings
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan
from tests.conftest import ColorEncoder
from tests.test_text_search import FakeTextEncoder


def test_search_moments_tool(tmp_path: Path, settings: Settings) -> None:
    captions = CaptionStore(tmp_path / "captions.json")
    span = ClipSpan("vid_cooking", "/path/cooking.mp4", 0.0, 4.0)
    captions.add(span, "Person chopping carrots on a cutting board", "model")

    service = SearchService(
        encoder=ColorEncoder(),
        store=VectorStore(dim=3, encoder_name="fake"),
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )

    tool = SearchMomentsTool(service)
    res = tool.execute(query="chopping carrots", top_k=2)

    assert not res.is_error
    assert len(res.output) == 1
    assert res.output[0]["video"] == "cooking.mp4"
    assert "Person chopping carrots" in res.output[0]["caption"]


def test_search_moments_tool_empty_query(tmp_path: Path, settings: Settings) -> None:
    service = SearchService(
        encoder=ColorEncoder(),
        store=VectorStore(dim=3, encoder_name="fake"),
        settings=settings,
    )
    tool = SearchMomentsTool(service)
    res = tool.execute(query="   ")
    assert res.is_error


def test_get_adjacent_clips_tool(tmp_path: Path) -> None:
    store = VectorStore(dim=3, encoder_name="fake")
    s1 = ClipSpan("v1", "/path/v1.mp4", 0.0, 4.0)
    s2 = ClipSpan("v1", "/path/v1.mp4", 4.0, 8.0)
    s3 = ClipSpan("v1", "/path/v1.mp4", 8.0, 12.0)
    s4 = ClipSpan("v2", "/path/v2.mp4", 0.0, 4.0)
    import numpy as np

    store.add([s1, s2, s3, s4], np.ones((4, 3), dtype=np.float32))

    captions = CaptionStore(tmp_path / "captions.json")
    captions.add(s2, "Opening the oven", "model")

    tool = GetAdjacentClipsTool(store, captions)
    res = tool.execute(video_id="v1", timestamp=5.0, window_sec=4.0)

    assert not res.is_error
    # Spans s1 (0-4), s2 (4-8), s3 (8-12) overlap [1.0, 9.0]
    assert len(res.output) >= 2
    assert res.output[1]["caption"] == "Opening the oven"
