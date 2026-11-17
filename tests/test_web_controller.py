"""Tests for WebController stats, search, QA dispatch, and video path resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentlens.agent.harness import VideoAgentHarness
from latentlens.agent.model import ModelTurn
from latentlens.agent.tools import build_default_registry
from latentlens.config import Settings
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.qa import GroundedQAPipeline
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan
from latentlens.web.controller import WebController
from latentlens.web.schemas import AskRequest
from tests.conftest import ColorEncoder
from tests.test_agent_harness import ScriptedAgentModel
from tests.test_qa_pipeline import _ScriptedReasoner
from tests.test_text_search import FakeTextEncoder


def test_web_controller_end_to_end(tmp_path: Path, settings: Settings) -> None:
    vid_file = tmp_path / "kitchen.mp4"
    vid_file.write_bytes(b"fake-mp4-bytes")

    store = VectorStore(dim=3, encoder_name="fake")
    span = ClipSpan("kitchen", str(vid_file), 0.0, 4.0)
    store.add([span], np.ones((1, 3), dtype=np.float32))

    captions = CaptionStore(tmp_path / "captions.json")
    captions.add(span, "Person slicing bread on board", "m")

    search = SearchService(
        encoder=ColorEncoder(),
        store=store,
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    qa = GroundedQAPipeline(
        search,
        _ScriptedReasoner("Slicing bread at [kitchen @ 00:00.0–00:04.0]."),
        min_score=0.0,
    )
    harness = VideoAgentHarness(
        model=ScriptedAgentModel([ModelTurn(text="Agent answer [kitchen @ 00:00.0–00:04.0].")]),
        registry=build_default_registry(search, store, captions),
    )

    ctrl = WebController(
        store=store,
        captions=captions,
        search_service=search,
        qa_pipeline=qa,
        agent_harness=harness,
    )

    stats = ctrl.get_stats()
    assert stats.total_clips == 1
    assert stats.total_captions == 1

    hits = ctrl.search_text("slicing bread", top_k=2)
    assert len(hits) == 1
    assert hits[0].clip.video_id == "kitchen"

    agent_resp = ctrl.ask(AskRequest(question="What is happening?", mode="agent"))
    assert "Agent answer" in agent_resp.answer
    assert agent_resp.trace is not None

    rag_resp = ctrl.ask(AskRequest(question="What is happening?", mode="rag"))
    assert "Slicing bread" in rag_resp.answer
    assert len(rag_resp.citations) == 1

    assert ctrl.resolve_video_path("kitchen") == vid_file
    assert ctrl.resolve_video_path("missing") is None

    empty_ctrl = WebController(store=store)
    with pytest.raises(RuntimeError):
        empty_ctrl.search_text("q")
    with pytest.raises(RuntimeError):
        empty_ctrl.ask(AskRequest(question="q"))
