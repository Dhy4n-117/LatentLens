"""Tests for GroundedQAPipeline including weak-evidence refusal and citation matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentlens.config import Settings
from latentlens.index.captions import CaptionStore
from latentlens.index.store import VectorStore
from latentlens.pipeline.qa import WEAK_EVIDENCE_REFUSAL, GroundedQAPipeline
from latentlens.pipeline.search import SearchService
from latentlens.types import ClipSpan
from tests.conftest import ColorEncoder
from tests.test_text_search import FakeTextEncoder


class _ScriptedReasoner:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "scripted"

    def generate(self, prompt: str, *, system: str = "") -> str:
        self.prompts.append(prompt)
        return self.reply


def test_grounded_qa_pipeline_returns_verified_citations(
    tmp_path: Path, settings: Settings
) -> None:
    store = VectorStore(dim=3, encoder_name="fake")
    span = ClipSpan("kitchen", "/videos/kitchen.mp4", 4.0, 8.0)
    store.add([span], np.ones((1, 3), dtype=np.float32))

    captions = CaptionStore(tmp_path / "captions.json")
    captions.add(span, "Person pours coffee into a white cup", "m")

    service = SearchService(
        encoder=ColorEncoder(),
        store=store,
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    reasoner = _ScriptedReasoner("They pour coffee into a cup at [kitchen.mp4 @ 00:04.0–00:08.0].")
    pipeline = GroundedQAPipeline(service, reasoner, min_score=0.0)
    result = pipeline.answer("What is poured into the cup?")

    assert "pour coffee" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].start_sec == 4.0
    assert len(result.hits) == 1


def test_grounded_qa_pipeline_refuses_when_evidence_is_weak(
    tmp_path: Path, settings: Settings
) -> None:
    store = VectorStore(dim=3, encoder_name="fake")
    captions = CaptionStore(tmp_path / "captions.json")
    service = SearchService(
        encoder=ColorEncoder(),
        store=store,
        settings=settings,
        text_encoder=FakeTextEncoder(),
        captions=captions,
    )
    reasoner = _ScriptedReasoner("Should not be called")
    pipeline = GroundedQAPipeline(service, reasoner, min_score=0.5)

    res = pipeline.answer("Did a spaceship land?")
    assert res.answer == WEAK_EVIDENCE_REFUSAL
    assert res.citations == ()
    assert reasoner.prompts == []


def test_grounded_qa_pipeline_validation(settings: Settings) -> None:
    store = VectorStore(dim=3, encoder_name="fake")
    service = SearchService(encoder=ColorEncoder(), store=store, settings=settings)
    reasoner = _ScriptedReasoner("ok")
    with pytest.raises(ValueError, match="top_k"):
        GroundedQAPipeline(service, reasoner, top_k=0)

    pipeline = GroundedQAPipeline(service, reasoner)
    with pytest.raises(ValueError, match="empty"):
        pipeline.answer("   ")
